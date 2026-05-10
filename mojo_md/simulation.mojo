from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.ghost import GhostBuilder
from mojo_md.integrator import Integrator
from mojo_md.neighbor import NeighborList
from mojo_md.pair_style import PairStyle

comptime _BLOCK_SIZE: Int = 256


# ---------------------------------------------------------------------------
# Shared kernel bodies — used by both CPU `parallelize` and GPU
# `enqueue_function` paths so the simulation-level helpers (zero forces,
# fold ghost forces back) are written once.
# ---------------------------------------------------------------------------

@always_inline
fn _zero_forces_body(idx: Int, f_ptr: UnsafePointer[Float64, MutAnyOrigin]):
    f_ptr[idx] = 0.0


fn _zero_forces_kernel(f_ptr: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        _zero_forces_body(i, f_ptr)


@always_inline
fn _reverse_comm_body(
    i:           Int,
    nlocal:      Int,
    ntotal:      Int,
    f_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    tag_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
):
    """Owner-computes ghost reduction. Thread for local atom i scans all
    ghosts and folds matching-tag ghost forces into f[i]. No atomics."""
    var i_tag = tag_ptr[i]
    var fx: Float64 = 0.0;  var fy: Float64 = 0.0;  var fz: Float64 = 0.0
    for g in range(nlocal, ntotal):
        if tag_ptr[g] == i_tag:
            fx += f_ptr[3 * g]
            fy += f_ptr[3 * g + 1]
            fz += f_ptr[3 * g + 2]
    f_ptr[3 * i]     += fx
    f_ptr[3 * i + 1] += fy
    f_ptr[3 * i + 2] += fz


fn _reverse_comm_kernel(
    f_ptr:   UnsafePointer[Float64, MutAnyOrigin],
    tag_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    nlocal:  Int,
    ntotal:  Int,
):
    var i = Int(global_idx.x)
    if i < nlocal:
        _reverse_comm_body(i, nlocal, ntotal, f_ptr, tag_ptr)


# ---------------------------------------------------------------------------
# Simulation driver — CPU. Uses the same trait interfaces (PairStyle,
# Integrator) and the same ghost/neighbor list code as the GPU driver below.
# ---------------------------------------------------------------------------

struct Simulation[P: PairStyle, I: Integrator](Movable):
    """
    Generic MD simulation driver (CPU). Template params P (pair style) and I
    (integrator) are resolved at compile time so the compiler can inline and
    auto-vectorize the hot path.

    Integration loop (LAMMPS Verlet):
        half_step_v → full_step_x → [rebuild ghosts/nlist] → zero forces →
        pair.compute() → reverse_comm → half_step_v → [print].
    """
    var atoms: Atoms
    var pair: Self.P
    var integrator: Self.I
    var nlist: NeighborList
    var ghosts: GhostBuilder
    var dt: Float64
    var skin: Float64
    var rebuild_interval: Int
    var step_count: Int

    fn __init__(
        out self,
        var atoms: Atoms,
        var pair: Self.P,
        var integrator: Self.I,
        dt: Float64,
        skin: Float64 = 0.3,
        rebuild_interval: Int = 10,
    ):
        self.atoms = atoms^
        self.pair = pair^
        self.integrator = integrator^
        self.dt = dt
        self.skin = skin
        self.rebuild_interval = rebuild_interval
        self.step_count = 0

        var rcut = self.pair.cutoff() + skin
        var rcut_short = self.pair.short_cutoff()
        self.ghosts = GhostBuilder(rcut)
        self.nlist = NeighborList(self.atoms.nlocal)

        self.ghosts.rebuild_ghosts(self.atoms)
        self.nlist.build(self.atoms, rcut, rcut_short)
        self.atoms.zero_forces()
        _ = self.pair.compute(self.atoms, self.nlist)
        self.ghosts.reverse_comm(self.atoms)

    fn step(mut self) -> Float64:
        """Advance one timestep. Returns potential energy."""
        var rcut = self.pair.cutoff() + self.skin
        var rcut_short = self.pair.short_cutoff()

        self.step_count += 1

        # Half velocity step (uses forces from previous step)
        self.integrator.half_step_v(self.atoms, self.dt)
        # Full position step
        self.integrator.full_step_x(self.atoms, self.dt)

        # Rebuild ghosts and neighbor list periodically
        if self.step_count % self.rebuild_interval == 0:
            self.ghosts.cutoff_with_skin = rcut
            self.ghosts.rebuild_ghosts(self.atoms)
            self.nlist.build(self.atoms, rcut, rcut_short)

        # Force evaluation
        self.atoms.zero_forces()
        var pe = self.pair.compute(self.atoms, self.nlist)
        self.ghosts.reverse_comm(self.atoms)

        # Second half velocity step (uses new forces)
        self.integrator.half_step_v(self.atoms, self.dt)

        return pe

    fn run(mut self, nsteps: Int, print_interval: Int = 100):
        """Advance simulation by nsteps timesteps."""
        for _ in range(nsteps):
            var pe = self.step()
            if self.step_count % print_interval == 0:
                var ke = self.atoms.kinetic_energy()
                print("step", self.step_count,
                      "PE", pe,
                      "KE", ke,
                      "TE", pe + ke)


# ---------------------------------------------------------------------------
# Simulation driver — GPU. Same Verlet loop, same trait calls (just the GPU
# variants), GPU-resident state. The only CPU round-trip per step is when
# the rebuild_interval lands on this step: real-atom positions GPU→CPU, then
# CPU rebuilds ghosts + neighbor list, then ghost data + nlist CSR CPU→GPU.
# Force kernels and integrator share their kernel bodies with the CPU driver.
# ---------------------------------------------------------------------------

struct SimulationGPU[P: PairStyle, I: Integrator](Movable):
    """
    GPU-resident MD simulation driver. Pair style and integrator must implement
    the same traits as the CPU driver — their `compute_gpu` / `*_gpu` methods
    call the same kernel bodies that the CPU paths do (one source of truth
    for the physics).
    """
    var cpu_atoms:        Atoms                          # host-side mirror, used during rebuilds
    var cpu_nlist:        NeighborList
    var ghosts:           GhostBuilder
    var atoms:            GPUAtoms                       # device-side state
    var nlist:            GPUNeighborList
    var pair:             Self.P
    var pair_params_dev:  DeviceBuffer[DType.float64]    # GPU mirror of pair params
    var integrator:       Self.I
    var ctx:              DeviceContext
    var dt:               Float64
    var skin:             Float64
    var rebuild_interval: Int
    var step:             Int

    fn __init__(
        out self,
        var cpu_atoms: Atoms,
        var pair: Self.P,
        var integrator: Self.I,
        var ctx: DeviceContext,
        dt: Float64,
        skin: Float64 = 0.3,
        rebuild_interval: Int = 10,
    ) raises:
        self.dt = dt
        self.skin = skin
        self.rebuild_interval = rebuild_interval
        self.step = 0
        self.pair = pair^
        self.integrator = integrator^
        self.ctx = ctx^

        var rcut = self.pair.cutoff() + skin
        var rcut_short = self.pair.short_cutoff()

        self.ghosts = GhostBuilder(rcut)
        self.cpu_nlist = NeighborList(cpu_atoms.nlocal)
        self.ghosts.rebuild_ghosts(cpu_atoms)
        self.cpu_nlist.build(cpu_atoms, rcut, rcut_short)
        cpu_atoms.zero_forces()
        self.cpu_atoms = cpu_atoms^

        self.atoms = GPUAtoms.from_cpu(self.cpu_atoms, self.ctx)
        self.nlist = GPUNeighborList.from_cpu(self.cpu_nlist, self.ctx)
        self.pair_params_dev = self.pair.make_gpu_params(self.ctx)

        # Initial force evaluation (so the first half_step_v has a valid f).
        self._zero_forces_gpu()
        _ = self.pair.compute_gpu(self.atoms, self.nlist, self.pair_params_dev, self.ctx)
        self._reverse_comm_gpu()

    fn _zero_forces_gpu(mut self) raises:
        var n = 3 * self.atoms.n()
        var n_blocks = ceildiv(n, _BLOCK_SIZE)
        self.ctx.enqueue_function[_zero_forces_kernel, _zero_forces_kernel](
            self.atoms.f.unsafe_ptr(), n,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

    fn _reverse_comm_gpu(mut self) raises:
        var nlocal = self.atoms.nlocal
        if self.atoms.nghost == 0:
            return
        var n_blocks = ceildiv(nlocal, _BLOCK_SIZE)
        self.ctx.enqueue_function[_reverse_comm_kernel, _reverse_comm_kernel](
            self.atoms.f.unsafe_ptr(),
            self.atoms.tag.unsafe_ptr(),
            nlocal,
            self.atoms.n(),
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

    fn run(mut self, nsteps: Int, print_interval: Int = 100) raises:
        var rcut = self.pair.cutoff() + self.skin
        var rcut_short = self.pair.short_cutoff()

        for _ in range(nsteps):
            self.step += 1

            self.integrator.half_step_v_gpu(self.atoms, self.ctx, self.dt)
            self.integrator.full_step_x_gpu(self.atoms, self.ctx, self.dt)

            if self.step % self.rebuild_interval == 0:
                # The one allowed CPU round-trip: pull positions, rebuild
                # ghosts + neighbor list on CPU, push back updated bookkeeping.
                self.atoms.read_positions_to_cpu(self.cpu_atoms, self.ctx)
                self.ghosts.cutoff_with_skin = rcut
                self.ghosts.rebuild_ghosts(self.cpu_atoms)
                self.cpu_nlist.build(self.cpu_atoms, rcut, rcut_short)
                self.atoms.refresh_from_cpu(self.cpu_atoms, self.ctx)
                self.nlist.refresh_from_cpu(self.cpu_nlist, self.ctx)

            self._zero_forces_gpu()
            var pe = self.pair.compute_gpu(self.atoms, self.nlist, self.pair_params_dev, self.ctx)
            self._reverse_comm_gpu()

            self.integrator.half_step_v_gpu(self.atoms, self.ctx, self.dt)

            if self.step % print_interval == 0:
                var ke = self.atoms.read_ke_to_cpu(self.ctx)
                print("step", self.step,
                      "PE", pe,
                      "KE", ke,
                      "TE", pe + ke)
