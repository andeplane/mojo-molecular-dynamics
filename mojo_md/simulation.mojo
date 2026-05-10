from std.algorithm import parallelize
from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.ghost import GhostBuilder
from mojo_md.gpu_rebuild import (
    wrap_into_box_gpu, rebuild_ghosts_gpu, nlist_build_gpu,
    MAX_NEIGHBORS_PER_ATOM, MAX_SHORT_NEIGHBORS_PER_ATOM, MAX_ATOMS_PER_CELL,
)
from mojo_md.integrator import Integrator
from mojo_md.neighbor import NeighborList
from mojo_md.pair_style import PairStyle

comptime _BLOCK_SIZE: Int = 256


# ---------------------------------------------------------------------------
# Shared kernel bodies — used by both CPU `parallelize` and GPU
# `enqueue_function` paths so the simulation-level helpers (zero forces,
# fold ghost forces back) are written once.
# ---------------------------------------------------------------------------

# CPU kernel bodies (Float64) — used by Simulation via Atoms.zero_forces / GhostBuilder.reverse_comm.

# GPU kernels — Float32 for Metal/MPS compatibility.

fn _zero_forces_kernel(f_ptr: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        f_ptr[i] = 0.0


# One thread per ghost. Atomically add the ghost's force to its source atom.
# O(N_ghost) — replaces the previous O(N_local × N_ghost) tag-scan.
fn _reverse_comm_kernel(
    f_ptr:   UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    nlocal:  Int,
    nghost:  Int,
):
    var g = Int(global_idx.x)
    if g >= nghost:
        return
    var gi  = nlocal + g
    var src = Int(src_ptr[g])
    _ = Atomic.fetch_add(f_ptr + 3 * src,     f_ptr[3 * gi])
    _ = Atomic.fetch_add(f_ptr + 3 * src + 1, f_ptr[3 * gi + 1])
    _ = Atomic.fetch_add(f_ptr + 3 * src + 2, f_ptr[3 * gi + 2])


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
    var atoms:            GPUAtoms                       # device-side state (positions, velocities, forces, …)
    var nlist:            GPUNeighborList                # device-side strided neighbor list
    var pair:             Self.P
    var pair_params_dev:  DeviceBuffer[DType.float32]    # GPU mirror of pair params (Float32 for Metal/MPS)
    var integrator:       Self.I
    var ctx:              DeviceContext
    var dt:               Float64
    var skin:             Float64
    var rebuild_interval: Int
    var step:             Int
    # Box dimensions on host (constant for NVE), used as kernel scalar args.
    var lx:               Float32
    var ly:               Float32
    var lz:               Float32
    # Cell list grid (fixed at init).
    var nc_x:             Int
    var nc_y:             Int
    var nc_z:             Int
    var cell_lx:          Float32
    var cell_ly:          Float32
    var cell_lz:          Float32
    # GPU rebuild scratch buffers (allocated once at init, reused every rebuild).
    var cell_count_dev:   DeviceBuffer[DType.int32]      # nc Int32
    var cell_atoms_dev:   DeviceBuffer[DType.int32]      # nc * MAX_ATOMS_PER_CELL Int32
    var nghost_dev:       DeviceBuffer[DType.int32]      # 1 Int32 (atomic counter)
    var overflow_dev:     DeviceBuffer[DType.int32]      # 1 Int32 (flag)

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

        var rcut       = self.pair.cutoff() + skin
        var rcut_short = self.pair.short_cutoff()

        # Box scalars cached on host.
        self.lx = Float32(cpu_atoms.box[0])
        self.ly = Float32(cpu_atoms.box[1])
        self.lz = Float32(cpu_atoms.box[2])

        # Cell list grid sized off initial box + cutoff.
        self.nc_x = max(1, Int(cpu_atoms.box[0] / rcut))
        self.nc_y = max(1, Int(cpu_atoms.box[1] / rcut))
        self.nc_z = max(1, Int(cpu_atoms.box[2] / rcut))
        self.cell_lx = self.lx / Float32(self.nc_x)
        self.cell_ly = self.ly / Float32(self.nc_y)
        self.cell_lz = self.lz / Float32(self.nc_z)
        var nc = self.nc_x * self.nc_y * self.nc_z

        # Grow CPU mirror to give the GPU ghost build enough headroom in nmax.
        # Worst-case ghost count is bounded by atoms-near-boundary × ~7 (corner
        # atoms with 7 ghost copies). For typical orthogonal boxes a 100%
        # headroom is more than enough; for tiny systems use a flat 4096 floor.
        var nlocal = cpu_atoms.nlocal
        var ghost_headroom = max(nlocal, 4096)
        cpu_atoms.grow(nlocal + ghost_headroom)
        cpu_atoms.zero_forces()

        # Upload initial state to GPU (positions, velocities, types, etc.).
        self.atoms = GPUAtoms.from_cpu(cpu_atoms, self.ctx)
        self.nlist = GPUNeighborList.with_capacity(
            nlocal, MAX_NEIGHBORS_PER_ATOM, MAX_SHORT_NEIGHBORS_PER_ATOM, self.ctx,
        )
        self.pair_params_dev = self.pair.make_gpu_params(self.ctx)

        # Rebuild scratch buffers.
        self.cell_count_dev = self.ctx.enqueue_create_buffer[DType.int32](nc)
        self.cell_atoms_dev = self.ctx.enqueue_create_buffer[DType.int32](nc * MAX_ATOMS_PER_CELL)
        self.nghost_dev     = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.overflow_dev   = self.ctx.enqueue_create_buffer[DType.int32](1)

        # First-time GPU rebuild: wrap, ghosts, neighbor list — entirely on GPU.
        self._gpu_rebuild(rcut, rcut_short)

        # Initial force evaluation so the first half_step_v has valid forces.
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
        var nghost = self.atoms.nghost
        if nghost == 0:
            return
        var n_blocks = ceildiv(nghost, _BLOCK_SIZE)
        self.ctx.enqueue_function[_reverse_comm_kernel, _reverse_comm_kernel](
            self.atoms.f.unsafe_ptr(),
            self.atoms.source_idx.unsafe_ptr(),
            self.atoms.nlocal,
            nghost,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

    fn _gpu_rebuild(mut self, rcut: Float64, rcut_short: Float64) raises:
        """Full GPU-resident rebuild: wrap positions, ghosts, cell list, nlist.
        Only a single 4-byte readback (nghost) and 4-byte readback (overflow flag)."""
        wrap_into_box_gpu(self.atoms, self.lx, self.ly, self.lz, self.ctx)
        rebuild_ghosts_gpu(
            self.atoms, self.nghost_dev, self.overflow_dev,
            self.lx, self.ly, self.lz, Float32(rcut), self.ctx,
        )
        nlist_build_gpu(
            self.atoms,
            self.cell_count_dev, self.cell_atoms_dev,
            self.nlist.neighbors, self.nlist.short_neighbors,
            self.overflow_dev,
            self.nc_x, self.nc_y, self.nc_z,
            self.cell_lx, self.cell_ly, self.cell_lz,
            Float32(rcut * rcut),
            Float32(rcut_short * rcut_short),
            self.ctx,
        )

    fn run(mut self, nsteps: Int, print_interval: Int = 100) raises:
        var rcut = self.pair.cutoff() + self.skin
        var rcut_short = self.pair.short_cutoff()

        for _ in range(nsteps):
            self.step += 1

            self.integrator.half_step_v_gpu(self.atoms, self.ctx, self.dt)
            self.integrator.full_step_x_gpu(self.atoms, self.ctx, self.dt)

            if self.step % self.rebuild_interval == 0:
                self._gpu_rebuild(rcut, rcut_short)

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
