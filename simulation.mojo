from atom import Atoms
from ghost import GhostBuilder
from integrator import Integrator
from neighbor import NeighborList
from pair_style import PairStyle


struct Simulation[P: PairStyle, I: Integrator]:
    """
    Generic MD simulation driver.

    Template parameters P (pair style) and I (integrator) are resolved at
    compile time, giving the compiler full visibility into the hot path for
    inlining and auto-vectorization.

    The integration loop follows LAMMPS's Verlet structure:
        initial half-step v  →  full step x
        [rebuild ghosts + nlist if needed]
        zero forces  →  pair.compute()
        final half-step v
        [output]

    Ghost atoms are rebuilt at the same interval as the neighbor list.
    After force evaluation, ghost forces are reverse-communicated to their
    source real atoms (see GhostBuilder.reverse_comm).
    """
    var atoms: Atoms
    var pair: P
    var integrator: I
    var nlist: NeighborList
    var ghosts: GhostBuilder
    var dt: Float64
    var skin: Float64
    var rebuild_interval: Int
    var step: Int

    fn __moveinit__(out self, deinit other: Simulation[P, I]):
        self.atoms = other.atoms^
        self.pair = other.pair^
        self.integrator = other.integrator^
        self.nlist = other.nlist^
        self.ghosts = other.ghosts^
        self.dt = other.dt
        self.skin = other.skin
        self.rebuild_interval = other.rebuild_interval
        self.step = other.step

    fn __init__(
        out self,
        owned atoms: Atoms,
        owned pair: P,
        owned integrator: I,
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
        self.step = 0

        var rcut = self.pair.cutoff() + skin
        var rcut_short = self.pair.short_cutoff()
        self.ghosts = GhostBuilder(rcut)
        self.nlist = NeighborList(self.atoms.nlocal)

        # Initial setup
        self.ghosts.rebuild_ghosts(self.atoms)
        self.nlist.build(self.atoms, rcut, rcut_short)
        self.atoms.zero_forces()
        _ = self.pair.compute(self.atoms, self.nlist)
        self.ghosts.reverse_comm(self.atoms)

    fn run(mut self, nsteps: Int, print_interval: Int = 100):
        """Advance simulation by nsteps timesteps."""
        var rcut = self.pair.cutoff() + self.skin
        var rcut_short = self.pair.short_cutoff()

        for _ in range(nsteps):
            self.step += 1

            # Half velocity step (uses forces from previous step)
            self.integrator.half_step_v(self.atoms, self.dt)
            # Full position step
            self.integrator.full_step_x(self.atoms, self.dt)

            # Rebuild ghosts and neighbor list periodically
            if self.step % self.rebuild_interval == 0:
                self.ghosts.cutoff_with_skin = rcut
                self.ghosts.rebuild_ghosts(self.atoms)
                self.nlist.build(self.atoms, rcut, rcut_short)

            # Force evaluation
            self.atoms.zero_forces()
            var pe = self.pair.compute(self.atoms, self.nlist)
            self.ghosts.reverse_comm(self.atoms)

            # Second half velocity step (uses new forces)
            self.integrator.half_step_v(self.atoms, self.dt)

            if self.step % print_interval == 0:
                var ke = self.atoms.kinetic_energy()
                print("step", self.step,
                      "PE", pe,
                      "KE", ke,
                      "TE", pe + ke)
