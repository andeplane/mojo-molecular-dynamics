from mojo_md import Atoms, NeighborList, PairStyle, Simulation, VelocityVerlet

struct HarmonicPair(PairStyle):
    var k: Float64
    var r0: Float64

    fn __init__(out self, k: Float64, r0: Float64):
        self.k = k
        self.r0 = r0

    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        var pe: Float64 = 0.0
        for i in range(atoms.nlocal):
            for idx in range(nlist.offsets[i], nlist.offsets[i+1]):
                var j = nlist.neighbors[idx]
                var dx = atoms.x[3*j]   - atoms.x[3*i]
                var dy = atoms.x[3*j+1] - atoms.x[3*i+1]
                var dz = atoms.x[3*j+2] - atoms.x[3*i+2]
                var r = (dx*dx + dy*dy + dz*dz) ** 0.5
                var dr = r - self.r0
                var f_mag = -self.k * dr / r
                atoms.f[3*i]   += f_mag * dx
                atoms.f[3*i+1] += f_mag * dy
                atoms.f[3*i+2] += f_mag * dz
                pe += 0.5 * self.k * dr * dr
        return pe * 0.5  # full list double-counts

    fn cutoff(self) -> Float64: return self.r0 + 2.0
    fn short_cutoff(self) -> Float64: return 0.0

fn main():
    var atoms = Atoms(4, 15.0, 15.0, 15.0)
    atoms.x[0] = 2.0; atoms.x[1] = 2.0; atoms.x[2] = 2.0
    atoms.x[3] = 6.0; atoms.x[4] = 2.0; atoms.x[5] = 2.0
    atoms.x[6] = 2.0; atoms.x[7] = 6.0; atoms.x[8] = 2.0
    atoms.x[9] = 6.0; atoms.x[10] = 6.0; atoms.x[11] = 2.0
    for i in range(4):
        atoms.mass[i] = 1.0
        atoms.type_id[i] = 0
        atoms.tag[i] = i

    var pair = HarmonicPair(1.0, 4.0)
    var integrator = VelocityVerlet()
    var sim = Simulation[HarmonicPair, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.001, skin=0.5, rebuild_interval=10,
    )

    for _ in range(50):
        _ = sim.step()

    print("custom_pair demo complete. Final KE =", sim.atoms.kinetic_energy())
