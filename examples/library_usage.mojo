from mojo_md import (
    Atoms, Simulation, PairLJ, VelocityVerlet, init_velocities_mb
)

fn main():
    # 1. Create atoms: 4 argon atoms in a 10 Å box
    var atoms = Atoms(4, 10.0, 10.0, 10.0)
    atoms.x[0] = 1.0; atoms.x[1] = 1.0; atoms.x[2] = 1.0
    atoms.x[3] = 5.0; atoms.x[4] = 1.0; atoms.x[5] = 1.0
    atoms.x[6] = 1.0; atoms.x[7] = 5.0; atoms.x[8] = 1.0
    atoms.x[9] = 5.0; atoms.x[10] = 5.0; atoms.x[11] = 5.0
    for i in range(4):
        atoms.mass[i] = 39.948  # argon in amu
        atoms.type_id[i] = 0
        atoms.tag[i] = i

    # 2. Configure LJ potential (ε=0.0104 eV, σ=3.4 Å, cutoff=8.5 Å)
    var pair = PairLJ(1)  # 1 atom type
    pair.set_pair(0, 0, 0.01040, 3.4, 8.5)

    # 3. Initialize Maxwell-Boltzmann velocities at T ≈ 94 K
    init_velocities_mb(atoms, 0.0081)

    # 4. Build simulation
    var integrator = VelocityVerlet()
    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.002
    )

    # 5. Run 1000 steps (prints thermo every 100 steps)
    sim.run(1000, print_interval=100)
    print("Final KE =", sim.atoms.kinetic_energy())
