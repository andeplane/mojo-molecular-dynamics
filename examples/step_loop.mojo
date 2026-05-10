from mojo_md import (
    Atoms, Simulation, PairLJ, VelocityVerlet, init_velocities_mb
)

fn main():
    var a: Float64 = 5.405
    var nx = 3; var ny = 3; var nz = 3
    var lx = Float64(nx) * a; var ly = Float64(ny) * a; var lz = Float64(nz) * a
    var nlocal = nx * ny * nz * 4

    var fcc_bx = List[Float64](); var fcc_by = List[Float64](); var fcc_bz = List[Float64]()
    fcc_bx.append(0.0); fcc_by.append(0.0); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.5); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.0); fcc_bz.append(0.5)
    fcc_bx.append(0.0); fcc_by.append(0.5); fcc_bz.append(0.5)

    var atoms = Atoms(nlocal, lx, ly, lz)
    var idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in range(4):
                    atoms.x[3*idx]   = (Float64(ix) + fcc_bx[b]) * a
                    atoms.x[3*idx+1] = (Float64(iy) + fcc_by[b]) * a
                    atoms.x[3*idx+2] = (Float64(iz) + fcc_bz[b]) * a
                    atoms.mass[idx] = 39.948
                    atoms.type_id[idx] = 0
                    atoms.tag[idx] = idx
                    idx += 1

    var pair = PairLJ(1)
    pair.set_pair(0, 0, 0.01040, 3.4, 8.5)
    init_velocities_mb(atoms, 0.0081)

    var integrator = VelocityVerlet()
    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.002, skin=0.3, rebuild_interval=10,
    )

    for _ in range(100):
        var pe = sim.step()
        var ke = sim.atoms.kinetic_energy()
        print("PE =", pe, "KE =", ke, "TE =", pe + ke)
