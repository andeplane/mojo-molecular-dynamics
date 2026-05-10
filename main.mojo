from atom import Atoms
from integrator import VelocityVerlet
from sim_io import load_input
from pair_lj import PairLJ
from pair_vashishta import PairVashishta, VashishtaParam
from random_utils import init_velocities_mb
from simulation import Simulation
from sys import argv


# ---------------------------------------------------------------------------
# JSON-driven entry point
# ---------------------------------------------------------------------------

fn run_from_file(path: String) raises:
    """Load and run a simulation defined by a JSON config file."""
    var inp = load_input(path)
    var intg = VelocityVerlet()
    if inp.pair_style == "lj":
        var sim = Simulation[PairLJ, VelocityVerlet](
            inp.atoms^, inp.lj_pair^, intg^,
            dt=inp.dt, skin=inp.skin, rebuild_interval=inp.rebuild_interval,
        )
        sim.run(inp.nsteps, inp.print_interval)
    elif inp.pair_style == "vashishta":
        var sim = Simulation[PairVashishta, VelocityVerlet](
            inp.atoms^, inp.vashishta_pair^, intg^,
            dt=inp.dt, skin=inp.skin, rebuild_interval=inp.rebuild_interval,
        )
        sim.run(inp.nsteps, inp.print_interval)
    else:
        raise Error("Unknown pair_style: " + inp.pair_style)


# ---------------------------------------------------------------------------
# Built-in demos (no JSON file needed)
# ---------------------------------------------------------------------------

fn demo_argon():
    """108 Ar atoms on FCC lattice, LJ potential."""
    print("=== Demo: Argon (Lennard-Jones) ===")

    var a: Float64 = 5.405
    var nx = 3; var ny = 3; var nz = 3
    var lx = Float64(nx) * a;  var ly = Float64(ny) * a;  var lz = Float64(nz) * a
    var n_basis = 4
    var nlocal = nx * ny * nz * n_basis

    var fcc_bx = List[Float64](); var fcc_by = List[Float64](); var fcc_bz = List[Float64]()
    fcc_bx.append(0.0); fcc_by.append(0.0); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.5); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.0); fcc_bz.append(0.5)
    fcc_bx.append(0.0); fcc_by.append(0.5); fcc_bz.append(0.5)

    var atoms = Atoms(nlocal, lx, ly, lz)
    var ar_mass: Float64 = 39.948
    var idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in range(n_basis):
                    atoms.x[3 * idx]     = (Float64(ix) + fcc_bx[b]) * a
                    atoms.x[3 * idx + 1] = (Float64(iy) + fcc_by[b]) * a
                    atoms.x[3 * idx + 2] = (Float64(iz) + fcc_bz[b]) * a
                    atoms.mass[idx] = ar_mass
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
    sim.run(1000, print_interval=100)
    print("Argon demo complete. Final KE =", sim.atoms.kinetic_energy())


fn demo_sio2():
    """9-atom SiO₂ cluster, Vashishta potential."""
    print("=== Demo: SiO₂ (Vashishta) ===")

    var lx: Float64 = 9.28;  var ly: Float64 = 9.28;  var lz: Float64 = 10.22
    var nlocal = 9
    var atoms = Atoms(nlocal, lx, ly, lz)
    var si_mass: Float64 = 28.086;  var o_mass: Float64 = 15.999

    var pos_x = List[Float64](); var pos_y = List[Float64](); var pos_z = List[Float64]()
    pos_x.append(0.465*lx); pos_y.append(0.000*ly); pos_z.append(0.000*lz)
    pos_x.append(0.000*lx); pos_y.append(0.465*ly); pos_z.append(0.333*lz)
    pos_x.append(0.535*lx); pos_y.append(0.535*ly); pos_z.append(0.667*lz)
    pos_x.append(0.413*lx); pos_y.append(0.268*ly); pos_z.append(0.119*lz)
    pos_x.append(0.268*lx); pos_y.append(0.413*ly); pos_z.append(0.881*lz)
    pos_x.append(0.732*lx); pos_y.append(0.868*ly); pos_z.append(0.452*lz)
    pos_x.append(0.868*lx); pos_y.append(0.732*ly); pos_z.append(0.548*lz)
    pos_x.append(0.132*lx); pos_y.append(0.587*ly); pos_z.append(0.786*lz)
    pos_x.append(0.587*lx); pos_y.append(0.132*ly); pos_z.append(0.215*lz)

    for i in range(nlocal):
        atoms.x[3*i] = pos_x[i]; atoms.x[3*i+1] = pos_y[i]; atoms.x[3*i+2] = pos_z[i]
        atoms.tag[i] = i
        if i < 3: atoms.type_id[i] = 0; atoms.mass[i] = si_mass
        else:     atoms.type_id[i] = 1; atoms.mass[i] = o_mass

    var pair = PairVashishta(2)
    var p_sisi = VashishtaParam(bigh=0.82023, eta=11.0, zi=1.6, zj=1.6, lambda1=999.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.0, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_sioo = VashishtaParam(bigh=188.0, eta=9.0, zi=1.6, zj=-0.8, lambda1=10.0, bigd=1.245, lambda4=4.43, bigw=22.1179, cut=5.5, bigb=4.7325, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.777)
    var p_osio = VashishtaParam(bigh=188.0, eta=9.0, zi=-0.8, zj=1.6, lambda1=10.0, bigd=1.245, lambda4=4.43, bigw=22.1179, cut=5.5, bigb=19.972, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.333)
    var p_ooo  = VashishtaParam(bigh=88.0, eta=7.0, zi=-0.8, zj=-0.8, lambda1=10.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.5, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_zero = VashishtaParam(bigh=0.0, eta=9.0, zi=0.0, zj=0.0, lambda1=999.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.0, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    pair.set_param(0, 0, 0, p_sisi)
    pair.set_param(0, 0, 1, p_zero)
    pair.set_param(0, 1, 0, p_zero)
    pair.set_param(0, 1, 1, p_sioo)
    pair.set_param(1, 0, 0, p_zero)
    pair.set_param(1, 0, 1, p_osio)
    pair.set_param(1, 1, 0, p_zero)
    pair.set_param(1, 1, 1, p_ooo)

    init_velocities_mb(atoms, 0.01)
    var integrator = VelocityVerlet()
    var sim = Simulation[PairVashishta, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.001, skin=0.3, rebuild_interval=5,
    )
    sim.run(200, print_interval=50)
    print("SiO₂ demo complete. Final KE =", sim.atoms.kinetic_energy())


# ---------------------------------------------------------------------------
# Entry point: `mojo run main.mojo [path/to/config.json]`
# ---------------------------------------------------------------------------

fn main() raises:
    var args = argv()
    if len(args) > 1:
        run_from_file(args[1])
    else:
        demo_argon()
        print()
        demo_sio2()
