from mojo_md import Atoms, VelocityVerlet, PairLJ, PairVashishta, VashishtaParam, init_velocities_mb, Simulation
from mojo_md.sim_io import run_from_file
from mojo_md.pair_vashishta import make_vashishta_param
from mojo_md.simulation import SimulationGPU
from std.gpu.host import DeviceContext
from std.sys import argv, has_accelerator


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

fn _has_flag(args: List[String], flag: String) -> Bool:
    for i in range(len(args)):
        if args[i] == flag:
            return True
    return False


fn _flag_value(args: List[String], flag: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == flag:
            return args[i + 1]
    return ""


# ---------------------------------------------------------------------------
# Atom / pair builders shared by CPU and GPU demos
# ---------------------------------------------------------------------------

fn _build_argon_atoms() -> Atoms:
    var a: Float64 = 5.405
    var nx = 3; var ny = 3; var nz = 3
    var lx = Float64(nx) * a
    var ly = Float64(ny) * a
    var lz = Float64(nz) * a
    var fcc_bx = List[Float64](); var fcc_by = List[Float64](); var fcc_bz = List[Float64]()
    fcc_bx.append(0.0); fcc_by.append(0.0); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.5); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.0); fcc_bz.append(0.5)
    fcc_bx.append(0.0); fcc_by.append(0.5); fcc_bz.append(0.5)
    var nlocal = nx * ny * nz * 4
    var atoms = Atoms(nlocal, lx, ly, lz)
    var idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in range(4):
                    atoms.x[3*idx]   = (Float64(ix) + fcc_bx[b]) * a
                    atoms.x[3*idx+1] = (Float64(iy) + fcc_by[b]) * a
                    atoms.x[3*idx+2] = (Float64(iz) + fcc_bz[b]) * a
                    atoms.mass[idx]    = 39.948
                    atoms.type_id[idx] = 0
                    atoms.tag[idx]     = idx
                    idx += 1
    return atoms^


fn _build_sio2_atoms() -> Atoms:
    var lx: Float64 = 9.28; var ly: Float64 = 9.28; var lz: Float64 = 10.22
    var nlocal = 9
    var atoms = Atoms(nlocal, lx, ly, lz)
    var pos = List[Float64]()
    pos.append(0.465*lx); pos.append(0.000*ly); pos.append(0.000*lz)
    pos.append(0.000*lx); pos.append(0.465*ly); pos.append(0.333*lz)
    pos.append(0.535*lx); pos.append(0.535*ly); pos.append(0.667*lz)
    pos.append(0.413*lx); pos.append(0.268*ly); pos.append(0.119*lz)
    pos.append(0.268*lx); pos.append(0.413*ly); pos.append(0.881*lz)
    pos.append(0.732*lx); pos.append(0.868*ly); pos.append(0.452*lz)
    pos.append(0.868*lx); pos.append(0.732*ly); pos.append(0.548*lz)
    pos.append(0.132*lx); pos.append(0.587*ly); pos.append(0.786*lz)
    pos.append(0.587*lx); pos.append(0.132*ly); pos.append(0.215*lz)
    for i in range(nlocal):
        atoms.x[3*i]   = pos[3*i]
        atoms.x[3*i+1] = pos[3*i+1]
        atoms.x[3*i+2] = pos[3*i+2]
        atoms.tag[i] = i
        if i < 3:
            atoms.type_id[i] = 0; atoms.mass[i] = 28.086
        else:
            atoms.type_id[i] = 1; atoms.mass[i] = 15.999
    return atoms^


fn _make_lj_pair() -> PairLJ:
    var pair = PairLJ(1)
    pair.set_pair(0, 0, 0.01040, 3.4, 8.5)
    return pair^


fn _make_vashishta_pair() -> PairVashishta:
    var pair = PairVashishta(2)
    var p_sisi = make_vashishta_param(bigh=0.82023, eta=11.0, zi=1.6,  zj=1.6,  lambda1=999.0, bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.0, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_sioo = make_vashishta_param(bigh=188.0,  eta=9.0,  zi=1.6,  zj=-0.8, lambda1=10.0,  bigd=1.245, lambda4=4.43,  bigw=22.1179, cut=5.5, bigb=4.7325, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.777)
    var p_osio = make_vashishta_param(bigh=188.0,  eta=9.0,  zi=-0.8, zj=1.6,  lambda1=10.0,  bigd=1.245, lambda4=4.43,  bigw=22.1179, cut=5.5, bigb=19.972, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.333)
    var p_ooo  = make_vashishta_param(bigh=88.0,   eta=7.0,  zi=-0.8, zj=-0.8, lambda1=10.0,  bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.5, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_zero = make_vashishta_param(bigh=0.0,    eta=9.0,  zi=0.0,  zj=0.0,  lambda1=999.0, bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.0, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    pair.set_param(0, 0, 0, p_sisi)
    pair.set_param(0, 0, 1, p_zero)
    pair.set_param(0, 1, 0, p_zero)
    pair.set_param(0, 1, 1, p_sioo)
    pair.set_param(1, 0, 0, p_zero)
    pair.set_param(1, 0, 1, p_osio)
    pair.set_param(1, 1, 0, p_zero)
    pair.set_param(1, 1, 1, p_ooo)
    return pair^


# ---------------------------------------------------------------------------
# CPU demos
# ---------------------------------------------------------------------------

fn demo_argon():
    """108 Ar atoms on FCC lattice, LJ potential."""
    print("=== Demo: Argon (Lennard-Jones) ===")
    var atoms = _build_argon_atoms()
    var pair  = _make_lj_pair()
    init_velocities_mb(atoms, 0.0081)
    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(), dt=0.002, skin=0.3, rebuild_interval=10,
    )
    sim.run(1000, print_interval=100)
    print("Argon demo complete. Final KE =", sim.atoms.kinetic_energy())


fn demo_sio2():
    """9-atom SiO₂ cluster, Vashishta potential."""
    print("=== Demo: SiO₂ (Vashishta) ===")
    var atoms = _build_sio2_atoms()
    var pair  = _make_vashishta_pair()
    init_velocities_mb(atoms, 0.01)
    var sim = Simulation[PairVashishta, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(), dt=0.001, skin=0.3, rebuild_interval=5,
    )
    sim.run(200, print_interval=50)
    print("SiO₂ demo complete. Final KE =", sim.atoms.kinetic_energy())


fn demo_argon_gpu() raises:
    """108 Ar atoms on FCC lattice, LJ potential — GPU path."""
    # comptime if False: Mojo 0.26.2 Metal backend crashes on any GPU kernel.
    # Switch to `comptime if has_accelerator()` on NVIDIA CUDA hardware.
    comptime if False:
        print("=== Demo: Argon (Lennard-Jones) [GPU] ===")
        var atoms = _build_argon_atoms()
        var pair  = _make_lj_pair()
        init_velocities_mb(atoms, 0.0081)
        var ctx = DeviceContext()
        var sim = SimulationGPU[PairLJ, VelocityVerlet](
            atoms^, pair^, VelocityVerlet(), ctx^,
            dt=0.002, skin=0.3, rebuild_interval=10,
        )
        sim.run(1000, print_interval=100)
        print("Argon GPU demo complete.")
    else:
        print("[mojo-md] GPU demos require NVIDIA CUDA (Mojo 0.26.2 Metal unsupported).")


fn demo_sio2_gpu() raises:
    """9-atom SiO₂ cluster, Vashishta potential — GPU path."""
    comptime if False:
        print("=== Demo: SiO₂ (Vashishta) [GPU] ===")
        var atoms = _build_sio2_atoms()
        var pair  = _make_vashishta_pair()
        init_velocities_mb(atoms, 0.01)
        var ctx = DeviceContext()
        var sim = SimulationGPU[PairVashishta, VelocityVerlet](
            atoms^, pair^, VelocityVerlet(), ctx^,
            dt=0.001, skin=0.3, rebuild_interval=5,
        )
        sim.run(200, print_interval=50)
        print("SiO₂ GPU demo complete.")
    else:
        print("[mojo-md] GPU demos require NVIDIA CUDA (Mojo 0.26.2 Metal unsupported).")


# ---------------------------------------------------------------------------
# Entry point
#   mojo run main.mojo -in <config.json>
#   mojo run main.mojo --demo lj
#   mojo run main.mojo --demo vashishta
#   mojo run main.mojo --gpu --demo lj
#   mojo run main.mojo --gpu --demo vashishta
#   mojo run main.mojo --gpu -in <config.json>
#   mojo build main.mojo -o mojo-md  →  ./mojo-md -in config.json
# ---------------------------------------------------------------------------

fn main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    var use_gpu = _has_flag(args, "--gpu")
    var demo    = _flag_value(args, "--demo")
    var in_path = _flag_value(args, "-in")

    comptime if has_accelerator():
        if not use_gpu:
            print("[mojo-md] WARNING: A GPU was detected. For significantly better performance, add the --gpu flag.")

        if use_gpu:
            if demo == "lj":
                demo_argon_gpu()
                return
            elif demo == "vashishta":
                demo_sio2_gpu()
                return
            elif demo != "":
                print("Unknown demo:", demo, "(choices: lj, vashishta)")
                return
            elif in_path != "":
                # GPU file-driven runs are not yet wired up; fall back to CPU
                # config loader. (Future: GPU run_from_file.)
                run_from_file(in_path)
                return
            else:
                print("Usage:")
                print("  mojo run main.mojo --gpu --demo lj")
                print("  mojo run main.mojo --gpu --demo vashishta")
                return
    else:
        if use_gpu:
            print("[mojo-md] ERROR: --gpu requested but no compatible GPU was detected.")
            return

    # CPU path
    if demo == "lj":
        demo_argon()
    elif demo == "vashishta":
        demo_sio2()
    elif demo != "":
        print("Unknown demo:", demo, "(choices: lj, vashishta)")
    elif in_path != "":
        run_from_file(in_path)
    else:
        print("Usage:")
        print("  mojo run main.mojo -in <config.json>")
        print("  mojo run main.mojo --demo lj")
        print("  mojo run main.mojo --demo vashishta")
        print("  mojo run main.mojo --gpu --demo lj")
        print("  mojo run main.mojo --gpu --demo vashishta")
        print("  mojo run main.mojo --gpu -in <config.json>")
        print("  mojo build main.mojo -o mojo-md  →  ./mojo-md -in config.json")
