"""Targeted performance measurement.

LJ:        10M atoms  — CPU (3 steps, extrapolated) + GPU (100 steps)
Vashishta: 128k atoms — CPU (20 steps) + GPU (100 steps)

Reports Matom·steps/s for each run.
"""
from mojo_md.atom import Atoms
from mojo_md.integrator import VelocityVerlet
from mojo_md.pair_lj import PairLJ
from mojo_md.pair_vashishta import PairVashishta, make_vashishta_param
from mojo_md.simulation import Simulation, SimulationGPU
from std.sys import has_accelerator
from time import perf_counter_ns


fn _build_fcc_atoms(n_target: Int) -> Atoms:
    var a: Float64 = 5.405
    var nx = max(1, Int(Float64(n_target // 4) ** (1.0 / 3.0)))
    while nx * nx * nx * 4 < n_target:
        nx += 1
    var nlocal = nx * nx * nx * 4
    var lx = Float64(nx) * a
    var bx = List[Float64](); var by = List[Float64](); var bz = List[Float64]()
    bx.append(0.0); by.append(0.0); bz.append(0.0)
    bx.append(0.5); by.append(0.5); bz.append(0.0)
    bx.append(0.5); by.append(0.0); bz.append(0.5)
    bx.append(0.0); by.append(0.5); bz.append(0.5)
    var atoms = Atoms(nlocal, lx, lx, lx)
    var idx = 0
    for ix in range(nx):
        for iy in range(nx):
            for iz in range(nx):
                for b in range(4):
                    atoms.x[3*idx]   = (Float64(ix) + bx[b]) * a
                    atoms.x[3*idx+1] = (Float64(iy) + by[b]) * a
                    atoms.x[3*idx+2] = (Float64(iz) + bz[b]) * a
                    atoms.mass[idx] = 39.948; atoms.type_id[idx] = 0; atoms.tag[idx] = idx
                    idx += 1
    return atoms^


fn _build_sio2_random(n_target: Int) -> Atoms:
    var n_si = n_target // 3; var n_o = n_target - n_si
    var nlocal = n_si + n_o
    var nc = max(1, Int(Float64(nlocal) ** (1.0 / 3.0)) + 1)
    var a: Float64 = 3.1; var lx = Float64(nc) * a
    var atoms = Atoms(nlocal, lx, lx, lx)
    var idx = 0
    for ix in range(nc):
        if idx >= nlocal: break
        for iy in range(nc):
            if idx >= nlocal: break
            for iz in range(nc):
                if idx >= nlocal: break
                atoms.x[3*idx]   = (Float64(ix) + 0.05) * a
                atoms.x[3*idx+1] = (Float64(iy) + 0.05) * a
                atoms.x[3*idx+2] = (Float64(iz) + 0.05) * a
                atoms.tag[idx] = idx
                if idx < n_si:
                    atoms.type_id[idx] = 0; atoms.mass[idx] = 28.086
                else:
                    atoms.type_id[idx] = 1; atoms.mass[idx] = 15.999
                idx += 1
    return atoms^


fn _make_lj_pair() -> PairLJ:
    var pair = PairLJ(1); pair.set_pair(0, 0, 0.01040, 3.4, 8.5); return pair^


fn _make_vashishta_pair() -> PairVashishta:
    var pair = PairVashishta(2)
    var p_sisi = make_vashishta_param(bigh=0.82023, eta=11.0, zi=1.6,  zj=1.6,  lambda1=999.0, bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.0, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_sioo = make_vashishta_param(bigh=188.0,  eta=9.0,  zi=1.6,  zj=-0.8, lambda1=10.0,  bigd=1.245, lambda4=4.43,  bigw=22.1179, cut=5.5, bigb=4.7325, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.777)
    var p_osio = make_vashishta_param(bigh=188.0,  eta=9.0,  zi=-0.8, zj=1.6,  lambda1=10.0,  bigd=1.245, lambda4=4.43,  bigw=22.1179, cut=5.5, bigb=19.972, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.333)
    var p_ooo  = make_vashishta_param(bigh=88.0,   eta=7.0,  zi=-0.8, zj=-0.8, lambda1=10.0,  bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.5, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_zero = make_vashishta_param(bigh=0.0,    eta=9.0,  zi=0.0,  zj=0.0,  lambda1=999.0, bigd=0.0,   lambda4=999.0, bigw=0.0,     cut=5.0, bigb=0.0,    gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    pair.set_param(0, 0, 0, p_sisi); pair.set_param(0, 0, 1, p_zero)
    pair.set_param(0, 1, 0, p_zero); pair.set_param(0, 1, 1, p_sioo)
    pair.set_param(1, 0, 0, p_zero); pair.set_param(1, 0, 1, p_osio)
    pair.set_param(1, 1, 0, p_zero); pair.set_param(1, 1, 1, p_ooo)
    return pair^


fn _report(label: String, n: Int, steps: Int, elapsed_ns: Int):
    var elapsed = Float64(elapsed_ns) / 1.0e9
    var matom_s = Float64(n) * Float64(steps) / elapsed / 1.0e6
    print(label, "  N =", n, " steps =", steps,
          " elapsed =", elapsed, "s  =>", matom_s, "Matom·steps/s")


def main() raises:
    # ----------------------------------------------------------------
    # LJ  —  10M atoms
    # ----------------------------------------------------------------
    var lj_n       = 10_000_000
    var lj_cpu_steps = 3    # ~30s/step on CPU; 3 steps gives a good rate
    var lj_gpu_steps = 100

    print("Building LJ system (~10M atoms)…")
    var lj_atoms_cpu = _build_fcc_atoms(lj_n)
    var lj_n_real    = lj_atoms_cpu.nlocal
    print("  actual N =", lj_n_real)

    print("LJ CPU: running", lj_cpu_steps, "steps…")
    var lj_pair_cpu = _make_lj_pair()
    var lj_sim_cpu  = Simulation[PairLJ, VelocityVerlet](
        lj_atoms_cpu^, lj_pair_cpu^, VelocityVerlet(),
        dt=0.002, skin=0.3, rebuild_interval=10,
    )
    var t0 = perf_counter_ns()
    lj_sim_cpu.run(lj_cpu_steps, print_interval=lj_cpu_steps + 1)
    _report("LJ CPU  ", lj_n_real, lj_cpu_steps, perf_counter_ns() - t0)

    # ----------------------------------------------------------------
    # Vashishta  —  128k atoms
    # ----------------------------------------------------------------
    var vs_n        = 128_000
    var vs_cpu_steps = 20
    var vs_gpu_steps = 100

    print("\nBuilding Vashishta system (~128k atoms)…")
    var vs_atoms_cpu = _build_sio2_random(vs_n)
    var vs_n_real    = vs_atoms_cpu.nlocal
    print("  actual N =", vs_n_real)

    print("Vashishta CPU: running", vs_cpu_steps, "steps…")
    var vs_pair_cpu = _make_vashishta_pair()
    var vs_sim_cpu  = Simulation[PairVashishta, VelocityVerlet](
        vs_atoms_cpu^, vs_pair_cpu^, VelocityVerlet(),
        dt=0.001, skin=0.3, rebuild_interval=10,
    )
    t0 = perf_counter_ns()
    vs_sim_cpu.run(vs_cpu_steps, print_interval=vs_cpu_steps + 1)
    _report("Vashishta CPU", vs_n_real, vs_cpu_steps, perf_counter_ns() - t0)

    # ----------------------------------------------------------------
    # GPU runs
    # ----------------------------------------------------------------
    comptime if has_accelerator():
        from std.gpu.host import DeviceContext

        print("\nLJ GPU: running", lj_gpu_steps, "steps…")
        var lj_atoms_gpu = _build_fcc_atoms(lj_n)
        var lj_pair_gpu  = _make_lj_pair()
        var lj_ctx       = DeviceContext()
        var lj_sim_gpu   = SimulationGPU[PairLJ, VelocityVerlet](
            lj_atoms_gpu^, lj_pair_gpu^, VelocityVerlet(), lj_ctx^,
            dt=0.002, skin=0.3, rebuild_interval=10,
        )
        t0 = perf_counter_ns()
        lj_sim_gpu.run(lj_gpu_steps, print_interval=lj_gpu_steps + 1)
        _report("LJ GPU  ", lj_n_real, lj_gpu_steps, perf_counter_ns() - t0)

        print("\nVashishta GPU: running", vs_gpu_steps, "steps…")
        var vs_atoms_gpu = _build_sio2_random(vs_n)
        var vs_pair_gpu  = _make_vashishta_pair()
        var vs_ctx       = DeviceContext()
        var vs_sim_gpu   = SimulationGPU[PairVashishta, VelocityVerlet](
            vs_atoms_gpu^, vs_pair_gpu^, VelocityVerlet(), vs_ctx^,
            dt=0.001, skin=0.3, rebuild_interval=10,
        )
        t0 = perf_counter_ns()
        vs_sim_gpu.run(vs_gpu_steps, print_interval=vs_gpu_steps + 1)
        _report("Vashishta GPU", vs_n_real, vs_gpu_steps, perf_counter_ns() - t0)
    else:
        print("\nNo GPU detected.")
