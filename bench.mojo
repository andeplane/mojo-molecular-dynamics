"""
bench.mojo — Benchmark entry point.

Iterates over {LJ, Vashishta} × {CPU, GPU} × size ladder.
Reports Matom·steps/s table to stdout; optionally writes CSV.

Usage:
    mojo run bench.mojo
    mojo run bench.mojo --csv results.csv

Timeout logic: if a CPU run at size N takes > 30 s, all larger sizes for
that pair style + CPU backend are marked TIMEOUT and skipped.  GPU runs are
always attempted (they are fast enough at any listed size).
"""

from atom import Atoms
from integrator import VelocityVerlet
from pair_lj import PairLJ
from pair_vashishta import PairVashishta, make_vashishta_param
from simulation import Simulation
from std.sys import argv, has_accelerator
from time import perf_counter_ns


# ---------------------------------------------------------------------------
# Size ladder (T018)
# ---------------------------------------------------------------------------

alias SIZE_LADDER = List[Int]

fn _size_ladder() -> SIZE_LADDER:
    var s = SIZE_LADDER()
    s.append(1000); s.append(2500); s.append(5000); s.append(10000)
    s.append(25000); s.append(50000); s.append(100000); s.append(125000)
    s.append(250000); s.append(500000); s.append(1000000); s.append(2500000)
    s.append(5000000); s.append(10000000)
    return s^


# ---------------------------------------------------------------------------
# BenchmarkResult (T018)
# ---------------------------------------------------------------------------

struct BenchmarkResult(Movable):
    var pair_style:    String
    var n_atoms:       Int
    var backend:       String
    var n_steps:       Int
    var elapsed_s:     Float64
    var matom_steps_s: Float64
    var timed_out:     Bool
    var unavailable:   Bool

    fn __init__(
        out self,
        pair_style: String, n_atoms: Int, backend: String,
        n_steps: Int, elapsed_s: Float64, matom_steps_s: Float64,
        timed_out: Bool, unavailable: Bool,
    ):
        self.pair_style    = pair_style
        self.n_atoms       = n_atoms
        self.backend       = backend
        self.n_steps       = n_steps
        self.elapsed_s     = elapsed_s
        self.matom_steps_s = matom_steps_s
        self.timed_out     = timed_out
        self.unavailable   = unavailable


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------

fn _build_fcc_atoms(n_target: Int) -> Atoms:
    """FCC Ar lattice with ≥ n_target atoms (4 atoms per unit cell, a = 5.405 Å)."""
    var a: Float64 = 5.405
    var cells_needed = (n_target + 3) // 4
    var nx = max(1, Int(Float64(cells_needed) ** (1.0 / 3.0)))
    while nx * nx * nx * 4 < n_target:
        nx += 1
    var ny = nx; var nz = nx
    var nlocal = nx * ny * nz * 4
    var lx = Float64(nx) * a
    var fcc_bx = List[Float64](); var fcc_by = List[Float64](); var fcc_bz = List[Float64]()
    fcc_bx.append(0.0); fcc_by.append(0.0); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.5); fcc_bz.append(0.0)
    fcc_bx.append(0.5); fcc_by.append(0.0); fcc_bz.append(0.5)
    fcc_bx.append(0.0); fcc_by.append(0.5); fcc_bz.append(0.5)
    var atoms = Atoms(nlocal, lx, lx, lx)
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


fn _build_sio2_random(n_target: Int) -> Atoms:
    """
    Random Si/O placement scaled to target N.
    Si : O = 1 : 2.  Lattice constant ~3.1 Å gives SiO₂-like density.
    """
    var n_si = n_target // 3
    var n_o  = n_target - n_si
    var nlocal = n_si + n_o
    var nc = max(1, Int(Float64(nlocal) ** (1.0 / 3.0)) + 1)
    var a: Float64 = 3.1
    var lx = Float64(nc) * a
    var atoms = Atoms(nlocal, lx, lx, lx)
    var idx = 0
    for ix in range(nc):
        if idx >= nlocal:
            break
        for iy in range(nc):
            if idx >= nlocal:
                break
            for iz in range(nc):
                if idx >= nlocal:
                    break
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
    pair.set_param(0, 0, 0, p_sisi); pair.set_param(0, 0, 1, p_zero)
    pair.set_param(0, 1, 0, p_zero); pair.set_param(0, 1, 1, p_sioo)
    pair.set_param(1, 0, 0, p_zero); pair.set_param(1, 0, 1, p_osio)
    pair.set_param(1, 1, 0, p_zero); pair.set_param(1, 1, 1, p_ooo)
    return pair^


# ---------------------------------------------------------------------------
# Step-count heuristic: target ~5 s per run
# ---------------------------------------------------------------------------

fn _n_steps_for(n_atoms: Int) -> Int:
    """
    Rough step count that keeps wall-clock near 5 s on a modern CPU.
    We use a conservative pre-estimate; actual timing determines whether
    the result is marked TIMEOUT.
    """
    if n_atoms <= 5000:
        return 500
    elif n_atoms <= 25000:
        return 200
    elif n_atoms <= 100000:
        return 50
    elif n_atoms <= 500000:
        return 20
    else:
        return 10


alias TIMEOUT_S: Float64 = 30.0


# ---------------------------------------------------------------------------
# CPU benchmarks — T019 (LJ), T021 (Vashishta)
# ---------------------------------------------------------------------------

fn _bench_cpu_lj(n_atoms: Int, timed_out: Bool) -> BenchmarkResult:
    """Run one LJ CPU benchmark at the given atom count."""
    if timed_out:
        return BenchmarkResult("LJ", n_atoms, "CPU", 0, 0.0, 0.0, True, False)

    var n_steps = _n_steps_for(n_atoms)
    var atoms   = _build_fcc_atoms(n_atoms)
    var actual_n = atoms.nlocal
    var pair    = _make_lj_pair()
    var sim     = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(), dt=0.002, skin=0.3, rebuild_interval=10,
    )

    var t0 = perf_counter_ns()
    sim.run(n_steps, print_interval=n_steps + 1)  # suppress per-step output
    var t1 = perf_counter_ns()

    var elapsed  = Float64(t1 - t0) / 1.0e9
    var matom_ss = Float64(actual_n) * Float64(n_steps) / elapsed / 1.0e6
    var did_tout = elapsed > TIMEOUT_S
    return BenchmarkResult("LJ", actual_n, "CPU", n_steps, elapsed, matom_ss, did_tout, False)


fn _bench_cpu_vashishta(n_atoms: Int, timed_out: Bool) -> BenchmarkResult:
    """Run one Vashishta CPU benchmark at the given atom count."""
    if timed_out:
        return BenchmarkResult("Vashishta", n_atoms, "CPU", 0, 0.0, 0.0, True, False)

    var n_steps = _n_steps_for(n_atoms)
    var atoms   = _build_sio2_random(n_atoms)
    var actual_n = atoms.nlocal
    var pair    = _make_vashishta_pair()
    var sim     = Simulation[PairVashishta, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(), dt=0.001, skin=0.3, rebuild_interval=10,
    )

    var t0 = perf_counter_ns()
    sim.run(n_steps, print_interval=n_steps + 1)
    var t1 = perf_counter_ns()

    var elapsed  = Float64(t1 - t0) / 1.0e9
    var matom_ss = Float64(actual_n) * Float64(n_steps) / elapsed / 1.0e6
    var did_tout = elapsed > TIMEOUT_S
    return BenchmarkResult("Vashishta", actual_n, "CPU", n_steps, elapsed, matom_ss, did_tout, False)


# ---------------------------------------------------------------------------
# Table formatter — T022
# ---------------------------------------------------------------------------

fn _pad_right(s: String, width: Int) -> String:
    var out = s
    while len(out) < width:
        out = out + " "
    return out


fn _pad_left(s: String, width: Int) -> String:
    var out = s
    while len(out) < width:
        out = " " + out
    return out


fn _fmt_float(v: Float64, decimals: Int = 2) -> String:
    # Simple float formatter: integer part + "." + decimal digits
    var neg = v < 0.0
    var abs_v = -v if neg else v
    var int_part = Int(abs_v)
    var frac = abs_v - Float64(int_part)
    var scale: Float64 = 1.0
    for _ in range(decimals):
        scale *= 10.0
    var frac_int = Int(frac * scale + 0.5)
    var frac_str = String(frac_int)
    while len(frac_str) < decimals:
        frac_str = "0" + frac_str
    var result = String(int_part) + "." + frac_str
    if neg:
        result = "-" + result
    return result


fn _fmt_int_comma(n: Int) -> String:
    """Format an integer with comma thousands separators."""
    var s = String(n)
    var out = ""
    var count = 0
    for i in range(len(s) - 1, -1, -1):
        if count > 0 and count % 3 == 0:
            out = "," + out
        out = String(s[i]) + out
        count += 1
    return out


fn _print_separator():
    print("------------+------------+---------+---------+-----------------+----------")


fn _print_header():
    print("Pair style  |  N atoms   | Backend | Steps   | Matom·steps/s   | Speedup  ")
    _print_separator()


fn _speedup_str(gpu_ms: Float64, cpu_ms: Float64) -> String:
    if cpu_ms <= 0.0:
        return "N/A"
    var s = gpu_ms / cpu_ms
    return _fmt_float(s, 2) + "×"


fn _print_result(r: BenchmarkResult, cpu_matom_s: Float64):
    var ps   = _pad_right(r.pair_style, 10)
    var na   = _pad_left(_fmt_int_comma(r.n_atoms), 10)
    var back = _pad_right(r.backend, 7)
    var steps_s: String
    if r.timed_out:
        steps_s = _pad_left("TIMEOUT", 7)
    elif r.unavailable:
        steps_s = _pad_left("N/A", 7)
    else:
        steps_s = _pad_left(String(r.n_steps), 7)
    var ms_s: String
    if r.timed_out:
        ms_s = _pad_left("TIMEOUT", 15)
    elif r.unavailable:
        ms_s = _pad_left("UNAVAILABLE", 15)
    else:
        ms_s = _pad_left(_fmt_float(r.matom_steps_s, 2), 15)
    var spdup: String
    if r.backend == "CPU" or r.timed_out or r.unavailable:
        spdup = _pad_left("—", 8)
    else:
        spdup = _pad_left(_speedup_str(r.matom_steps_s, cpu_matom_s), 8)
    print(ps + " | " + na + " | " + back + " | " + steps_s + " | " + ms_s + " | " + spdup)


# ---------------------------------------------------------------------------
# CSV writer — T023
# ---------------------------------------------------------------------------

fn _write_csv(results: List[BenchmarkResult], path: String) raises:
    from std.python import Python
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "w")
    _ = f.write("pair_style,n_atoms,backend,n_steps,elapsed_s,matom_steps_s,timed_out,unavailable\n")
    for r in results:
        var line = (r.pair_style + "," +
                    String(r.n_atoms) + "," +
                    r.backend + "," +
                    String(r.n_steps) + "," +
                    _fmt_float(r.elapsed_s, 6) + "," +
                    _fmt_float(r.matom_steps_s, 4) + "," +
                    ("true" if r.timed_out   else "false") + "," +
                    ("true" if r.unavailable else "false") + "\n")
        _ = f.write(line)
    f.close()


# ---------------------------------------------------------------------------
# main — T018, T019, T020, T021, T022, T023
# ---------------------------------------------------------------------------

fn main() raises:
    var args     = argv()
    var csv_path = ""
    for i in range(len(args) - 1):
        if args[i] == "--csv":
            csv_path = args[i + 1]
            break

    var sizes   = _size_ladder()
    var results = List[BenchmarkResult]()

    # ----------------------------------------------------------------
    # LJ benchmarks — T019 (CPU), T020 (GPU)
    # ----------------------------------------------------------------
    var lj_cpu_timed_out = False
    var lj_cpu_matom: List[Float64] = List[Float64]()

    for i in range(len(sizes)):
        var n = sizes[i]
        var r = _bench_cpu_lj(n, lj_cpu_timed_out)
        if r.timed_out and not lj_cpu_timed_out:
            lj_cpu_timed_out = True
        lj_cpu_matom.append(r.matom_steps_s)
        results.append(r^)

    # GPU LJ — T020
    comptime if has_accelerator():
        from atom_gpu import GPUAtoms, GPUNeighborList
        from integrator_gpu import VelocityVerletGPU
        from pair_lj_gpu import PairLJGPU
        from simulation_gpu import SimulationGPU
        from std.gpu.host import DeviceContext

        for i in range(len(sizes)):
            var n       = sizes[i]
            var n_steps = _n_steps_for(n)
            var atoms   = _build_fcc_atoms(n)
            var actual_n = atoms.nlocal
            var cpu_pair = _make_lj_pair()
            var ctx      = DeviceContext()
            var gpu_pair = PairLJGPU.from_cpu(cpu_pair, ctx)
            var sim      = SimulationGPU[PairLJGPU, VelocityVerletGPU](
                atoms^, gpu_pair^, VelocityVerletGPU(), ctx^,
                dt=0.002, skin=0.3, rebuild_interval=10,
            )
            var t0 = perf_counter_ns()
            sim.run(n_steps, print_interval=n_steps + 1)
            var t1 = perf_counter_ns()
            var elapsed  = Float64(t1 - t0) / 1.0e9
            var matom_ss = Float64(actual_n) * Float64(n_steps) / elapsed / 1.0e6
            results.append(BenchmarkResult("LJ", actual_n, "GPU", n_steps, elapsed, matom_ss, False, False))
    else:
        for i in range(len(sizes)):
            results.append(BenchmarkResult("LJ", sizes[i], "GPU", 0, 0.0, 0.0, False, True))

    # ----------------------------------------------------------------
    # Vashishta benchmarks — T021 (CPU + GPU)
    # ----------------------------------------------------------------
    var vash_cpu_timed_out = False
    var vash_cpu_matom: List[Float64] = List[Float64]()

    for i in range(len(sizes)):
        var n = sizes[i]
        var r = _bench_cpu_vashishta(n, vash_cpu_timed_out)
        if r.timed_out and not vash_cpu_timed_out:
            vash_cpu_timed_out = True
        vash_cpu_matom.append(r.matom_steps_s)
        results.append(r^)

    # GPU Vashishta — T021
    comptime if has_accelerator():
        from pair_vashishta_gpu import PairVashishtaGPU

        for i in range(len(sizes)):
            var n       = sizes[i]
            var n_steps = _n_steps_for(n)
            var atoms   = _build_sio2_random(n)
            var actual_n = atoms.nlocal
            var cpu_pair = _make_vashishta_pair()
            var ctx      = DeviceContext()
            var gpu_pair = PairVashishtaGPU.from_cpu(cpu_pair, ctx)
            var sim      = SimulationGPU[PairVashishtaGPU, VelocityVerletGPU](
                atoms^, gpu_pair^, VelocityVerletGPU(), ctx^,
                dt=0.001, skin=0.3, rebuild_interval=10,
            )
            var t0 = perf_counter_ns()
            sim.run(n_steps, print_interval=n_steps + 1)
            var t1 = perf_counter_ns()
            var elapsed  = Float64(t1 - t0) / 1.0e9
            var matom_ss = Float64(actual_n) * Float64(n_steps) / elapsed / 1.0e6
            results.append(BenchmarkResult("Vashishta", actual_n, "GPU", n_steps, elapsed, matom_ss, False, False))
    else:
        for i in range(len(sizes)):
            results.append(BenchmarkResult("Vashishta", sizes[i], "GPU", 0, 0.0, 0.0, False, True))

    # ----------------------------------------------------------------
    # Print table — T022
    # ----------------------------------------------------------------
    _print_header()

    # Build a lookup of CPU Matom·steps/s by (pair_style, n_atoms)
    # Print in order: LJ CPU, LJ GPU interleaved by size; then Vashishta
    var n_sizes = len(sizes)
    for i in range(n_sizes):
        var cpu_r  = results[i]
        var cpu_ms = lj_cpu_matom[i]
        _print_result(cpu_r,  0.0)
        # GPU LJ result is at index n_sizes + i
        var gpu_r  = results[n_sizes + i]
        _print_result(gpu_r, cpu_ms)

    _print_separator()

    for i in range(n_sizes):
        var cpu_r  = results[2 * n_sizes + i]
        var cpu_ms = vash_cpu_matom[i]
        _print_result(cpu_r, 0.0)
        var gpu_r  = results[3 * n_sizes + i]
        _print_result(gpu_r, cpu_ms)

    # ----------------------------------------------------------------
    # CSV output — T023
    # ----------------------------------------------------------------
    if csv_path != "":
        _write_csv(results, csv_path)
        print("\nResults written to", csv_path)
