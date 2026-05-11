from std.testing import assert_almost_equal
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.neighbor import NeighborList
from mojo_md.pair_lj import PairLJ

comptime EPS: Float64 = 0.01040
comptime SIG: Float64 = 3.4
comptime RC:  Float64 = 8.5


fn _make_2atom(r: Float64) -> Atoms:
    var a = Atoms(2, 50.0, 50.0, 50.0)
    a.x[0] = 0.0; a.x[1] = 0.0; a.x[2] = 0.0
    a.x[3] = r;   a.x[4] = 0.0; a.x[5] = 0.0
    a.mass[0] = 39.948; a.mass[1] = 39.948
    a.type_id[0] = 0;   a.type_id[1] = 0
    a.tag[0] = 0;       a.tag[1] = 1
    return a^


fn _make_nlist_2atom() -> NeighborList:
    var nlist = NeighborList(2)
    nlist.offsets.append(0); nlist.offsets.append(1); nlist.offsets.append(2)
    nlist.neighbors.append(1); nlist.neighbors.append(0)
    nlist.short_offsets.append(0); nlist.short_offsets.append(0); nlist.short_offsets.append(0)
    nlist.build_cutoff = RC + 0.3
    return nlist^


fn test_lj_forces_cpu_vs_gpu() raises:
    comptime if not has_accelerator():
        print("test_lj_forces_cpu_vs_gpu: skip (no GPU)")
        return
    var r: Float64 = 4.0
    var atoms = _make_2atom(r)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    # CPU forces
    atoms.zero_forces()
    var pe_cpu = pair.compute(atoms, nlist)
    var f0x = atoms.f[0]; var f0y = atoms.f[1]; var f0z = atoms.f[2]
    var f1x = atoms.f[3]; var f1y = atoms.f[4]; var f1z = atoms.f[5]

    # GPU forces — upload zeroed atoms so device forces start at zero
    atoms.zero_forces()
    var ctx = DeviceContext()
    var gpu_atoms = GPUAtoms.from_cpu(atoms, ctx)
    var gpu_nlist = GPUNeighborList.from_cpu(nlist, ctx)
    var params_dev = pair.make_gpu_params(ctx)

    _ = pair.compute_gpu(gpu_atoms, gpu_nlist, params_dev, ctx)
    var pe_gpu = gpu_atoms.read_pe_to_cpu(ctx) * 0.5  # full list double-counts

    var h_f = ctx.enqueue_create_host_buffer[DType.float32](6)
    ctx.enqueue_copy(h_f, gpu_atoms.f)
    ctx.synchronize()

    assert_almost_equal(Float64(h_f[0]), f0x, atol=1e-5)
    assert_almost_equal(Float64(h_f[1]), f0y, atol=1e-7)
    assert_almost_equal(Float64(h_f[2]), f0z, atol=1e-7)
    assert_almost_equal(Float64(h_f[3]), f1x, atol=1e-5)
    assert_almost_equal(Float64(h_f[4]), f1y, atol=1e-7)
    assert_almost_equal(Float64(h_f[5]), f1z, atol=1e-7)
    assert_almost_equal(pe_gpu, pe_cpu, atol=1e-5)


fn test_lj_force_symmetry_gpu() raises:
    comptime if not has_accelerator():
        print("test_lj_force_symmetry_gpu: skip (no GPU)")
        return
    # Full list: f[i] + f[j] == 0 (Newton's 3rd law) on GPU
    var atoms = _make_2atom(4.0)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    var ctx = DeviceContext()
    var gpu_atoms = GPUAtoms.from_cpu(atoms, ctx)
    var gpu_nlist = GPUNeighborList.from_cpu(nlist, ctx)
    var params_dev = pair.make_gpu_params(ctx)

    _ = pair.compute_gpu(gpu_atoms, gpu_nlist, params_dev, ctx)

    var h_f = ctx.enqueue_create_host_buffer[DType.float32](6)
    ctx.enqueue_copy(h_f, gpu_atoms.f)
    ctx.synchronize()

    assert_almost_equal(Float64(h_f[0]) + Float64(h_f[3]), 0.0, atol=1e-6)
    assert_almost_equal(Float64(h_f[1]) + Float64(h_f[4]), 0.0, atol=1e-7)
    assert_almost_equal(Float64(h_f[2]) + Float64(h_f[5]), 0.0, atol=1e-7)


fn test_lj_beyond_cutoff_zero_gpu() raises:
    comptime if not has_accelerator():
        print("test_lj_beyond_cutoff_zero_gpu: skip (no GPU)")
        return
    # Atoms beyond cutoff: GPU force and energy must be zero
    var atoms = _make_2atom(RC + 1.0)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    var ctx = DeviceContext()
    var gpu_atoms = GPUAtoms.from_cpu(atoms, ctx)
    var gpu_nlist = GPUNeighborList.from_cpu(nlist, ctx)
    var params_dev = pair.make_gpu_params(ctx)

    _ = pair.compute_gpu(gpu_atoms, gpu_nlist, params_dev, ctx)
    var pe_gpu = gpu_atoms.read_pe_to_cpu(ctx) * 0.5

    var h_f = ctx.enqueue_create_host_buffer[DType.float32](6)
    ctx.enqueue_copy(h_f, gpu_atoms.f)
    ctx.synchronize()

    assert_almost_equal(pe_gpu, 0.0, atol=1e-7)
    assert_almost_equal(Float64(h_f[0]), 0.0, atol=1e-7)
    assert_almost_equal(Float64(h_f[3]), 0.0, atol=1e-7)


fn main() raises:
    test_lj_forces_cpu_vs_gpu()
    test_lj_force_symmetry_gpu()
    test_lj_beyond_cutoff_zero_gpu()
    print("test_gpu_lj: all passed")
