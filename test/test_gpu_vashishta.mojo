from std.testing import assert_almost_equal
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.neighbor import NeighborList
from mojo_md.ghost import GhostBuilder
from mojo_md.pair_vashishta import PairVashishta, make_vashishta_param


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


fn test_vashishta_2body_cpu_vs_gpu() raises:
    comptime if not has_accelerator():
        print("test_vashishta_2body_cpu_vs_gpu: skip (no GPU)")
        return
    var atoms = _build_sio2_atoms()
    var pair = _make_vashishta_pair()

    var skin: Float64 = 0.3
    var rcut = pair.cutoff() + skin
    var rcut_short = pair.short_cutoff()

    var ghosts = GhostBuilder(rcut)
    ghosts.rebuild_ghosts(atoms)

    var nlist = NeighborList(atoms.nlocal)
    nlist.build(atoms, rcut, rcut_short)

    # CPU: 2-body only — temporarily zero the short list so 3-body contribution is absent
    # We do this by building a nlist with short_cutoff=0 (no short neighbors).
    var nlist_2b = NeighborList(atoms.nlocal)
    nlist_2b.build(atoms, rcut, 0.0)

    atoms.zero_forces()
    var pe_cpu = pair.compute(atoms, nlist_2b)
    # CPU compute calls both 2-body and 3-body passes; with empty short list the
    # 3-body pass is a no-op, so pe_cpu = 2-body energy only.
    var f_cpu = List[Float64]()
    for i in range(3 * atoms.nlocal):
        f_cpu.append(atoms.f[i])

    # GPU: same neighbor list (no short list)
    atoms.zero_forces()
    var ctx = DeviceContext()
    var gpu_atoms = GPUAtoms.from_cpu(atoms, ctx)
    var gpu_nlist = GPUNeighborList.from_cpu(nlist_2b, ctx)
    var params_dev = pair.make_gpu_params(ctx)

    _ = pair.compute_gpu(gpu_atoms, gpu_nlist, params_dev, ctx)
    var pe_gpu = gpu_atoms.read_pe_to_cpu(ctx)

    var h_f = ctx.enqueue_create_host_buffer[DType.float32](3 * atoms.nlocal)
    ctx.enqueue_copy(h_f, gpu_atoms.f)
    ctx.synchronize()

    assert_almost_equal(pe_gpu, pe_cpu, atol=1e-2)
    for i in range(3 * atoms.nlocal):
        assert_almost_equal(Float64(h_f[i]), f_cpu[i], atol=1e-3)


fn test_vashishta_full_cpu_vs_gpu() raises:
    comptime if not has_accelerator():
        print("test_vashishta_full_cpu_vs_gpu: skip (no GPU)")
        return
    # Full Vashishta (2-body + 3-body): GPU forces must match CPU forces.
    # This specifically exercises _vashishta_3body_kernel case-A and case-B.
    var atoms = _build_sio2_atoms()
    var pair = _make_vashishta_pair()

    var skin: Float64 = 0.3
    var rcut = pair.cutoff() + skin
    var rcut_short = pair.short_cutoff()

    var ghosts = GhostBuilder(rcut)
    ghosts.rebuild_ghosts(atoms)

    var nlist = NeighborList(atoms.nlocal)
    nlist.build(atoms, rcut, rcut_short)

    atoms.zero_forces()
    var pe_cpu = pair.compute(atoms, nlist)
    var f_cpu = List[Float64]()
    for i in range(3 * atoms.nlocal):
        f_cpu.append(atoms.f[i])

    atoms.zero_forces()
    var ctx = DeviceContext()
    var gpu_atoms = GPUAtoms.from_cpu(atoms, ctx)
    var gpu_nlist = GPUNeighborList.from_cpu(nlist, ctx)
    var params_dev = pair.make_gpu_params(ctx)

    _ = pair.compute_gpu(gpu_atoms, gpu_nlist, params_dev, ctx)
    var pe_gpu = gpu_atoms.read_pe_to_cpu(ctx)

    var h_f = ctx.enqueue_create_host_buffer[DType.float32](3 * atoms.nlocal)
    ctx.enqueue_copy(h_f, gpu_atoms.f)
    ctx.synchronize()

    assert_almost_equal(pe_gpu, pe_cpu, atol=1e-1)
    for i in range(3 * atoms.nlocal):
        assert_almost_equal(Float64(h_f[i]), f_cpu[i], atol=1e-2)


fn main() raises:
    test_vashishta_2body_cpu_vs_gpu()
    test_vashishta_full_cpu_vs_gpu()
    print("test_gpu_vashishta: all passed")
