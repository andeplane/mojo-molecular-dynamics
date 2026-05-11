from std.testing import assert_almost_equal
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from mojo_md.atom import Atoms, GPUAtoms
from mojo_md.integrator import VelocityVerlet


fn _single_gpu_atom(
    x0: Float64, x1: Float64, x2: Float64,
    v0: Float64, v1: Float64, v2: Float64,
    f0: Float64, f1: Float64, f2: Float64,
    mass: Float64,
) raises -> GPUAtoms:
    var ctx = DeviceContext()
    var a = Atoms(1, 100.0, 100.0, 100.0)
    a.x[0] = x0; a.x[1] = x1; a.x[2] = x2
    a.v[0] = v0; a.v[1] = v1; a.v[2] = v2
    a.f[0] = f0; a.f[1] = f1; a.f[2] = f2
    a.mass[0] = mass; a.type_id[0] = 0; a.tag[0] = 0
    return GPUAtoms.from_cpu(a, ctx)


fn test_half_step_v_gpu() raises:
    comptime if not has_accelerator():
        print("test_half_step_v_gpu: skip (no GPU)")
        return
    # mass=2, v=(1,0,0), f=(4,0,0), dt=0.1
    # Expected: v[0] += 0.5*0.1 * 4/2 = 0.1  →  v = (1.1, 0, 0)
    var ctx = DeviceContext()
    var a = Atoms(1, 100.0, 100.0, 100.0)
    a.x[0] = 0.0; a.x[1] = 0.0; a.x[2] = 0.0
    a.v[0] = 1.0; a.v[1] = 0.0; a.v[2] = 0.0
    a.f[0] = 4.0; a.f[1] = 0.0; a.f[2] = 0.0
    a.mass[0] = 2.0; a.type_id[0] = 0; a.tag[0] = 0
    var gpu = GPUAtoms.from_cpu(a, ctx)
    var integ = VelocityVerlet()
    integ.half_step_v_gpu(gpu, ctx, 0.1)
    var h_v = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(h_v, gpu.v)
    ctx.synchronize()
    assert_almost_equal(Float64(h_v[0]), 1.1, rtol=1e-5)
    assert_almost_equal(Float64(h_v[1]), 0.0, atol=1e-7)
    assert_almost_equal(Float64(h_v[2]), 0.0, atol=1e-7)


fn test_full_step_x_gpu() raises:
    comptime if not has_accelerator():
        print("test_full_step_x_gpu: skip (no GPU)")
        return
    # x=(1,2,3), v=(0.5,0.1,-0.2), dt=0.1
    # Expected: x += dt*v  →  x = (1.05, 2.01, 2.98)
    var ctx = DeviceContext()
    var a = Atoms(1, 100.0, 100.0, 100.0)
    a.x[0] = 1.0; a.x[1] = 2.0; a.x[2] = 3.0
    a.v[0] = 0.5; a.v[1] = 0.1; a.v[2] = -0.2
    a.f[0] = 0.0; a.f[1] = 0.0; a.f[2] = 0.0
    a.mass[0] = 1.0; a.type_id[0] = 0; a.tag[0] = 0
    var gpu = GPUAtoms.from_cpu(a, ctx)
    var integ = VelocityVerlet()
    integ.full_step_x_gpu(gpu, ctx, 0.1)
    var h_x = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(h_x, gpu.x)
    ctx.synchronize()
    assert_almost_equal(Float64(h_x[0]), 1.05, rtol=1e-5)
    assert_almost_equal(Float64(h_x[1]), 2.01, rtol=1e-5)
    assert_almost_equal(Float64(h_x[2]), 2.98, rtol=1e-5)


fn test_half_then_full_step_gpu() raises:
    comptime if not has_accelerator():
        print("test_half_then_full_step_gpu: skip (no GPU)")
        return
    # mass=1, x=0, v=0, f=2, dt=0.5
    # After half_v:  v = 0 + 0.5*0.5*2/1 = 0.5
    # After full_x:  x = 0 + 0.5*0.5 = 0.25
    var ctx = DeviceContext()
    var a = Atoms(1, 100.0, 100.0, 100.0)
    a.x[0] = 0.0; a.x[1] = 0.0; a.x[2] = 0.0
    a.v[0] = 0.0; a.v[1] = 0.0; a.v[2] = 0.0
    a.f[0] = 2.0; a.f[1] = 0.0; a.f[2] = 0.0
    a.mass[0] = 1.0; a.type_id[0] = 0; a.tag[0] = 0
    var gpu = GPUAtoms.from_cpu(a, ctx)
    var integ = VelocityVerlet()
    integ.half_step_v_gpu(gpu, ctx, 0.5)
    integ.full_step_x_gpu(gpu, ctx, 0.5)
    var h_x = ctx.enqueue_create_host_buffer[DType.float32](3)
    var h_v = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(h_x, gpu.x)
    ctx.enqueue_copy(h_v, gpu.v)
    ctx.synchronize()
    assert_almost_equal(Float64(h_v[0]), 0.5, rtol=1e-5)
    assert_almost_equal(Float64(h_x[0]), 0.25, rtol=1e-5)


fn main() raises:
    test_half_step_v_gpu()
    test_full_step_x_gpu()
    test_half_then_full_step_gpu()
    print("test_gpu_integrator: all passed")
