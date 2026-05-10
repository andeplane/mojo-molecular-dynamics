from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from mojo_md.atom import Atoms, GPUAtoms

comptime _BLOCK_SIZE: Int = 256


# ---------------------------------------------------------------------------
# Shared kernel bodies — used by BOTH the CPU `parallelize` path and the GPU
# `enqueue_function` path. Owner-computes: each thread writes only to atom i,
# no atomics, no race conditions, identical algorithm on either backend.
# ---------------------------------------------------------------------------

@always_inline
fn _half_step_v_body(
    i: Int,
    v_ptr:    UnsafePointer[Float64, MutAnyOrigin],
    f_ptr:    UnsafePointer[Float64, MutAnyOrigin],
    mass_ptr: UnsafePointer[Float64, MutAnyOrigin],
    half_dt:  Float64,
):
    """v[i] += 0.5 * dt * f[i] / mass[i]"""
    var inv_m = 1.0 / mass_ptr[i]
    v_ptr[3 * i]     += half_dt * f_ptr[3 * i]     * inv_m
    v_ptr[3 * i + 1] += half_dt * f_ptr[3 * i + 1] * inv_m
    v_ptr[3 * i + 2] += half_dt * f_ptr[3 * i + 2] * inv_m


@always_inline
fn _full_step_x_body(
    i: Int,
    x_ptr: UnsafePointer[Float64, MutAnyOrigin],
    v_ptr: UnsafePointer[Float64, MutAnyOrigin],
    dt:    Float64,
):
    """x[i] += dt * v[i]"""
    x_ptr[3 * i]     += dt * v_ptr[3 * i]
    x_ptr[3 * i + 1] += dt * v_ptr[3 * i + 1]
    x_ptr[3 * i + 2] += dt * v_ptr[3 * i + 2]


# GPU kernels — thin wrappers that compute the thread index and call the body.

fn _half_step_v_kernel(
    v_ptr:    UnsafePointer[Float64, MutAnyOrigin],
    f_ptr:    UnsafePointer[Float64, MutAnyOrigin],
    mass_ptr: UnsafePointer[Float64, MutAnyOrigin],
    half_dt:  Float64,
    nlocal:   Int,
):
    var i = Int(global_idx.x)
    if i < nlocal:
        _half_step_v_body(i, v_ptr, f_ptr, mass_ptr, half_dt)


fn _full_step_x_kernel(
    x_ptr:  UnsafePointer[Float64, MutAnyOrigin],
    v_ptr:  UnsafePointer[Float64, MutAnyOrigin],
    dt:     Float64,
    nlocal: Int,
):
    var i = Int(global_idx.x)
    if i < nlocal:
        _full_step_x_body(i, x_ptr, v_ptr, dt)


# ---------------------------------------------------------------------------
# Integrator trait — both half-steps available on CPU and GPU.
# ---------------------------------------------------------------------------

trait Integrator(Movable, ImplicitlyDestructible):
    """Time-integrator interface. Each method has a CPU and a GPU dispatcher
    that share their kernel body, so the algorithm is written once."""

    fn half_step_v(mut self, mut atoms: Atoms, dt: Float64):
        ...
    fn full_step_x(mut self, mut atoms: Atoms, dt: Float64):
        ...
    fn half_step_v_gpu(mut self, mut atoms: GPUAtoms, ctx: DeviceContext, dt: Float64) raises:
        ...
    fn full_step_x_gpu(mut self, mut atoms: GPUAtoms, ctx: DeviceContext, dt: Float64) raises:
        ...


struct VelocityVerlet(Integrator):
    """Standard velocity Verlet (NVE). One kernel body per half-step, dispatched
    via parallelize on CPU and enqueue_function on GPU."""

    fn __init__(out self):
        pass

    # ---- CPU dispatch ----
    fn half_step_v(mut self, mut atoms: Atoms, dt: Float64):
        var half_dt = 0.5 * dt
        var nlocal = atoms.nlocal
        var v_ptr = atoms.v.unsafe_ptr()
        var f_ptr = atoms.f.unsafe_ptr()
        var mass_ptr = atoms.mass.unsafe_ptr()

        @parameter
        fn body(i: Int):
            _half_step_v_body(i, v_ptr, f_ptr, mass_ptr, half_dt)

        parallelize[body](nlocal)

    fn full_step_x(mut self, mut atoms: Atoms, dt: Float64):
        var nlocal = atoms.nlocal
        var x_ptr = atoms.x.unsafe_ptr()
        var v_ptr = atoms.v.unsafe_ptr()

        @parameter
        fn body(i: Int):
            _full_step_x_body(i, x_ptr, v_ptr, dt)

        parallelize[body](nlocal)

    # ---- GPU dispatch ----
    fn half_step_v_gpu(mut self, mut atoms: GPUAtoms, ctx: DeviceContext, dt: Float64) raises:
        var nlocal = atoms.nlocal
        var n_blocks = ceildiv(nlocal, _BLOCK_SIZE)
        ctx.enqueue_function[_half_step_v_kernel, _half_step_v_kernel](
            atoms.v.unsafe_ptr(),
            atoms.f.unsafe_ptr(),
            atoms.mass.unsafe_ptr(),
            0.5 * dt,
            nlocal,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

    fn full_step_x_gpu(mut self, mut atoms: GPUAtoms, ctx: DeviceContext, dt: Float64) raises:
        var nlocal = atoms.nlocal
        var n_blocks = ceildiv(nlocal, _BLOCK_SIZE)
        ctx.enqueue_function[_full_step_x_kernel, _full_step_x_kernel](
            atoms.x.unsafe_ptr(),
            atoms.v.unsafe_ptr(),
            dt,
            nlocal,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )
