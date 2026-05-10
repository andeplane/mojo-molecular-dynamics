from testing import assert_almost_equal
from atom import Atoms
from integrator import VelocityVerlet


fn _single_atom(m: Float64, vx: Float64, vy: Float64, vz: Float64,
                fx: Float64, fy: Float64, fz: Float64) -> Atoms:
    var a = Atoms(1, 100.0, 100.0, 100.0)
    a.mass[0] = m
    a.v[0] = vx; a.v[1] = vy; a.v[2] = vz
    a.f[0] = fx; a.f[1] = fy; a.f[2] = fz
    a.x[0] = 0.0; a.x[1] = 0.0; a.x[2] = 0.0
    a.tag[0] = 0; a.type_id[0] = 0
    return a^


fn test_half_step_v_formula() raises:
    """v_new = v + 0.5 * dt * f / m."""
    var a = _single_atom(2.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0)
    var vv = VelocityVerlet()
    var dt: Float64 = 0.1
    vv.half_step_v(a, dt)
    # v_x += 0.5 * 0.1 * 4.0 / 2.0 = 0.1  →  1.0 + 0.1 = 1.1
    assert_almost_equal(a.v[0], 1.1, atol=1e-14)
    assert_almost_equal(a.v[1], 0.0, atol=1e-14)
    assert_almost_equal(a.v[2], 0.0, atol=1e-14)


fn test_half_step_v_zero_force() raises:
    """With zero force, velocity unchanged."""
    var a = _single_atom(1.0, 3.0, -2.0, 1.0, 0.0, 0.0, 0.0)
    var vv = VelocityVerlet()
    vv.half_step_v(a, 0.5)
    assert_almost_equal(a.v[0],  3.0, atol=1e-14)
    assert_almost_equal(a.v[1], -2.0, atol=1e-14)
    assert_almost_equal(a.v[2],  1.0, atol=1e-14)


fn test_half_step_v_all_components() raises:
    """All three velocity components updated correctly."""
    var a = _single_atom(1.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0)
    var vv = VelocityVerlet()
    var dt: Float64 = 0.2
    vv.half_step_v(a, dt)
    assert_almost_equal(a.v[0], 0.0 + 0.5*0.2*2.0, atol=1e-14)
    assert_almost_equal(a.v[1], 0.0 + 0.5*0.2*4.0, atol=1e-14)
    assert_almost_equal(a.v[2], 0.0 + 0.5*0.2*6.0, atol=1e-14)


fn test_full_step_x_formula() raises:
    """x_new = x + dt * v."""
    var a = _single_atom(1.0, 3.0, -1.0, 2.0, 0.0, 0.0, 0.0)
    a.x[0] = 1.0; a.x[1] = 2.0; a.x[2] = 3.0
    var vv = VelocityVerlet()
    var dt: Float64 = 0.5
    vv.full_step_x(a, dt)
    assert_almost_equal(a.x[0], 1.0 + 0.5*3.0,  atol=1e-14)
    assert_almost_equal(a.x[1], 2.0 + 0.5*(-1.0), atol=1e-14)
    assert_almost_equal(a.x[2], 3.0 + 0.5*2.0,  atol=1e-14)


fn test_full_step_x_zero_velocity() raises:
    """With zero velocity, position unchanged."""
    var a = _single_atom(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    a.x[0] = 5.0; a.x[1] = 6.0; a.x[2] = 7.0
    var vv = VelocityVerlet()
    vv.full_step_x(a, 0.1)
    assert_almost_equal(a.x[0], 5.0, atol=1e-14)
    assert_almost_equal(a.x[1], 6.0, atol=1e-14)
    assert_almost_equal(a.x[2], 7.0, atol=1e-14)


fn test_half_step_time_reversibility() raises:
    """
    Velocity Verlet is time-reversible:
    forward half_step_v, full_step_x, half_step_v then
    negate v and repeat → returns to original x.
    """
    var a = _single_atom(1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    a.x[0] = 3.0
    var x0 = a.x[0]
    var v0 = a.v[0]
    var vv = VelocityVerlet()
    var dt: Float64 = 0.1

    # Forward step
    vv.half_step_v(a, dt)
    vv.full_step_x(a, dt)
    vv.half_step_v(a, dt)

    # Reverse: negate v and step back
    a.v[0] = -a.v[0]
    vv.half_step_v(a, dt)
    vv.full_step_x(a, dt)
    vv.half_step_v(a, dt)
    a.v[0] = -a.v[0]

    assert_almost_equal(a.x[0], x0, atol=1e-12)
    assert_almost_equal(a.v[0], v0, atol=1e-12)


fn main() raises:
    test_half_step_v_formula()
    test_half_step_v_zero_force()
    test_half_step_v_all_components()
    test_full_step_x_formula()
    test_full_step_x_zero_velocity()
    test_half_step_time_reversibility()
    print("test_integrator: all passed")
