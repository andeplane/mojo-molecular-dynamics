from testing import assert_equal, assert_almost_equal, assert_true
from atom import Atoms, minimum_image, wrap_into_box, _floor


fn test_n_total() raises:
    var a = Atoms(4, 10.0, 10.0, 10.0)
    assert_equal(a.n(), 4)
    a.nghost = 3
    assert_equal(a.n(), 7)


fn test_zero_forces() raises:
    var a = Atoms(3, 10.0, 10.0, 10.0)
    # Set non-zero forces
    for i in range(9):
        a.f[i] = Float64(i + 1)
    a.zero_forces()
    for i in range(9):
        assert_almost_equal(a.f[i], 0.0, atol=1e-15)


fn test_kinetic_energy() raises:
    var a = Atoms(3, 10.0, 10.0, 10.0)
    # Atom 0: m=2, v=(1,0,0) → KE = 1.0
    a.mass[0] = 2.0
    a.v[0] = 1.0; a.v[1] = 0.0; a.v[2] = 0.0
    # Atom 1: m=1, v=(0,2,0) → KE = 2.0
    a.mass[1] = 1.0
    a.v[3] = 0.0; a.v[4] = 2.0; a.v[5] = 0.0
    # Atom 2: m=4, v=(0,0,1) → KE = 2.0
    a.mass[2] = 4.0
    a.v[6] = 0.0; a.v[7] = 0.0; a.v[8] = 1.0

    assert_almost_equal(a.kinetic_energy(), 5.0, atol=1e-12)


fn test_temperature() raises:
    var a = Atoms(4, 10.0, 10.0, 10.0)
    for i in range(4):
        a.mass[i] = 1.0
        a.v[3*i] = 1.0; a.v[3*i+1] = 0.0; a.v[3*i+2] = 0.0
    # KE = 4 * 0.5 = 2.0; dof = 3*4-3 = 9; T = 2*KE/dof = 4/9
    assert_almost_equal(a.temperature(), 4.0 / 9.0, atol=1e-12)


fn test_minimum_image_positive() raises:
    # dx = 0.6L → should return -0.4L
    assert_almost_equal(minimum_image(6.0, 10.0), -4.0, atol=1e-14)


fn test_minimum_image_negative() raises:
    # dx = -0.1L → should stay -1.0
    assert_almost_equal(minimum_image(-1.0, 10.0), -1.0, atol=1e-14)


fn test_minimum_image_zero() raises:
    assert_almost_equal(minimum_image(0.0, 10.0), 0.0, atol=1e-14)


fn test_minimum_image_at_half() raises:
    # dx = +5.0 in box 10.0 → exactly at ±half; should not change sign
    var mi = minimum_image(5.0, 10.0)
    assert_true(mi == 5.0 or mi == -5.0)


fn test_wrap_into_box_positive_overshoot() raises:
    var a = Atoms(1, 10.0, 10.0, 10.0)
    a.x[0] = 11.0; a.x[1] = 5.0; a.x[2] = 5.0
    wrap_into_box(a)
    assert_almost_equal(a.x[0], 1.0, atol=1e-12)


fn test_wrap_into_box_negative() raises:
    var a = Atoms(1, 10.0, 10.0, 10.0)
    a.x[0] = -0.5; a.x[1] = 5.0; a.x[2] = 5.0
    wrap_into_box(a)
    assert_almost_equal(a.x[0], 9.5, atol=1e-12)


fn test_wrap_into_box_no_change() raises:
    var a = Atoms(1, 10.0, 10.0, 10.0)
    a.x[0] = 5.0; a.x[1] = 3.0; a.x[2] = 7.0
    wrap_into_box(a)
    assert_almost_equal(a.x[0], 5.0, atol=1e-12)
    assert_almost_equal(a.x[1], 3.0, atol=1e-12)
    assert_almost_equal(a.x[2], 7.0, atol=1e-12)


fn test_grow() raises:
    var a = Atoms(2, 10.0, 10.0, 10.0)
    var old_nmax = a.nmax
    a.grow(old_nmax * 2)
    assert_true(a.nmax == old_nmax * 2)
    assert_true(len(a.x) >= 3 * a.nmax)
    assert_true(len(a.f) >= 3 * a.nmax)
    assert_true(len(a.mass) >= a.nmax)


fn test_floor_positive() raises:
    assert_almost_equal(_floor(2.9), 2.0, atol=1e-15)
    assert_almost_equal(_floor(3.0), 3.0, atol=1e-15)


fn test_floor_negative() raises:
    assert_almost_equal(_floor(-0.1), -1.0, atol=1e-15)
    assert_almost_equal(_floor(-3.0), -3.0, atol=1e-15)
    assert_almost_equal(_floor(-3.1), -4.0, atol=1e-15)


fn main() raises:
    test_n_total()
    test_zero_forces()
    test_kinetic_energy()
    test_temperature()
    test_minimum_image_positive()
    test_minimum_image_negative()
    test_minimum_image_zero()
    test_minimum_image_at_half()
    test_wrap_into_box_positive_overshoot()
    test_wrap_into_box_negative()
    test_wrap_into_box_no_change()
    test_grow()
    test_floor_positive()
    test_floor_negative()
    print("test_atom: all passed")
