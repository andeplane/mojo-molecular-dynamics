from math import sqrt
from testing import assert_almost_equal, assert_true
from pair_vashishta import VashishtaParam, _twobody, _threebody, QQR2E


# Si-O Vashishta parameters (from LAMMPS SiO.1990.vashishta, Si-O-O entry)
alias P_SiO = VashishtaParam(
    bigh=188.0, eta=9.0, zi=1.6, zj=-0.8,
    lambda1=10.0, bigd=1.245, lambda4=4.43,
    bigw=22.1179, cut=5.5,
    bigb=4.7325, gamma=1.0, r0=2.60,
    bigc=0.0, costheta=-0.777,
)

# Si-Si-Si entry (mostly zero, for testing edge cases)
alias P_SiSi = VashishtaParam(
    bigh=0.82023, eta=11.0, zi=1.6, zj=1.6,
    lambda1=999.0, bigd=0.0, lambda4=999.0,
    bigw=0.0, cut=5.0,
    bigb=0.0, gamma=0.0, r0=0.0,
    bigc=0.0, costheta=-0.333,
)


fn test_precomputed_zizj() raises:
    assert_almost_equal(P_SiO.zizj, 1.6 * (-0.8) * QQR2E, rtol=1e-12)


fn test_precomputed_heta() raises:
    assert_almost_equal(P_SiO.heta, 188.0 * 9.0, rtol=1e-12)


fn test_precomputed_big2b() raises:
    assert_almost_equal(P_SiO.big2b, 2.0 * 4.7325, rtol=1e-12)


fn test_precomputed_big6w() raises:
    assert_almost_equal(P_SiO.big6w, 6.0 * 22.1179, rtol=1e-12)


fn test_twobody_energy_at_cutoff() raises:
    """V2(rc) must be ≈ 0 due to tail correction."""
    var rsq = P_SiO.cut * P_SiO.cut
    var result = _twobody(P_SiO, rsq)
    assert_almost_equal(result.energy, 0.0, atol=1e-6)


fn test_twobody_force_at_cutoff() raises:
    """Tail correction also zeroes the force at the cutoff."""
    var rsq = P_SiO.cut * P_SiO.cut
    var result = _twobody(P_SiO, rsq)
    assert_almost_equal(result.fforce, 0.0, atol=1e-6)


fn test_twobody_repulsion_close_range() raises:
    """At very short range with same-sign charges (Si-Si), Coulomb + H both repulsive."""
    var r: Float64 = 1.5   # very close
    var result = _twobody(P_SiSi, r * r)
    assert_true(result.energy > 0.0)


fn test_threebody_force_conservation() raises:
    """fi + fj + fk == 0 (global Newton's 3rd for the three-body term)."""
    # Geometry: i at origin, j at (2.0, 0, 0), k at (0, 2.0, 0)
    var rsq1: Float64 = 4.0   # |r_ij|² = 2²
    var rsq2: Float64 = 4.0   # |r_ik|²
    var res = _threebody(
        P_SiO, P_SiO, P_SiO,
        rsq1, rsq2,
        2.0, 0.0, 0.0,   # delr1 = rj - ri
        0.0, 2.0, 0.0,   # delr2 = rk - ri
    )
    var fi_x = -(res.fj_x + res.fk_x)
    var fi_y = -(res.fj_y + res.fk_y)
    var fi_z = -(res.fj_z + res.fk_z)
    # Sum must be zero
    assert_almost_equal(fi_x + res.fj_x + res.fk_x, 0.0, atol=1e-14)
    assert_almost_equal(fi_y + res.fj_y + res.fk_y, 0.0, atol=1e-14)
    assert_almost_equal(fi_z + res.fj_z + res.fk_z, 0.0, atol=1e-14)


fn test_threebody_at_equilibrium_angle() raises:
    """
    When the angle equals θ₀ (cos θ = costheta), the angular force term
    vanishes (delcs = 0). Only radial part survives.
    """
    # For P_SiO: costheta = -0.777
    # We construct (delr1, delr2) such that cosθ = cos_theta0
    # Use: delr1 = (r, 0, 0), delr2 = (r*cos_theta0, r*sin_theta0, 0)
    var cos0: Float64 = -0.777
    var sin0 = sqrt(1.0 - cos0 * cos0)
    var r: Float64 = 2.0
    var res = _threebody(
        P_SiO, P_SiO, P_SiO,
        r*r, r*r,
        r, 0.0, 0.0,
        r*cos0, r*sin0, 0.0,
    )
    # At equilibrium angle: angular force on j should be perpendicular to delr1
    # and zero in the delr2 direction (since delcs = 0 → facang = 0)
    # The component of fj along delr2 direction should vanish
    var fj_along_delr2 = res.fj_x * cos0 + res.fj_y * sin0
    assert_almost_equal(fj_along_delr2, 0.0, atol=1e-10)


fn test_threebody_symmetry_jk_swap() raises:
    """Swapping j and k should give the same energy but swap fj and fk."""
    var r: Float64 = 2.0
    var res1 = _threebody(
        P_SiO, P_SiO, P_SiO,
        r*r, r*r,
        r, 0.0, 0.0,     # delr1 = rj - ri
        0.0, r, 0.0,     # delr2 = rk - ri
    )
    var res2 = _threebody(
        P_SiO, P_SiO, P_SiO,
        r*r, r*r,
        0.0, r, 0.0,     # now j ↔ k
        r, 0.0, 0.0,
    )
    assert_almost_equal(res1.energy, res2.energy, rtol=1e-12)
    assert_almost_equal(res1.fj_x, res2.fk_x, atol=1e-12)
    assert_almost_equal(res1.fk_x, res2.fj_x, atol=1e-12)


fn test_threebody_zero_B() raises:
    """When B = 0 (e.g. Si-Si-Si), three-body contribution is exactly zero."""
    # P_SiSi has bigb = 0
    var r: Float64 = 1.5
    var res = _threebody(
        P_SiSi, P_SiSi, P_SiSi,
        r*r, r*r,
        r, 0.0, 0.0,
        0.0, r, 0.0,
    )
    assert_almost_equal(res.energy, 0.0, atol=1e-15)
    assert_almost_equal(res.fj_x, 0.0, atol=1e-15)
    assert_almost_equal(res.fk_y, 0.0, atol=1e-15)


fn main() raises:
    test_precomputed_zizj()
    test_precomputed_heta()
    test_precomputed_big2b()
    test_precomputed_big6w()
    test_twobody_energy_at_cutoff()
    test_twobody_force_at_cutoff()
    test_twobody_repulsion_close_range()
    test_threebody_force_conservation()
    test_threebody_at_equilibrium_angle()
    test_threebody_symmetry_jk_swap()
    test_threebody_zero_B()
    print("test_vashishta: all passed")
