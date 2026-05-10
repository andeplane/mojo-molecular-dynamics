from testing import assert_almost_equal, assert_true
from atom import Atoms
from neighbor import NeighborList
from pair_lj import PairLJ, LJParams


# Argon reference parameters
alias EPS: Float64 = 0.01040   # eV
alias SIG: Float64 = 3.4       # Å
alias RC:  Float64 = 8.5       # Å

# r_eq = 2^(1/6) * sigma — potential minimum
alias R_EQ: Float64 = 3.8163709642518681  # Å = 2^(1/6) * 3.4 (exact to double precision)


fn _make_2atom(r: Float64) -> Atoms:
    """2 atoms on x-axis, distance r, box large enough to avoid PBC images."""
    var a = Atoms(2, 50.0, 50.0, 50.0)
    a.x[0] = 0.0; a.x[1] = 0.0; a.x[2] = 0.0
    a.x[3] = r;   a.x[4] = 0.0; a.x[5] = 0.0
    a.mass[0] = 39.948; a.mass[1] = 39.948
    a.type_id[0] = 0;   a.type_id[1] = 0
    a.tag[0] = 0;       a.tag[1] = 1
    return a^


fn _make_nlist_2atom() -> NeighborList:
    """Full neighbor list for 2 atoms: each is the other's neighbor."""
    var nlist = NeighborList(2)
    nlist.offsets.append(0); nlist.offsets.append(1); nlist.offsets.append(2)
    nlist.neighbors.append(1); nlist.neighbors.append(0)
    nlist.short_offsets.append(0); nlist.short_offsets.append(0); nlist.short_offsets.append(0)
    nlist.build_cutoff = RC + 0.3
    return nlist^


fn test_lj_params_lj1() raises:
    var p = LJParams(EPS, SIG, RC)
    var sig6 = SIG ** 6
    assert_almost_equal(p.lj1, 48.0 * EPS * sig6 * sig6, rtol=1e-12)


fn test_lj_params_lj2() raises:
    var p = LJParams(EPS, SIG, RC)
    assert_almost_equal(p.lj2, 24.0 * EPS * SIG ** 6, rtol=1e-12)


fn test_lj_energy_shift_at_cutoff() raises:
    """energy_shift = V_LJ(rc), so V(rc) - energy_shift = 0 exactly."""
    var p = LJParams(EPS, SIG, RC)
    var rc2inv = 1.0 / (RC * RC)
    var rc6inv = rc2inv * rc2inv * rc2inv
    var v_rc = rc6inv * (p.lj3 * rc6inv - p.lj4)
    assert_almost_equal(v_rc, p.energy_shift, rtol=1e-12)


fn test_lj_equilibrium_force() raises:
    """At r = r_eq = 2^(1/6)*sigma the LJ force is exactly zero."""
    var atoms = _make_2atom(R_EQ)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    _ = pair.compute(atoms, nlist)

    assert_almost_equal(atoms.f[0], 0.0, atol=1e-8)


fn test_lj_equilibrium_energy() raises:
    """At r = r_eq the energy is -epsilon (minus shift, which is tiny at rc=8.5)."""
    var atoms = _make_2atom(R_EQ)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    var energy = pair.compute(atoms, nlist)
    # Full list halves energy; unshifted V_min = -epsilon
    assert_almost_equal(energy, -EPS - pair.params[0].energy_shift, atol=1e-8)


fn test_lj_repulsion_close() raises:
    """Below r_eq the force is repulsive (pushes atoms apart)."""
    var atoms = _make_2atom(0.8 * SIG)   # inside the well
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    _ = pair.compute(atoms, nlist)

    # atom 0 should be pushed in -x direction (away from atom 1 at +x)
    assert_true(atoms.f[0] < 0.0)
    # atom 1 should be pushed in +x direction
    assert_true(atoms.f[3] > 0.0)


fn test_lj_force_symmetry() raises:
    """Full list: f[i] + f[j] == 0 (Newton's 3rd law)."""
    var atoms = _make_2atom(4.0)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    _ = pair.compute(atoms, nlist)

    assert_almost_equal(atoms.f[0] + atoms.f[3], 0.0, atol=1e-12)
    assert_almost_equal(atoms.f[1] + atoms.f[4], 0.0, atol=1e-12)
    assert_almost_equal(atoms.f[2] + atoms.f[5], 0.0, atol=1e-12)


fn test_lj_energy_full_list_factor() raises:
    """
    Energy from full-list compute = pair energy (energy_shift-corrected).
    We verify the 1/2 factor: with both i→j and j→i counted, the returned
    total should equal one pair energy.
    """
    var r: Float64 = 4.0
    var atoms = _make_2atom(r)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    var energy = pair.compute(atoms, nlist)

    # Hand-compute expected energy
    var p = pair.params[0]
    var r2inv = 1.0 / (r * r)
    var r6inv = r2inv * r2inv * r2inv
    var expected = r6inv * (p.lj3 * r6inv - p.lj4) - p.energy_shift
    assert_almost_equal(energy, expected, rtol=1e-10)


fn test_lj_beyond_cutoff_zero_force() raises:
    """Atoms beyond cutoff: force and energy == 0."""
    var atoms = _make_2atom(RC + 1.0)
    var nlist = _make_nlist_2atom()
    var pair = PairLJ(1)
    pair.set_pair(0, 0, EPS, SIG, RC)

    atoms.zero_forces()
    var energy = pair.compute(atoms, nlist)

    assert_almost_equal(energy, 0.0, atol=1e-15)
    assert_almost_equal(atoms.f[0], 0.0, atol=1e-15)


fn main() raises:
    test_lj_params_lj1()
    test_lj_params_lj2()
    test_lj_energy_shift_at_cutoff()
    test_lj_equilibrium_force()
    test_lj_equilibrium_energy()
    test_lj_repulsion_close()
    test_lj_force_symmetry()
    test_lj_energy_full_list_factor()
    test_lj_beyond_cutoff_zero_force()
    print("test_lj: all passed")
