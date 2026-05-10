from std.math import sqrt
from std.testing import assert_almost_equal, assert_true
from mojo_md import Atoms, VelocityVerlet, NeighborList, PairLJ, Simulation
from mojo_md.ghost import GhostBuilder


# LJ Argon parameters
comptime EPS: Float64 = 0.01040
comptime SIG: Float64 = 3.4
comptime RC:  Float64 = 8.5


fn _ar_pair() -> PairLJ:
    var p = PairLJ(1)
    p.set_pair(0, 0, EPS, SIG, RC)
    return p^


fn _2atom_atoms(r: Float64) -> Atoms:
    """2 Ar atoms on x-axis, separation r, large box."""
    var a = Atoms(2, 50.0, 50.0, 50.0)
    a.x[0]=0.0; a.x[1]=0.0; a.x[2]=0.0
    a.x[3]=r;   a.x[4]=0.0; a.x[5]=0.0
    a.mass[0] = 39.948; a.mass[1] = 39.948
    a.type_id[0] = 0;   a.type_id[1] = 0
    a.tag[0] = 0;       a.tag[1] = 1
    return a^


fn _build_nlist(atoms: Atoms, cutoff: Float64) -> NeighborList:
    var nlist = NeighborList(atoms.nlocal)
    nlist.build(atoms, cutoff)
    return nlist^


fn test_lj_force_symmetry_2atom() raises:
    """Full-list LJ: f[0] + f[1] == 0 for all components."""
    var atoms = _2atom_atoms(4.0)
    var nlist = _build_nlist(atoms, RC + 0.3)
    var pair = _ar_pair()

    atoms.zero_forces()
    _ = pair.compute(atoms, nlist)

    assert_almost_equal(atoms.f[0] + atoms.f[3], 0.0, atol=1e-12)
    assert_almost_equal(atoms.f[1] + atoms.f[4], 0.0, atol=1e-12)
    assert_almost_equal(atoms.f[2] + atoms.f[5], 0.0, atol=1e-12)


fn test_lj_energy_conservation() raises:
    """
    4 Ar atoms in a small box at near-equilibrium positions.
    Run 100 steps; total energy should drift < 0.1%.

    Equilibrium nearest-neighbour distance = 2^(1/6)*sigma ≈ 3.817 Å.
    We use a 2×2×2 FCC sub-cell: box = 2*a where a ≈ 5.405 Å, giving
    density close to liquid Ar.
    """
    var a_lat: Float64 = 5.405
    var lbox = 2.0 * a_lat
    var atoms = Atoms(4, lbox, lbox, lbox)
    # SC (not FCC) so 4 atoms in a cube — simple but still tests conservation
    var half = a_lat
    atoms.x[0]=0.0;   atoms.x[1]=0.0;   atoms.x[2]=0.0
    atoms.x[3]=half;  atoms.x[4]=0.0;   atoms.x[5]=0.0
    atoms.x[6]=0.0;   atoms.x[7]=half;  atoms.x[8]=0.0
    atoms.x[9]=0.0;   atoms.x[10]=0.0;  atoms.x[11]=half
    for i in range(4):
        atoms.mass[i] = 39.948; atoms.type_id[i] = 0; atoms.tag[i] = i
    # Small velocities: ~100 K
    atoms.v[0] = 0.01; atoms.v[3] = -0.01  # px balanced

    var pair = _ar_pair()
    var integrator = VelocityVerlet()
    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, integrator^,
        dt=0.001, skin=0.5, rebuild_interval=5,
    )

    var ke0 = sim.atoms.kinetic_energy()
    # We need to force initial PE without relying on private fields.
    # Use a two-pass: get PE from a fresh compute
    var nlist0 = NeighborList(4)
    nlist0.build(sim.atoms, RC + 0.5)
    var pair2 = _ar_pair()
    sim.atoms.zero_forces()
    var pe0 = pair2.compute(sim.atoms, nlist0)
    var te0 = ke0 + pe0

    sim.run(100, print_interval=200)  # suppress print

    var ke1 = sim.atoms.kinetic_energy()
    var nlist1 = NeighborList(4)
    nlist1.build(sim.atoms, RC + 0.5)
    var pair3 = _ar_pair()
    sim.atoms.zero_forces()
    var pe1 = pair3.compute(sim.atoms, nlist1)
    var te1 = ke1 + pe1

    var drift = abs(te1 - te0) / (abs(te0) + 1e-30)
    assert_true(drift < 0.001)  # < 0.1%


fn test_ghost_forces_pbc_consistency() raises:
    """
    Single Ar atom in a small box.
    With ghosts (full PBC treatment) the force on the real atom must be
    self-consistently zero by symmetry (since all image contributions cancel
    for a single atom with full cubic symmetry).
    """
    var a = Atoms(1, 10.0, 10.0, 10.0)
    a.x[0] = 5.0; a.x[1] = 5.0; a.x[2] = 5.0
    a.mass[0] = 39.948; a.type_id[0] = 0; a.tag[0] = 0

    var gb = GhostBuilder(RC + 0.3)
    gb.rebuild_ghosts(a)

    var nlist = NeighborList(1)
    nlist.build(a, RC + 0.3)

    var pair = _ar_pair()
    a.zero_forces()
    _ = pair.compute(a, nlist)
    gb.reverse_comm(a)

    # By cubic symmetry, ghost contributions cancel → f ≈ 0 on real atom
    assert_almost_equal(a.f[0], 0.0, atol=1e-10)
    assert_almost_equal(a.f[1], 0.0, atol=1e-10)
    assert_almost_equal(a.f[2], 0.0, atol=1e-10)


fn test_rebuild_interval_invariant() raises:
    """
    Running with rebuild_every=1 vs rebuild_every=20 should give the same
    final energy (within floating-point noise) for a stable system.
    """
    fn _make_sim(rebuild_every: Int) -> Simulation[PairLJ, VelocityVerlet]:
        var at = Atoms(4, 2.0 * 5.405, 2.0 * 5.405, 2.0 * 5.405)
        var half = 5.405
        at.x[0]=0.0;at.x[1]=0.0;at.x[2]=0.0
        at.x[3]=half;at.x[4]=0.0;at.x[5]=0.0
        at.x[6]=0.0;at.x[7]=half;at.x[8]=0.0
        at.x[9]=0.0;at.x[10]=0.0;at.x[11]=half
        for i in range(4):
            at.mass[i]=39.948; at.type_id[i]=0; at.tag[i]=i
        # Same tiny velocity
        at.v[0]=0.005; at.v[3]=-0.005
        var pr = _ar_pair()
        var iv = VelocityVerlet()
        return Simulation[PairLJ, VelocityVerlet](
            at^, pr^, iv^, dt=0.001, skin=0.5, rebuild_interval=rebuild_every)

    var sim1 = _make_sim(1)
    var sim2 = _make_sim(20)

    sim1.run(50, print_interval=1000)
    sim2.run(50, print_interval=1000)

    var ke1 = sim1.atoms.kinetic_energy()
    var ke2 = sim2.atoms.kinetic_energy()

    # KE should be very similar (same system, same initial conditions)
    # Allow 1% tolerance for numerical differences
    var delta = abs(ke1 - ke2) / (abs(ke1) + 1e-30)
    assert_true(delta < 0.01)


fn main() raises:
    test_lj_force_symmetry_2atom()
    test_lj_energy_conservation()
    test_ghost_forces_pbc_consistency()
    test_rebuild_interval_invariant()
    print("test_integration: all passed")
