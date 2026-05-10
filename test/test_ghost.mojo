from std.testing import assert_equal, assert_almost_equal, assert_true
from atom import Atoms
from ghost import GhostBuilder


fn _make_atom(x: Float64, y: Float64, z: Float64, box: Float64) -> Atoms:
    var a = Atoms(1, box, box, box)
    a.x[0] = x; a.x[1] = y; a.x[2] = z
    a.mass[0] = 1.0; a.type_id[0] = 0; a.tag[0] = 0
    return a^


fn test_ghost_count_interior() raises:
    """Atom well inside box: no ghosts needed."""
    var a = _make_atom(5.0, 5.0, 5.0, 10.0)
    var gb = GhostBuilder(3.0)  # rc + skin = 3 Å
    gb.rebuild_ghosts(a)
    assert_equal(a.nghost, 0)


fn test_ghost_count_corner_atom() raises:
    """Atom at (0,0,0): all 7 corner-image directions overlap [0,L)."""
    var a = _make_atom(0.0, 0.0, 0.0, 10.0)
    var gb = GhostBuilder(2.0)
    gb.rebuild_ghosts(a)
    # All 8 corners of {-1,0}^3 \ (0,0,0) = 7 images overlap [0,L)+skin
    assert_equal(a.nghost, 7)


fn test_ghost_positions_x_only() raises:
    """Atom near x=0 face: should get a ghost at x + lx."""
    var a = _make_atom(0.5, 5.0, 5.0, 10.0)
    var gb = GhostBuilder(1.5)   # rc+skin = 1.5, so x+lx = 10.5 is within [0, 11.5)
    gb.rebuild_ghosts(a)
    # At least one ghost from the +x image (shift sx=+1)
    var found = False
    for g in range(a.nghost):
        var gidx = a.nlocal + g
        if abs(a.x[3*gidx] - 10.5) < 1e-10:
            found = True
    assert_true(found)


fn test_ghost_tag_copied() raises:
    """Ghost atoms must carry the same tag as their source."""
    var a = _make_atom(0.5, 5.0, 5.0, 10.0)
    a.tag[0] = 42
    var gb = GhostBuilder(2.0)
    gb.rebuild_ghosts(a)
    for g in range(a.nghost):
        assert_equal(a.tag[a.nlocal + g], 42)


fn test_ghost_type_and_mass_copied() raises:
    """Ghost atoms must carry the same type_id and mass as their source."""
    var a = _make_atom(0.5, 5.0, 5.0, 10.0)
    a.type_id[0] = 3; a.mass[0] = 2.5
    var gb = GhostBuilder(2.0)
    gb.rebuild_ghosts(a)
    for g in range(a.nghost):
        var gidx = a.nlocal + g
        assert_equal(a.type_id[gidx], 3)
        assert_almost_equal(a.mass[gidx], 2.5, atol=1e-15)


fn test_reverse_comm_single_ghost() raises:
    """Force on ghost must fold back to real atom after reverse_comm."""
    var a = _make_atom(0.5, 5.0, 5.0, 10.0)
    a.tag[0] = 0
    var gb = GhostBuilder(2.0)
    gb.rebuild_ghosts(a)
    assert_true(a.nghost > 0)

    # Zero real force, put 1 eV/Å on the first ghost
    a.f[0] = 0.0; a.f[1] = 0.0; a.f[2] = 0.0
    var gidx = a.nlocal
    a.f[3*gidx] = 1.0; a.f[3*gidx+1] = 0.0; a.f[3*gidx+2] = 0.0

    gb.reverse_comm(a)
    assert_almost_equal(a.f[0], 1.0, atol=1e-14)


fn test_reverse_comm_zero_ghost_force() raises:
    """If ghost force is zero, reverse_comm changes nothing."""
    var a = _make_atom(0.5, 5.0, 5.0, 10.0)
    a.f[0] = 3.0
    var gb = GhostBuilder(2.0)
    gb.rebuild_ghosts(a)
    gb.reverse_comm(a)
    assert_almost_equal(a.f[0], 3.0, atol=1e-14)


fn main() raises:
    test_ghost_count_interior()
    test_ghost_count_corner_atom()
    test_ghost_positions_x_only()
    test_ghost_tag_copied()
    test_ghost_type_and_mass_copied()
    test_reverse_comm_single_ghost()
    test_reverse_comm_zero_ghost_force()
    print("test_ghost: all passed")
