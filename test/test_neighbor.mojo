from std.testing import assert_equal, assert_true, assert_almost_equal
from mojo_md import Atoms, NeighborList


fn _cube4(box: Float64) -> Atoms:
    """4 atoms at corners of a unit cube scaled by 1 Å, inside a box."""
    var a = Atoms(4, box, box, box)
    # (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    a.x[0]=0.0; a.x[1]=0.0; a.x[2]=0.0
    a.x[3]=1.0; a.x[4]=0.0; a.x[5]=0.0
    a.x[6]=0.0; a.x[7]=1.0; a.x[8]=0.0
    a.x[9]=0.0; a.x[10]=0.0; a.x[11]=1.0
    for i in range(4):
        a.mass[i] = 1.0; a.type_id[i] = 0; a.tag[i] = i
    return a^


fn test_all_pairs_found() raises:
    """4 atoms, rc=1.5 Å — each has exactly 3 neighbors."""
    var a = _cube4(10.0)
    var nlist = NeighborList(4)
    nlist.build(a, 1.5)
    for i in range(4):
        assert_equal(nlist.num_neighbors(i), 3)


fn test_no_self_neighbor() raises:
    var a = _cube4(10.0)
    var nlist = NeighborList(4)
    nlist.build(a, 1.5)
    for i in range(4):
        var start = nlist.neighbor_start(i)
        var end = start + nlist.num_neighbors(i)
        for nb in range(start, end):
            assert_true(nlist.neighbors[nb] != i)


fn test_full_list_symmetry() raises:
    """Full list: if j in neigh(i) then i in neigh(j)."""
    var a = _cube4(10.0)
    var nlist = NeighborList(4)
    nlist.build(a, 1.5)
    for i in range(4):
        var s = nlist.neighbor_start(i)
        for nb in range(s, s + nlist.num_neighbors(i)):
            var j = nlist.neighbors[nb]
            # verify i appears in j's list
            var found = False
            var sj = nlist.neighbor_start(j)
            for nb2 in range(sj, sj + nlist.num_neighbors(j)):
                if nlist.neighbors[nb2] == i:
                    found = True
            assert_true(found)


fn test_cutoff_excludes_far_atom() raises:
    """2 atoms 5 Å apart, rc=4 Å → empty neighbor lists."""
    var a = Atoms(2, 20.0, 20.0, 20.0)
    a.x[0]=0.0; a.x[1]=0.0; a.x[2]=0.0
    a.x[3]=5.0; a.x[4]=0.0; a.x[5]=0.0
    for i in range(2): a.mass[i]=1.0; a.type_id[i]=0; a.tag[i]=i
    var nlist = NeighborList(2)
    nlist.build(a, 4.0)
    assert_equal(nlist.num_neighbors(0), 0)
    assert_equal(nlist.num_neighbors(1), 0)


fn test_short_list_subset() raises:
    """Short list is a subset of the full list."""
    var a = _cube4(10.0)
    var nlist = NeighborList(4)
    nlist.build(a, 1.5, 0.8)  # short_cutoff < cutoff

    # All short neighbors must also appear in full neighbors
    for i in range(4):
        var ss = nlist.short_start(i)
        for nb in range(ss, ss + nlist.num_short(i)):
            var j = nlist.short_neighbors[nb]
            var found = False
            var s = nlist.neighbor_start(i)
            for nb2 in range(s, s + nlist.num_neighbors(i)):
                if nlist.neighbors[nb2] == j:
                    found = True
            assert_true(found)


fn test_csr_offsets_consistent() raises:
    var a = _cube4(10.0)
    var nlist = NeighborList(4)
    nlist.build(a, 1.5)
    for i in range(4):
        assert_equal(nlist.offsets[i + 1] - nlist.offsets[i], nlist.num_neighbors(i))


fn test_two_atoms_within_cutoff() raises:
    """2 atoms 2 Å apart, rc=3 Å → each has the other as neighbor."""
    var a = Atoms(2, 20.0, 20.0, 20.0)
    a.x[0]=0.0; a.x[1]=0.0; a.x[2]=0.0
    a.x[3]=2.0; a.x[4]=0.0; a.x[5]=0.0
    for i in range(2): a.mass[i]=1.0; a.type_id[i]=0; a.tag[i]=i
    var nlist = NeighborList(2)
    nlist.build(a, 3.0)
    assert_equal(nlist.num_neighbors(0), 1)
    assert_equal(nlist.num_neighbors(1), 1)
    assert_equal(nlist.neighbors[nlist.neighbor_start(0)], 1)
    assert_equal(nlist.neighbors[nlist.neighbor_start(1)], 0)


fn main() raises:
    test_all_pairs_found()
    test_no_self_neighbor()
    test_full_list_symmetry()
    test_cutoff_excludes_far_atom()
    test_short_list_subset()
    test_csr_offsets_consistent()
    test_two_atoms_within_cutoff()
    print("test_neighbor: all passed")
