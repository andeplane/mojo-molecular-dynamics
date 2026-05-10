from math import sqrt
from atom import Atoms


struct NeighborList:
    """
    Full neighbor list stored as a flat CSR (compressed sparse row) array.

    For each local atom i, neighbors[offsets[i] : offsets[i+1]] contains the
    indices j (0 <= j < n_total) of all atoms within the list cutoff, j != i.
    The flat layout is GPU-friendly: a kernel for atom i loads offsets[i] and
    offsets[i+1], then streams the relevant range of neighbors[].

    This is a *full* list: if j is in i's list, i is in j's list too. This
    removes all write-conflicts when parallelizing the force loop over i —
    each thread writes only to f[i]. The cost is evaluating each pair twice,
    which we correct for with a factor of 1/2 on the energy.

    Vashishta's three-body terms also need a short neighbor list (r < r0_max).
    We store that separately as short_neighbors / short_offsets.
    """
    var nlocal: Int
    var offsets: List[Int]       # len = nlocal + 1  (prefix sums)
    var neighbors: List[Int]     # flat list of j indices
    var short_offsets: List[Int] # same structure for j with r < r0_max
    var short_neighbors: List[Int]
    var build_cutoff: Float64
    var build_short_cutoff: Float64

    fn __moveinit__(out self, owned other: NeighborList):
        self.nlocal = other.nlocal
        self.offsets = other.offsets^
        self.neighbors = other.neighbors^
        self.short_offsets = other.short_offsets^
        self.short_neighbors = other.short_neighbors^
        self.build_cutoff = other.build_cutoff
        self.build_short_cutoff = other.build_short_cutoff

    fn __init__(out self, nlocal: Int):
        self.nlocal = nlocal
        self.offsets = List[Int](capacity=nlocal + 1)
        self.neighbors = List[Int](capacity=nlocal * 50)
        self.short_offsets = List[Int](capacity=nlocal + 1)
        self.short_neighbors = List[Int](capacity=nlocal * 20)
        self.build_cutoff = 0.0
        self.build_short_cutoff = 0.0

    fn num_neighbors(self, i: Int) -> Int:
        return self.offsets[i + 1] - self.offsets[i]

    fn neighbor_start(self, i: Int) -> Int:
        return self.offsets[i]

    fn num_short(self, i: Int) -> Int:
        return self.short_offsets[i + 1] - self.short_offsets[i]

    fn short_start(self, i: Int) -> Int:
        return self.short_offsets[i]

    fn build(mut self, read atoms: Atoms, cutoff: Float64, short_cutoff: Float64 = 0.0):
        """
        Build full neighbor list using a cell/bin algorithm.

        Divides the simulation box into cubic cells of side >= cutoff.
        For each local atom i, checks all atoms j in the 27 neighboring cells
        (including i's own cell), adds j to i's list if |r_ij| < cutoff and j != i.
        Ghost atoms (indices >= nlocal) are included as potential neighbors.
        """
        self.build_cutoff = cutoff
        self.build_short_cutoff = short_cutoff
        self.nlocal = atoms.nlocal

        # --- Set up cells ---
        var lx = atoms.box[0]
        var ly = atoms.box[1]
        var lz = atoms.box[2]

        var nc_x = max(1, Int(lx / cutoff))
        var nc_y = max(1, Int(ly / cutoff))
        var nc_z = max(1, Int(lz / cutoff))
        var nc = nc_x * nc_y * nc_z
        var cell_lx = lx / Float64(nc_x)
        var cell_ly = ly / Float64(nc_y)
        var cell_lz = lz / Float64(nc_z)

        # Bin all atoms (local + ghost) into cells
        var n_total = atoms.n()
        var cell_count = List[Int](capacity=nc)
        for _ in range(nc):
            cell_count.append(0)

        var atom_cell = List[Int](capacity=n_total)
        for j in range(n_total):
            var cx = _clamp(Int(atoms.x[3 * j] / cell_lx), 0, nc_x - 1)
            var cy = _clamp(Int(atoms.x[3 * j + 1] / cell_ly), 0, nc_y - 1)
            var cz = _clamp(Int(atoms.x[3 * j + 2] / cell_lz), 0, nc_z - 1)
            var cid = (cx * nc_y + cy) * nc_z + cz
            atom_cell.append(cid)
            cell_count[cid] += 1

        # Build per-cell atom lists (prefix sums + sort)
        var cell_offsets = List[Int](capacity=nc + 1)
        cell_offsets.append(0)
        for c in range(nc):
            cell_offsets.append(cell_offsets[c] + cell_count[c])
        var cell_cursor = List[Int](capacity=nc)
        for c in range(nc):
            cell_cursor.append(cell_offsets[c])
        var cell_atoms = List[Int](capacity=n_total)
        for _ in range(n_total):
            cell_atoms.append(0)
        for j in range(n_total):
            var cid = atom_cell[j]
            cell_atoms[cell_cursor[cid]] = j
            cell_cursor[cid] += 1

        # --- Build neighbor lists ---
        self.offsets = List[Int](capacity=self.nlocal + 1)
        self.neighbors = List[Int](capacity=n_total)
        self.short_offsets = List[Int](capacity=self.nlocal + 1)
        self.short_neighbors = List[Int](capacity=n_total)

        self.offsets.append(0)
        self.short_offsets.append(0)
        var cutsq = cutoff * cutoff
        var short_cutsq = short_cutoff * short_cutoff

        for i in range(self.nlocal):
            var xi = atoms.x[3 * i]
            var yi = atoms.x[3 * i + 1]
            var zi = atoms.x[3 * i + 2]

            var ci_x = _clamp(Int(xi / cell_lx), 0, nc_x - 1)
            var ci_y = _clamp(Int(yi / cell_ly), 0, nc_y - 1)
            var ci_z = _clamp(Int(zi / cell_lz), 0, nc_z - 1)

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        var nx = ci_x + dx
                        var ny = ci_y + dy
                        var nz = ci_z + dz
                        # Periodic wrap of cell index
                        nx = ((nx % nc_x) + nc_x) % nc_x
                        ny = ((ny % nc_y) + nc_y) % nc_y
                        nz = ((nz % nc_z) + nc_z) % nc_z
                        var ncid = (nx * nc_y + ny) * nc_z + nz
                        var start = cell_offsets[ncid]
                        var end = cell_offsets[ncid + 1]
                        for idx in range(start, end):
                            var j = cell_atoms[idx]
                            if j == i:
                                continue
                            var ddx = xi - atoms.x[3 * j]
                            var ddy = yi - atoms.x[3 * j + 1]
                            var ddz = zi - atoms.x[3 * j + 2]
                            var rsq = ddx * ddx + ddy * ddy + ddz * ddz
                            if rsq < cutsq:
                                self.neighbors.append(j)
                            if short_cutsq > 0.0 and rsq < short_cutsq:
                                self.short_neighbors.append(j)

            self.offsets.append(len(self.neighbors))
            self.short_offsets.append(len(self.short_neighbors))


fn _clamp(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
