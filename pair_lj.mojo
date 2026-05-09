from algorithm import parallelize
from atom import Atoms
from neighbor import NeighborList
from pair_style import PairStyle


struct LJParams:
    """
    Precomputed Lennard-Jones coefficients for one (i,j) type pair.

    The LJ potential is:
        V(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]  -  V(rc)    [energy shifted]

    The force magnitude (positive = repulsive) is:
        f(r) = [ 48ε σ^12 / r^13 - 24ε σ^6 / r^7 ]
             = [ lj1 * r6inv - lj2 ] * r6inv * r2inv

    where r6inv = 1/r^6, r2inv = 1/r^2.
    """
    var lj1: Float64         # 48 * epsilon * sigma^12
    var lj2: Float64         # 24 * epsilon * sigma^6
    var lj3: Float64         # 4  * epsilon * sigma^12  (energy)
    var lj4: Float64         # 4  * epsilon * sigma^6   (energy)
    var rc_sq: Float64       # cutoff^2
    var energy_shift: Float64  # V(rc) subtracted from every r < rc

    fn __init__(out self, epsilon: Float64, sigma: Float64, rc: Float64):
        var sig6 = sigma ** 6
        var sig12 = sig6 * sig6
        self.lj1 = 48.0 * epsilon * sig12
        self.lj2 = 24.0 * epsilon * sig6
        self.lj3 = 4.0 * epsilon * sig12
        self.lj4 = 4.0 * epsilon * sig6
        self.rc_sq = rc * rc
        var r2inv_c = 1.0 / self.rc_sq
        var r6inv_c = r2inv_c * r2inv_c * r2inv_c
        self.energy_shift = r6inv_c * (self.lj3 * r6inv_c - self.lj4)


struct PairLJ(PairStyle):
    """
    Multi-type Lennard-Jones pair potential.

    Parameters are stored in a flat list of size n_types * n_types,
    indexed as params[n_types * itype + jtype].

    Uses the full neighbor list — each pair appears twice, so the summed
    potential energy is divided by 2 before returning.
    """
    var n_types: Int
    var params: List[LJParams]
    var _cutoff: Float64

    fn __init__(out self, n_types: Int):
        self.n_types = n_types
        self.params = List[LJParams](capacity=n_types * n_types)
        self._cutoff = 0.0

    fn set_pair(mut self, itype: Int, jtype: Int, epsilon: Float64, sigma: Float64, rc: Float64):
        """Set LJ parameters for types itype and jtype (symmetric)."""
        # Extend list if needed (simple approach: fill with default entries)
        while len(self.params) < self.n_types * self.n_types:
            self.params.append(LJParams(0.0, 1.0, 1.0))

        var p = LJParams(epsilon, sigma, rc)
        self.params[self.n_types * itype + jtype] = p
        self.params[self.n_types * jtype + itype] = p
        if rc > self._cutoff:
            self._cutoff = rc

    fn cutoff(self) -> Float64:
        return self._cutoff

    fn short_cutoff(self) -> Float64:
        return 0.0  # LJ is two-body only

    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        """
        Accumulate LJ forces into atoms.f, return total potential energy.

        Parallelized over local atom i. Each thread i writes exclusively to
        atoms.f[3*i .. 3*i+2] — no locks needed.
        Energy for each pair is halved at the end (full list double-counts).
        """
        var nlocal = atoms.nlocal
        var nt = self.n_types
        var energies = List[Float64](capacity=nlocal)
        for _ in range(nlocal):
            energies.append(0.0)

        @parameter
        fn compute_atom(i: Int):
            var xi = atoms.x[3 * i]
            var yi = atoms.x[3 * i + 1]
            var zi = atoms.x[3 * i + 2]
            var itype = atoms.type_id[i]

            var fxi: Float64 = 0.0
            var fyi: Float64 = 0.0
            var fzi: Float64 = 0.0
            var ei: Float64 = 0.0

            var start = nlist.offsets[i]
            var end = nlist.offsets[i + 1]
            for nb in range(start, end):
                var j = nlist.neighbors[nb]
                var jtype = atoms.type_id[j]
                var p = self.params[nt * itype + jtype]

                var dx = xi - atoms.x[3 * j]
                var dy = yi - atoms.x[3 * j + 1]
                var dz = zi - atoms.x[3 * j + 2]
                var rsq = dx * dx + dy * dy + dz * dz

                if rsq < p.rc_sq:
                    var r2inv = 1.0 / rsq
                    var r6inv = r2inv * r2inv * r2inv
                    var fpair = (p.lj1 * r6inv - p.lj2) * r6inv * r2inv
                    fxi += dx * fpair
                    fyi += dy * fpair
                    fzi += dz * fpair
                    ei += r6inv * (p.lj3 * r6inv - p.lj4) - p.energy_shift

            atoms.f[3 * i]     += fxi
            atoms.f[3 * i + 1] += fyi
            atoms.f[3 * i + 2] += fzi
            energies[i] = ei

        parallelize[compute_atom](nlocal)

        var total: Float64 = 0.0
        for e in energies:
            total += e[]
        return total * 0.5  # correct for double-counting
