from std.algorithm import parallelize
from std.math import sqrt, exp
from atom import Atoms
from neighbor import NeighborList
from pair_style import PairStyle

# Coulomb constant in eV·Å / e²  (LAMMPS "metal" units)
comptime QQR2E: Float64 = 14.3996


struct VashishtaParam(ImplicitlyCopyable, Movable):
    """
    All parameters for a Vashishta (i,j,k) triplet entry, including
    precomputed derived quantities to avoid redundant work in the inner loop.

    Two-body potential between types i and j (from the i-j-j entry):
        V2(r) = H/r^η  +  zizj·e²/r · exp(-r/λ₁)
                        -  D · exp(-r/λ₄) / r⁴
                        -  W / r⁶
    with a smooth tail correction enforcing V2(rc) = 0 and dV2/dr(rc) = 0.

    Three-body potential (apex atom i, legs j and k):
        V3(rij, rik, θ) = B · exp(γ/(rij-r₀) + γ/(rik-r₀)) · (cosθ - cosθ₀)²
                         / [1 + C·(cosθ - cosθ₀)²]
    Only active when rij < r₀ and rik < r₀.

    Precomputed quantities match LAMMPS setup_params() (pair_vashishta.cpp:432).
    """
    var bigh: Float64      # H  (repulsive coefficient)
    var eta: Float64       # η  (repulsive exponent)
    var zi: Float64        # partial charge on species i [e]
    var zj: Float64        # partial charge on species j [e]
    var lambda1: Float64   # λ₁ (Coulomb screening length) [Å]
    var bigd: Float64      # D  (charge-dipole coefficient)
    var lambda4: Float64   # λ₄ (charge-dipole screening length) [Å]
    var bigw: Float64      # W  (van der Waals coefficient)
    var cut: Float64       # two-body cutoff [Å]
    var bigb: Float64      # B
    var gamma: Float64     # γ
    var r0: Float64        # r₀ (three-body cutoff) [Å]
    var bigc: Float64      # C  (angle-penalty damping)
    var costheta: Float64  # cos(θ₀) equilibrium angle cosine
    # Precomputed (mirror LAMMPS naming)
    var heta: Float64      # H · η
    var zizj: Float64      # zi · zj · QQR2E  [eV·Å]
    var lam1inv: Float64   # 1/λ₁
    var mbigd: Float64     # D (copy)
    var lam4inv: Float64   # 1/λ₄
    var big2b: Float64     # 2 · B
    var big6w: Float64     # 6 · W
    var cutsq: Float64     # cut²
    var cutsq2: Float64    # r₀²  (three-body gate)
    var dvrc: Float64      # -dV2/dr evaluated at rc
    var c0: Float64        # rc · dvrc − V2(rc)  (energy shift constant)

    fn __init__(
        out self,
        bigh: Float64, eta: Float64,
        zi: Float64, zj: Float64, lambda1: Float64,
        bigd: Float64, lambda4: Float64,
        bigw: Float64, cut: Float64,
        bigb: Float64, gamma: Float64, r0: Float64,
        bigc: Float64, costheta: Float64,
    ):
        self.bigh = bigh;   self.eta = eta
        self.zi = zi;       self.zj = zj
        self.lambda1 = lambda1
        self.bigd = bigd;   self.lambda4 = lambda4
        self.bigw = bigw;   self.cut = cut
        self.bigb = bigb;   self.gamma = gamma
        self.r0 = r0;       self.bigc = bigc
        self.costheta = costheta

        self.heta   = bigh * eta
        self.zizj   = zi * zj * QQR2E
        self.lam1inv = 1.0 / lambda1 if lambda1 > 0.0 else 0.0
        self.mbigd  = bigd
        self.lam4inv = 1.0 / lambda4 if lambda4 > 0.0 else 0.0
        self.big2b  = 2.0 * bigb
        self.big6w  = 6.0 * bigw
        self.cutsq  = cut * cut
        self.cutsq2 = r0 * r0

        # Tail correction at r = rc (energy continuity)
        var rcinv  = 1.0 / cut
        var rc2inv = rcinv * rcinv
        var rc4inv = rc2inv * rc2inv
        var rc6inv = rc2inv * rc4inv
        var rceta  = rcinv ** eta
        var lam1rc = cut * self.lam1inv
        var lam4rc = cut * self.lam4inv
        var vrcc2  = self.zizj * rcinv * exp(-lam1rc)
        var vrcc3  = self.mbigd * rc4inv * exp(-lam4rc)
        var vrc    = bigh * rceta + vrcc2 - vrcc3 - bigw * rc6inv
        self.dvrc = (
            vrcc3 * (4.0 * rcinv + self.lam4inv)
            + self.big6w * rc6inv * rcinv
            - self.heta * rceta * rcinv
            - vrcc2 * (rcinv + self.lam1inv)
        )
        self.c0 = cut * self.dvrc - vrc


struct TwobodyResult(ImplicitlyCopyable, Movable):
    var fforce: Float64   # scalar: multiply by displacement to get force vector
    var energy: Float64

    fn __init__(out self, fforce: Float64, energy: Float64):
        self.fforce = fforce
        self.energy = energy


fn _twobody(param: VashishtaParam, rsq: Float64) -> TwobodyResult:
    """
    Scalar force factor and energy for the Vashishta two-body term.
    fforce: multiply by (ri - rj) to get force vector on i.
    Matches pair_vashishta.cpp:484 (twobody).
    """
    var r      = sqrt(rsq)
    var rinvsq = 1.0 / rsq
    var r4inv  = rinvsq * rinvsq
    var r6inv  = rinvsq * r4inv
    var reta   = r ** (-param.eta)
    var lam1r  = r * param.lam1inv
    var lam4r  = r * param.lam4inv
    var vc2    = param.zizj * exp(-lam1r) / r
    var vc3    = param.mbigd * r4inv * exp(-lam4r)

    var fforce = (
        param.dvrc * r
        - (4.0 * vc3 + lam4r * vc3 + param.big6w * r6inv
           - param.heta * reta - vc2 - lam1r * vc2)
    ) * rinvsq

    var eng = (param.bigh * reta + vc2 - vc3
               - param.bigw * r6inv - r * param.dvrc + param.c0)
    return TwobodyResult(fforce, eng)


struct ThreebodyResult(ImplicitlyCopyable, Movable):
    """Return type for _threebody to avoid bare tuple unpacking."""
    var fj_x: Float64; var fj_y: Float64; var fj_z: Float64
    var fk_x: Float64; var fk_y: Float64; var fk_z: Float64
    var energy: Float64

    fn __init__(out self, fj_x: Float64, fj_y: Float64, fj_z: Float64,
                           fk_x: Float64, fk_y: Float64, fk_z: Float64,
                           energy: Float64):
        self.fj_x = fj_x; self.fj_y = fj_y; self.fj_z = fj_z
        self.fk_x = fk_x; self.fk_y = fk_y; self.fk_z = fk_z
        self.energy = energy


fn _threebody(
    paramij: VashishtaParam,
    paramik: VashishtaParam,
    paramijk: VashishtaParam,
    rsq1: Float64, rsq2: Float64,
    delr1_x: Float64, delr1_y: Float64, delr1_z: Float64,
    delr2_x: Float64, delr2_y: Float64, delr2_z: Float64,
) -> ThreebodyResult:
    """
    Three-body contribution for triplet (apex=i, j, k).
    delr1 = rj - ri,  delr2 = rk - ri.
    Returns forces on j and k; force on i = -(fj + fk) by Newton's 3rd.
    Matches pair_vashishta.cpp:509 (threebody).
    """
    var r1        = sqrt(rsq1)
    var rinvsq1   = 1.0 / rsq1
    var rainv1    = 1.0 / (r1 - paramij.r0)
    var gsrainv1  = paramij.gamma * rainv1
    var gsrainvsq1 = gsrainv1 * rainv1 / r1
    var expg1     = exp(gsrainv1)

    var r2        = sqrt(rsq2)
    var rinvsq2   = 1.0 / rsq2
    var rainv2    = 1.0 / (r2 - paramik.r0)
    var gsrainv2  = paramik.gamma * rainv2
    var gsrainvsq2 = gsrainv2 * rainv2 / r2
    var expg2     = exp(gsrainv2)

    var rinv12   = 1.0 / (r1 * r2)
    var cs       = (delr1_x * delr2_x + delr1_y * delr2_y + delr1_z * delr2_z) * rinv12
    var delcs    = cs - paramijk.costheta
    var delcssq  = delcs * delcs
    var pcsinv   = paramijk.bigc * delcssq + 1.0
    var pcsinvsq = pcsinv * pcsinv
    var pcs      = delcssq / pcsinv

    var facexp   = expg1 * expg2
    var facrad   = paramijk.bigb * facexp * pcs
    var frad1    = facrad * gsrainvsq1
    var frad2    = facrad * gsrainvsq2
    var facang   = paramijk.big2b * facexp * delcs / pcsinvsq
    var facang12 = rinv12 * facang
    var csfacang = cs * facang
    var csfac1   = rinvsq1 * csfacang

    var res = ThreebodyResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    res.fj_x = delr1_x * (frad1 + csfac1) - delr2_x * facang12
    res.fj_y = delr1_y * (frad1 + csfac1) - delr2_y * facang12
    res.fj_z = delr1_z * (frad1 + csfac1) - delr2_z * facang12

    var csfac2 = rinvsq2 * csfacang
    res.fk_x = delr2_x * (frad2 + csfac2) - delr1_x * facang12
    res.fk_y = delr2_y * (frad2 + csfac2) - delr1_y * facang12
    res.fk_z = delr2_z * (frad2 + csfac2) - delr1_z * facang12
    res.energy = facrad
    return res


struct PairVashishta(PairStyle):
    """
    Vashishta interatomic potential (two-body + three-body).

    Parameters: flat list of n_types³ VashishtaParam, indexed
    (itype * nt + jtype) * nt + ktype, matching LAMMPS elem3param[i][j][k].

    Parallelization strategy:
      Two-body: full neighbor list, parallelize over i. Each thread writes only
                to f[i]. Energy halved (each pair visited twice). Safe.
      Three-body: SERIAL with Newton's 3rd law. Apex loop over i, legs j<k.
                  Each triplet visited once; forces applied to i, j, and k.
                  Applying to f[j], f[k] requires serial access — adding GPU
                  atomics here is the next step for a GPU port.
    """
    var n_types: Int
    var params: List[VashishtaParam]
    var _cutoff: Float64
    var _r0_max: Float64

    fn __init__(out self, n_types: Int):
        self.n_types = n_types
        self.params = List[VashishtaParam](capacity=n_types * n_types * n_types)
        self._cutoff = 0.0
        self._r0_max = 0.0

    fn _idx(self, i: Int, j: Int, k: Int) -> Int:
        return (i * self.n_types + j) * self.n_types + k

    fn set_param(mut self, itype: Int, jtype: Int, ktype: Int, p: VashishtaParam):
        """Set parameters for triplet (itype, jtype, ktype). Call for all n³ combinations."""
        while len(self.params) < self.n_types * self.n_types * self.n_types:
            self.params.append(p)  # placeholder
        self.params[self._idx(itype, jtype, ktype)] = p
        if p.cut > self._cutoff:
            self._cutoff = p.cut
        if p.r0 > self._r0_max:
            self._r0_max = p.r0

    fn cutoff(self) -> Float64:
        return self._cutoff

    fn short_cutoff(self) -> Float64:
        return self._r0_max

    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        var nlocal = atoms.nlocal

        # ----------------------------------------------------------------
        # Two-body pass — parallel over i, full list, energy /2
        # ----------------------------------------------------------------
        var energies_2b = List[Float64](capacity=nlocal)
        for _ in range(nlocal):
            energies_2b.append(0.0)

        @parameter
        fn twobody_atom(i: Int):
            var xi = atoms.x[3 * i];  var yi = atoms.x[3 * i + 1];  var zi = atoms.x[3 * i + 2]
            var itype = atoms.type_id[i]
            var fxi: Float64 = 0.0;  var fyi: Float64 = 0.0;  var fzi: Float64 = 0.0
            var ei: Float64 = 0.0

            var start = nlist.offsets[i]
            var end   = nlist.offsets[i + 1]
            for nb in range(start, end):
                var j = nlist.neighbors[nb]
                var jtype = atoms.type_id[j]
                var p = self.params[self._idx(itype, jtype, jtype)]

                var dx = xi - atoms.x[3 * j];  var dy = yi - atoms.x[3 * j + 1];  var dz = zi - atoms.x[3 * j + 2]
                var rsq = dx * dx + dy * dy + dz * dz
                if rsq >= p.cutsq:
                    continue
                var tb = _twobody(p, rsq)
                fxi += dx * tb.fforce
                fyi += dy * tb.fforce
                fzi += dz * tb.fforce
                ei += tb.energy

            atoms.f[3 * i]     += fxi
            atoms.f[3 * i + 1] += fyi
            atoms.f[3 * i + 2] += fzi
            energies_2b[i] = ei

        parallelize[twobody_atom](nlocal)

        var total_energy: Float64 = 0.0
        for e in energies_2b:
            total_energy += e * 0.5  # halve for double-count

        # ----------------------------------------------------------------
        # Three-body pass — SERIAL, Newton's 3rd law, energy counted once
        # ----------------------------------------------------------------
        for i in range(nlocal):
            var xi = atoms.x[3 * i];  var yi = atoms.x[3 * i + 1];  var zi = atoms.x[3 * i + 2]
            var itype = atoms.type_id[i]

            var ss = nlist.short_offsets[i]
            var se = nlist.short_offsets[i + 1]
            var nshort = se - ss

            for jj in range(nshort):
                var j = nlist.short_neighbors[ss + jj]
                var jtype = atoms.type_id[j]
                var pij = self.params[self._idx(itype, jtype, jtype)]

                var d1x = atoms.x[3 * j]     - xi
                var d1y = atoms.x[3 * j + 1] - yi
                var d1z = atoms.x[3 * j + 2] - zi
                var rsq1 = d1x * d1x + d1y * d1y + d1z * d1z
                if rsq1 >= pij.cutsq2:
                    continue

                for kk in range(jj + 1, nshort):
                    var k = nlist.short_neighbors[ss + kk]
                    var ktype = atoms.type_id[k]
                    var pik  = self.params[self._idx(itype, ktype, ktype)]
                    var pijk = self.params[self._idx(itype, jtype, ktype)]

                    var d2x = atoms.x[3 * k]     - xi
                    var d2y = atoms.x[3 * k + 1] - yi
                    var d2z = atoms.x[3 * k + 2] - zi
                    var rsq2 = d2x * d2x + d2y * d2y + d2z * d2z
                    if rsq2 >= pik.cutsq2:
                        continue

                    var res = _threebody(pij, pik, pijk,
                                        rsq1, rsq2,
                                        d1x, d1y, d1z,
                                        d2x, d2y, d2z)

                    # Newton's 3rd: fi = -(fj + fk)
                    atoms.f[3 * i]     -= res.fj_x + res.fk_x
                    atoms.f[3 * i + 1] -= res.fj_y + res.fk_y
                    atoms.f[3 * i + 2] -= res.fj_z + res.fk_z
                    atoms.f[3 * j]     += res.fj_x
                    atoms.f[3 * j + 1] += res.fj_y
                    atoms.f[3 * j + 2] += res.fj_z
                    atoms.f[3 * k]     += res.fk_x
                    atoms.f[3 * k + 1] += res.fk_y
                    atoms.f[3 * k + 2] += res.fk_z
                    total_energy += res.energy  # counted once per apex angle

        return total_energy
