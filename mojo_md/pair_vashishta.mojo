from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, exp, log, ceildiv
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.neighbor import NeighborList
from mojo_md.pair_style import PairStyle

# Coulomb constant in eV·Å / e²  (LAMMPS "metal" units)
comptime QQR2E: Float64 = 14.3996

# Flat layout: 25 floats per (i, j, k) triplet entry.
comptime _VS:           Int = 25
comptime _VS_BIGH:      Int = 0
comptime _VS_ETA:       Int = 1
comptime _VS_ZI:        Int = 2
comptime _VS_ZJ:        Int = 3
comptime _VS_LAMBDA1:   Int = 4
comptime _VS_BIGD:      Int = 5
comptime _VS_LAMBDA4:   Int = 6
comptime _VS_BIGW:      Int = 7
comptime _VS_CUT:       Int = 8
comptime _VS_BIGB:      Int = 9
comptime _VS_GAMMA:     Int = 10
comptime _VS_R0:        Int = 11
comptime _VS_BIGC:      Int = 12
comptime _VS_COSTH:     Int = 13
comptime _VS_HETA:      Int = 14
comptime _VS_ZIZJ:      Int = 15
comptime _VS_LAM1INV:   Int = 16
comptime _VS_MBIGD:     Int = 17
comptime _VS_LAM4INV:   Int = 18
comptime _VS_BIG2B:     Int = 19
comptime _VS_BIG6W:     Int = 20
comptime _VS_CUTSQ:     Int = 21
comptime _VS_CUTSQ2:    Int = 22
comptime _VS_DVRC:      Int = 23
comptime _VS_C0:        Int = 24

comptime _BLOCK_SIZE: Int = 256


struct VashishtaParam(ImplicitlyCopyable, Movable):
    """All 25 pre-computed Vashishta parameters for one (i,j,k) triplet.

    Only the direct-field constructor exists here — no math (no exp/pow).
    This keeps the struct GPU-safe: Metal compiles ALL methods of any struct
    used in a kernel, so CPU-only math must live outside the struct.
    Call `make_vashishta_param(...)` on the CPU to compute derived fields and
    obtain a fully initialised `VashishtaParam`.
    """
    var bigh: Float64
    var eta: Float64
    var zi: Float64
    var zj: Float64
    var lambda1: Float64
    var bigd: Float64
    var lambda4: Float64
    var bigw: Float64
    var cut: Float64
    var bigb: Float64
    var gamma: Float64
    var r0: Float64
    var bigc: Float64
    var costheta: Float64
    var heta: Float64
    var zizj: Float64
    var lam1inv: Float64
    var mbigd: Float64
    var lam4inv: Float64
    var big2b: Float64
    var big6w: Float64
    var cutsq: Float64
    var cutsq2: Float64
    var dvrc: Float64
    var c0: Float64

    fn __init__(
        out self,
        bigh: Float64, eta: Float64, zi: Float64, zj: Float64,
        lambda1: Float64, bigd: Float64, lambda4: Float64, bigw: Float64,
        cut: Float64, bigb: Float64, gamma: Float64, r0: Float64,
        bigc: Float64, costheta: Float64,
        heta: Float64, zizj: Float64, lam1inv: Float64, mbigd: Float64,
        lam4inv: Float64, big2b: Float64, big6w: Float64, cutsq: Float64,
        cutsq2: Float64, dvrc: Float64, c0: Float64,
    ):
        """Direct-field constructor. GPU-safe: pure field copies, no math."""
        self.bigh = bigh; self.eta = eta; self.zi = zi; self.zj = zj
        self.lambda1 = lambda1; self.bigd = bigd; self.lambda4 = lambda4
        self.bigw = bigw; self.cut = cut; self.bigb = bigb; self.gamma = gamma
        self.r0 = r0; self.bigc = bigc; self.costheta = costheta
        self.heta = heta; self.zizj = zizj; self.lam1inv = lam1inv
        self.mbigd = mbigd; self.lam4inv = lam4inv; self.big2b = big2b
        self.big6w = big6w; self.cutsq = cutsq; self.cutsq2 = cutsq2
        self.dvrc = dvrc; self.c0 = c0

    @staticmethod
    fn _from_flat(p_ptr: UnsafePointer[Float64, MutAnyOrigin], off: Int) -> VashishtaParam:
        """Reconstitute a VashishtaParam from the flat float buffer.
        Uses the direct-field constructor — no exp/pow, GPU-safe."""
        return VashishtaParam(
            p_ptr[off + _VS_BIGH], p_ptr[off + _VS_ETA],
            p_ptr[off + _VS_ZI],   p_ptr[off + _VS_ZJ],
            p_ptr[off + _VS_LAMBDA1], p_ptr[off + _VS_BIGD],
            p_ptr[off + _VS_LAMBDA4], p_ptr[off + _VS_BIGW],
            p_ptr[off + _VS_CUT],  p_ptr[off + _VS_BIGB],
            p_ptr[off + _VS_GAMMA], p_ptr[off + _VS_R0],
            p_ptr[off + _VS_BIGC], p_ptr[off + _VS_COSTH],
            p_ptr[off + _VS_HETA], p_ptr[off + _VS_ZIZJ],
            p_ptr[off + _VS_LAM1INV], p_ptr[off + _VS_MBIGD],
            p_ptr[off + _VS_LAM4INV], p_ptr[off + _VS_BIG2B],
            p_ptr[off + _VS_BIG6W], p_ptr[off + _VS_CUTSQ],
            p_ptr[off + _VS_CUTSQ2], p_ptr[off + _VS_DVRC],
            p_ptr[off + _VS_C0],
        )


fn make_vashishta_param(
    bigh: Float64, eta: Float64,
    zi: Float64, zj: Float64, lambda1: Float64,
    bigd: Float64, lambda4: Float64,
    bigw: Float64, cut: Float64,
    bigb: Float64, gamma: Float64, r0: Float64,
    bigc: Float64, costheta: Float64,
) -> VashishtaParam:
    """CPU-only: compute all 25 derived fields and return a VashishtaParam.
    Matches LAMMPS setup_params() (pair_vashishta.cpp:432).
    Not called from GPU kernels — Metal never compiles this function."""
    var heta    = bigh * eta
    var zizj    = zi * zj * QQR2E
    var lam1inv = 1.0 / lambda1 if lambda1 > 0.0 else 0.0
    var mbigd   = bigd
    var lam4inv = 1.0 / lambda4 if lambda4 > 0.0 else 0.0
    var big2b   = 2.0 * bigb
    var big6w   = 6.0 * bigw
    var cutsq   = cut * cut
    var cutsq2  = r0 * r0

    var rcinv  = 1.0 / cut
    var rc2inv = rcinv * rcinv
    var rc4inv = rc2inv * rc2inv
    var rc6inv = rc2inv * rc4inv
    var rceta  = rcinv ** eta
    var lam1rc = cut * lam1inv
    var lam4rc = cut * lam4inv
    var vrcc2  = zizj * rcinv * exp(-lam1rc)
    var vrcc3  = mbigd * rc4inv * exp(-lam4rc)
    var vrc    = bigh * rceta + vrcc2 - vrcc3 - bigw * rc6inv
    var dvrc   = (
        vrcc3 * (4.0 * rcinv + lam4inv)
        + big6w * rc6inv * rcinv
        - heta * rceta * rcinv
        - vrcc2 * (rcinv + lam1inv)
    )
    var c0 = cut * dvrc - vrc

    return VashishtaParam(
        bigh, eta, zi, zj, lambda1, bigd, lambda4, bigw,
        cut, bigb, gamma, r0, bigc, costheta,
        heta, zizj, lam1inv, mbigd, lam4inv, big2b, big6w,
        cutsq, cutsq2, dvrc, c0,
    )


struct TwobodyResult(ImplicitlyCopyable, Movable):
    var fforce: Float64
    var energy: Float64

    fn __init__(out self, fforce: Float64, energy: Float64):
        self.fforce = fforce
        self.energy = energy


struct TwobodyResultF32(ImplicitlyCopyable, Movable):
    var fforce: Float32
    var energy: Float32

    fn __init__(out self, fforce: Float32, energy: Float32):
        self.fforce = fforce
        self.energy = energy


fn _twobody(param: VashishtaParam, rsq: Float64) -> TwobodyResult:
    """Scalar force factor and energy for the Vashishta two-body term.
    fforce: multiply by (ri - rj) to get force vector on i. Matches
    pair_vashishta.cpp:484 (twobody)."""
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
    """Three-body contribution for triplet (apex=i, j, k). delr1 = rj - ri,
    delr2 = rk - ri. Returns forces on j and k; force on i = -(fj + fk) by
    Newton's 3rd. Matches pair_vashishta.cpp:509 (threebody)."""
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


# ---------------------------------------------------------------------------
# Shared kernel bodies — same code on CPU and GPU.
# ---------------------------------------------------------------------------

@always_inline
fn _idx(i: Int, j: Int, k: Int, n_types: Int) -> Int:
    return ((i * n_types + j) * n_types + k) * _VS


fn _vashishta_2body_body(
    i:           Int,
    x_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    f_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    pe_ptr:      UnsafePointer[Float64, MutAnyOrigin],   # nlocal
    type_id_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    p_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    off_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
    nb_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    n_types:     Int,
):
    var xi = x_ptr[3 * i]; var yi = x_ptr[3 * i + 1]; var zi = x_ptr[3 * i + 2]
    var itype = Int(type_id_ptr[i])

    var fxi: Float64 = 0.0
    var fyi: Float64 = 0.0
    var fzi: Float64 = 0.0
    var ei:  Float64 = 0.0

    var start = Int(off_ptr[i])
    var end   = Int(off_ptr[i + 1])

    for nb in range(start, end):
        var j = Int(nb_ptr[nb])
        var jtype = Int(type_id_ptr[j])
        var off_ij = _idx(itype, jtype, jtype, n_types)
        var pij = VashishtaParam._from_flat(p_ptr, off_ij)
        var dx = xi - x_ptr[3 * j]
        var dy = yi - x_ptr[3 * j + 1]
        var dz = zi - x_ptr[3 * j + 2]
        var rsq = dx * dx + dy * dy + dz * dz
        if rsq >= pij.cutsq:
            continue
        var tb = _twobody(pij, rsq)
        fxi += dx * tb.fforce
        fyi += dy * tb.fforce
        fzi += dz * tb.fforce
        ei  += tb.energy

    f_ptr[3 * i]     += fxi
    f_ptr[3 * i + 1] += fyi
    f_ptr[3 * i + 2] += fzi
    pe_ptr[i] = ei * 0.5  # full list double-counts


fn _vashishta_3body_body(
    m:            Int,
    nlocal:       Int,
    x_ptr:        UnsafePointer[Float64, MutAnyOrigin],
    f_ptr:        UnsafePointer[Float64, MutAnyOrigin],
    pe_ptr:       UnsafePointer[Float64, MutAnyOrigin],   # nlocal — m as apex counts the energy
    type_id_ptr:  UnsafePointer[Int32,   MutAnyOrigin],
    p_ptr:        UnsafePointer[Float64, MutAnyOrigin],
    soff_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
    snb_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    n_types:      Int,
):
    """Owner-computes 3-body: thread for atom m sums all forces on m from
    triplets where m is the apex OR a side atom. Each thread writes only to
    f[3*m..] and pe_atom[m] — no atomics, parallel-safe on CPU and GPU.
    Matches the apex-centric algorithm with j-position < k-position
    canonicalization, so results are bit-identical to the original."""
    var xm = x_ptr[3 * m]; var ym = x_ptr[3 * m + 1]; var zm = x_ptr[3 * m + 2]
    var mtype = Int(type_id_ptr[m])

    var fxm: Float64 = 0.0;  var fym: Float64 = 0.0;  var fzm: Float64 = 0.0
    var em:  Float64 = 0.0

    var ss_m = Int(soff_ptr[m])
    var se_m = Int(soff_ptr[m + 1])
    var nshort_m = se_m - ss_m

    # ---- Case A: m is the apex ----
    for jj in range(nshort_m):
        var j = Int(snb_ptr[ss_m + jj])
        var jtype = Int(type_id_ptr[j])
        var pmj = VashishtaParam._from_flat(p_ptr, _idx(mtype, jtype, jtype, n_types))
        var d1x = x_ptr[3 * j]     - xm
        var d1y = x_ptr[3 * j + 1] - ym
        var d1z = x_ptr[3 * j + 2] - zm
        var rsq1 = d1x * d1x + d1y * d1y + d1z * d1z
        if rsq1 >= pmj.cutsq2:
            continue
        for kk in range(jj + 1, nshort_m):
            var k = Int(snb_ptr[ss_m + kk])
            var ktype = Int(type_id_ptr[k])
            var pmk = VashishtaParam._from_flat(p_ptr, _idx(mtype, ktype, ktype, n_types))
            var d2x = x_ptr[3 * k]     - xm
            var d2y = x_ptr[3 * k + 1] - ym
            var d2z = x_ptr[3 * k + 2] - zm
            var rsq2 = d2x * d2x + d2y * d2y + d2z * d2z
            if rsq2 >= pmk.cutsq2:
                continue
            var pmjk = VashishtaParam._from_flat(p_ptr, _idx(mtype, jtype, ktype, n_types))
            var res = _threebody(pmj, pmk, pmjk, rsq1, rsq2,
                                 d1x, d1y, d1z, d2x, d2y, d2z)
            fxm -= res.fj_x + res.fk_x
            fym -= res.fj_y + res.fk_y
            fzm -= res.fj_z + res.fk_z
            em  += res.energy

    # ---- Case B: m is a SIDE atom of a triplet centered at A ----
    for ii in range(nshort_m):
        var A = Int(snb_ptr[ss_m + ii])
        if A >= nlocal:
            continue  # ghosts can't be apex
        var Atype = Int(type_id_ptr[A])
        var ss_A = Int(soff_ptr[A])
        var se_A = Int(soff_ptr[A + 1])
        var nshort_A = se_A - ss_A

        # m's position in A's short list (must exist by symmetry)
        var pos_m = -1
        for nn in range(nshort_A):
            if Int(snb_ptr[ss_A + nn]) == m:
                pos_m = nn
                break
        if pos_m < 0:
            continue

        var xA = x_ptr[3 * A];  var yA = x_ptr[3 * A + 1];  var zA = x_ptr[3 * A + 2]
        var d_Am_x = xm - xA;  var d_Am_y = ym - yA;  var d_Am_z = zm - zA
        var rsq_Am = d_Am_x * d_Am_x + d_Am_y * d_Am_y + d_Am_z * d_Am_z
        var pAm = VashishtaParam._from_flat(p_ptr, _idx(Atype, mtype, mtype, n_types))
        if rsq_Am >= pAm.cutsq2:
            continue

        # B-positions < pos_m → m is "k" → use res.fk
        for nn in range(pos_m):
            var B = Int(snb_ptr[ss_A + nn])
            var Btype = Int(type_id_ptr[B])
            var pAB = VashishtaParam._from_flat(p_ptr, _idx(Atype, Btype, Btype, n_types))
            var d_AB_x = x_ptr[3 * B]     - xA
            var d_AB_y = x_ptr[3 * B + 1] - yA
            var d_AB_z = x_ptr[3 * B + 2] - zA
            var rsq_AB = d_AB_x * d_AB_x + d_AB_y * d_AB_y + d_AB_z * d_AB_z
            if rsq_AB >= pAB.cutsq2:
                continue
            var pABm = VashishtaParam._from_flat(p_ptr, _idx(Atype, Btype, mtype, n_types))
            var res = _threebody(pAB, pAm, pABm,
                                 rsq_AB, rsq_Am,
                                 d_AB_x, d_AB_y, d_AB_z,
                                 d_Am_x, d_Am_y, d_Am_z)
            fxm += res.fk_x
            fym += res.fk_y
            fzm += res.fk_z

        # C-positions > pos_m → m is "j" → use res.fj
        for nn in range(pos_m + 1, nshort_A):
            var C = Int(snb_ptr[ss_A + nn])
            var Ctype = Int(type_id_ptr[C])
            var pAC = VashishtaParam._from_flat(p_ptr, _idx(Atype, Ctype, Ctype, n_types))
            var d_AC_x = x_ptr[3 * C]     - xA
            var d_AC_y = x_ptr[3 * C + 1] - yA
            var d_AC_z = x_ptr[3 * C + 2] - zA
            var rsq_AC = d_AC_x * d_AC_x + d_AC_y * d_AC_y + d_AC_z * d_AC_z
            if rsq_AC >= pAC.cutsq2:
                continue
            var pAmC = VashishtaParam._from_flat(p_ptr, _idx(Atype, mtype, Ctype, n_types))
            var res = _threebody(pAm, pAC, pAmC,
                                 rsq_Am, rsq_AC,
                                 d_Am_x, d_Am_y, d_Am_z,
                                 d_AC_x, d_AC_y, d_AC_z)
            fxm += res.fj_x
            fym += res.fj_y
            fzm += res.fj_z

    f_ptr[3 * m]     += fxm
    f_ptr[3 * m + 1] += fym
    f_ptr[3 * m + 2] += fzm
    # 2-body kernel writes pe_ptr[m] (halved); add 3-body apex energy here.
    pe_ptr[m] += em


# ---------------------------------------------------------------------------
# GPU kernels — Float32 (Metal/MPS). Same physics as the CPU bodies above,
# but all floats are Float32. Parameters are read directly from the Float32
# device buffer without going through VashishtaParam (which has Float64 fields).
# ---------------------------------------------------------------------------

@always_inline
fn _twobody_f32(
    heta: Float32, zizj: Float32, mbigd: Float32, big6w: Float32,
    bigw: Float32, dvrc: Float32, c0: Float32, lam1inv: Float32,
    lam4inv: Float32, eta: Float32, bigh: Float32, rsq: Float32,
) -> TwobodyResultF32:
    var r      = sqrt(rsq)
    var rinvsq = Float32(1.0) / rsq
    var r4inv  = rinvsq * rinvsq
    var r6inv  = rinvsq * r4inv
    var reta   = exp(-eta * log(r))   # r^{-eta}, Float32-safe
    var lam1r  = r * lam1inv;  var lam4r = r * lam4inv
    var vc2    = zizj * exp(-lam1r) / r
    var vc3    = mbigd * r4inv * exp(-lam4r)
    var fforce = (dvrc * r - (Float32(4.0) * vc3 + lam4r * vc3 + big6w * r6inv
                  - heta * reta - vc2 - lam1r * vc2)) * rinvsq
    var eng    = bigh * reta + vc2 - vc3 - bigw * r6inv - r * dvrc + c0
    return TwobodyResultF32(fforce, eng)


fn _vashishta_2body_kernel(
    x_ptr:       UnsafePointer[Float32, MutAnyOrigin],
    f_ptr:       UnsafePointer[Float32, MutAnyOrigin],
    pe_ptr:      UnsafePointer[Float32, MutAnyOrigin],
    type_id_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    p_ptr:       UnsafePointer[Float32, MutAnyOrigin],
    off_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
    nb_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    n_types:     Int,
    nlocal:      Int,
):
    var i = Int(global_idx.x)
    if i >= nlocal:
        return
    var xi = x_ptr[3 * i]; var yi = x_ptr[3 * i + 1]; var zi = x_ptr[3 * i + 2]
    var itype = Int(type_id_ptr[i])
    var fxi: Float32 = 0.0; var fyi: Float32 = 0.0; var fzi: Float32 = 0.0
    var ei:  Float32 = 0.0
    var start = Int(off_ptr[i]); var end = Int(off_ptr[i + 1])
    for nb in range(start, end):
        var j = Int(nb_ptr[nb])
        if j < 0:
            break   # GPU strided list sentinel
        var jtype = Int(type_id_ptr[j])
        var off_ij = ((itype * n_types + jtype) * n_types + jtype) * _VS
        var cutsq = p_ptr[off_ij + _VS_CUTSQ]
        var dx = xi - x_ptr[3 * j]; var dy = yi - x_ptr[3 * j + 1]; var dz = zi - x_ptr[3 * j + 2]
        var rsq = dx * dx + dy * dy + dz * dz
        if rsq >= cutsq:
            continue
        var res = _twobody_f32(
            p_ptr[off_ij + _VS_HETA],   p_ptr[off_ij + _VS_ZIZJ],
            p_ptr[off_ij + _VS_MBIGD],  p_ptr[off_ij + _VS_BIG6W],
            p_ptr[off_ij + _VS_BIGW],   p_ptr[off_ij + _VS_DVRC],
            p_ptr[off_ij + _VS_C0],     p_ptr[off_ij + _VS_LAM1INV],
            p_ptr[off_ij + _VS_LAM4INV], p_ptr[off_ij + _VS_ETA],
            p_ptr[off_ij + _VS_BIGH],   rsq,
        )
        fxi += dx * res.fforce; fyi += dy * res.fforce; fzi += dz * res.fforce
        ei  += res.energy
    f_ptr[3 * i] += fxi; f_ptr[3 * i + 1] += fyi; f_ptr[3 * i + 2] += fzi
    pe_ptr[i] = ei * Float32(0.5)


fn _vashishta_3body_kernel(
    x_ptr:        UnsafePointer[Float32, MutAnyOrigin],
    f_ptr:        UnsafePointer[Float32, MutAnyOrigin],
    pe_ptr:       UnsafePointer[Float32, MutAnyOrigin],
    type_id_ptr:  UnsafePointer[Int32,   MutAnyOrigin],
    p_ptr:        UnsafePointer[Float32, MutAnyOrigin],
    soff_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
    snb_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    n_types:      Int,
    nlocal:       Int,
):
    var m = Int(global_idx.x)
    if m >= nlocal:
        return
    var xm = x_ptr[3 * m]; var ym = x_ptr[3 * m + 1]; var zm = x_ptr[3 * m + 2]
    var mtype = Int(type_id_ptr[m])
    var fxm: Float32 = 0.0;  var fym: Float32 = 0.0;  var fzm: Float32 = 0.0
    var em:  Float32 = 0.0
    var ss_m = Int(soff_ptr[m]); var se_m = Int(soff_ptr[m + 1])
    var nshort_m = se_m - ss_m

    # ---- Case A: m is the apex ----
    for jj in range(nshort_m):
        var j = Int(snb_ptr[ss_m + jj])
        if j < 0:
            break   # GPU strided list sentinel
        var jtype = Int(type_id_ptr[j])
        var off_mj = ((mtype * n_types + jtype) * n_types + jtype) * _VS
        var d1x = x_ptr[3 * j]     - xm
        var d1y = x_ptr[3 * j + 1] - ym
        var d1z = x_ptr[3 * j + 2] - zm
        var rsq1 = d1x * d1x + d1y * d1y + d1z * d1z
        if rsq1 >= p_ptr[off_mj + _VS_CUTSQ2]:
            continue
        var r1      = sqrt(rsq1)
        var rainv1  = Float32(1.0) / (r1 - p_ptr[off_mj + _VS_R0])
        var gsrainv1 = p_ptr[off_mj + _VS_GAMMA] * rainv1
        var gsrainvsq1 = gsrainv1 * rainv1 / r1
        var expg1   = exp(gsrainv1)
        for kk in range(jj + 1, nshort_m):
            var k = Int(snb_ptr[ss_m + kk])
            if k < 0:
                break   # GPU strided list sentinel
            var ktype = Int(type_id_ptr[k])
            var off_mk = ((mtype * n_types + ktype) * n_types + ktype) * _VS
            var d2x = x_ptr[3 * k]     - xm
            var d2y = x_ptr[3 * k + 1] - ym
            var d2z = x_ptr[3 * k + 2] - zm
            var rsq2 = d2x * d2x + d2y * d2y + d2z * d2z
            if rsq2 >= p_ptr[off_mk + _VS_CUTSQ2]:
                continue
            var r2       = sqrt(rsq2)
            var rainv2   = Float32(1.0) / (r2 - p_ptr[off_mk + _VS_R0])
            var gsrainv2 = p_ptr[off_mk + _VS_GAMMA] * rainv2
            var gsrainvsq2 = gsrainv2 * rainv2 / r2
            var expg2    = exp(gsrainv2)
            var off_mjk  = ((mtype * n_types + jtype) * n_types + ktype) * _VS
            var bigb     = p_ptr[off_mjk + _VS_BIGB]
            var big2b    = p_ptr[off_mjk + _VS_BIG2B]
            var bigc     = p_ptr[off_mjk + _VS_BIGC]
            var costh    = p_ptr[off_mjk + _VS_COSTH]
            var rinv12   = Float32(1.0) / (r1 * r2)
            var cs       = (d1x * d2x + d1y * d2y + d1z * d2z) * rinv12
            var delcs    = cs - costh
            var delcssq  = delcs * delcs
            var pcsinv   = bigc * delcssq + Float32(1.0)
            var pcsinvsq = pcsinv * pcsinv
            var pcs      = delcssq / pcsinv
            var facexp   = expg1 * expg2
            var facrad   = bigb * facexp * pcs
            var frad1    = facrad * gsrainvsq1
            var frad2    = facrad * gsrainvsq2
            var facang   = big2b * facexp * delcs / pcsinvsq
            var facang12 = rinv12 * facang
            var csfacang = cs * facang
            var csfac1   = (Float32(1.0) / rsq1) * csfacang
            var csfac2   = (Float32(1.0) / rsq2) * csfacang
            fxm -= d1x * (frad1 + csfac1) - d2x * facang12
            fym -= d1y * (frad1 + csfac1) - d2y * facang12
            fzm -= d1z * (frad1 + csfac1) - d2z * facang12
            fxm -= d2x * (frad2 + csfac2) - d1x * facang12
            fym -= d2y * (frad2 + csfac2) - d1y * facang12
            fzm -= d2z * (frad2 + csfac2) - d1z * facang12
            em  += facrad

    # ---- Case B: m is a side atom of a triplet centered at A ----
    for ii in range(nshort_m):
        var A = Int(snb_ptr[ss_m + ii])
        if A < 0:
            break   # GPU strided list sentinel
        if A >= nlocal:
            continue
        var Atype = Int(type_id_ptr[A])
        var ss_A  = Int(soff_ptr[A]); var se_A = Int(soff_ptr[A + 1])
        var nshort_A = se_A - ss_A
        var pos_m = -1
        for nn in range(nshort_A):
            var entry = Int(snb_ptr[ss_A + nn])
            if entry < 0:
                break   # GPU strided list sentinel
            if entry == m:
                pos_m = nn
                break
        if pos_m < 0:
            continue
        var xA = x_ptr[3 * A]; var yA = x_ptr[3 * A + 1]; var zA = x_ptr[3 * A + 2]
        var d_Am_x = xm - xA;  var d_Am_y = ym - yA;  var d_Am_z = zm - zA
        var rsq_Am = d_Am_x * d_Am_x + d_Am_y * d_Am_y + d_Am_z * d_Am_z
        var off_Am = ((Atype * n_types + mtype) * n_types + mtype) * _VS
        if rsq_Am >= p_ptr[off_Am + _VS_CUTSQ2]:
            continue
        var r_Am     = sqrt(rsq_Am)
        var rainv_Am = Float32(1.0) / (r_Am - p_ptr[off_Am + _VS_R0])
        var gsrainv_Am = p_ptr[off_Am + _VS_GAMMA] * rainv_Am
        var gsrainvsq_Am = gsrainv_Am * rainv_Am / r_Am
        var expg_Am  = exp(gsrainv_Am)

        # B < pos_m → m is "k" in triplet (A, B, m)
        for nn in range(pos_m):
            var B = Int(snb_ptr[ss_A + nn])
            var Btype = Int(type_id_ptr[B])
            var off_AB = ((Atype * n_types + Btype) * n_types + Btype) * _VS
            var d_AB_x = x_ptr[3 * B]     - xA
            var d_AB_y = x_ptr[3 * B + 1] - yA
            var d_AB_z = x_ptr[3 * B + 2] - zA
            var rsq_AB = d_AB_x * d_AB_x + d_AB_y * d_AB_y + d_AB_z * d_AB_z
            if rsq_AB >= p_ptr[off_AB + _VS_CUTSQ2]:
                continue
            var r_AB     = sqrt(rsq_AB)
            var rainv_AB = Float32(1.0) / (r_AB - p_ptr[off_AB + _VS_R0])
            var gsrainv_AB = p_ptr[off_AB + _VS_GAMMA] * rainv_AB
            var gsrainvsq_AB = gsrainv_AB * rainv_AB / r_AB
            var expg_AB  = exp(gsrainv_AB)
            var off_ABm  = ((Atype * n_types + Btype) * n_types + mtype) * _VS
            var bigb     = p_ptr[off_ABm + _VS_BIGB]
            var big2b    = p_ptr[off_ABm + _VS_BIG2B]
            var bigc     = p_ptr[off_ABm + _VS_BIGC]
            var costh    = p_ptr[off_ABm + _VS_COSTH]
            var rinv12   = Float32(1.0) / (r_AB * r_Am)
            var cs       = (d_AB_x * d_Am_x + d_AB_y * d_Am_y + d_AB_z * d_Am_z) * rinv12
            var delcs    = cs - costh;  var delcssq = delcs * delcs
            var pcsinv   = bigc * delcssq + Float32(1.0); var pcsinvsq = pcsinv * pcsinv
            var pcs      = delcssq / pcsinv
            var facexp   = expg_AB * expg_Am
            var facrad   = bigb * facexp * pcs
            var frad2    = facrad * gsrainvsq_Am
            var facang   = big2b * facexp * delcs / pcsinvsq
            var facang12 = rinv12 * facang
            var csfacang = cs * facang
            var csfac2   = (Float32(1.0) / rsq_Am) * csfacang
            fxm += d_Am_x * (frad2 + csfac2) - d_AB_x * facang12
            fym += d_Am_y * (frad2 + csfac2) - d_AB_y * facang12
            fzm += d_Am_z * (frad2 + csfac2) - d_AB_z * facang12

        # C > pos_m → m is "j" in triplet (A, m, C)
        for nn in range(pos_m + 1, nshort_A):
            var C = Int(snb_ptr[ss_A + nn])
            if C < 0:
                break   # GPU strided list sentinel
            var Ctype = Int(type_id_ptr[C])
            var off_AC = ((Atype * n_types + Ctype) * n_types + Ctype) * _VS
            var d_AC_x = x_ptr[3 * C]     - xA
            var d_AC_y = x_ptr[3 * C + 1] - yA
            var d_AC_z = x_ptr[3 * C + 2] - zA
            var rsq_AC = d_AC_x * d_AC_x + d_AC_y * d_AC_y + d_AC_z * d_AC_z
            if rsq_AC >= p_ptr[off_AC + _VS_CUTSQ2]:
                continue
            var r_AC     = sqrt(rsq_AC)
            var rainv_AC = Float32(1.0) / (r_AC - p_ptr[off_AC + _VS_R0])
            var gsrainv_AC = p_ptr[off_AC + _VS_GAMMA] * rainv_AC
            var gsrainvsq_AC = gsrainv_AC * rainv_AC / r_AC
            var expg_AC  = exp(gsrainv_AC)
            var off_AmC  = ((Atype * n_types + mtype) * n_types + Ctype) * _VS
            var bigb     = p_ptr[off_AmC + _VS_BIGB]
            var big2b    = p_ptr[off_AmC + _VS_BIG2B]
            var bigc     = p_ptr[off_AmC + _VS_BIGC]
            var costh    = p_ptr[off_AmC + _VS_COSTH]
            var rinv12   = Float32(1.0) / (r_Am * r_AC)
            var cs       = (d_Am_x * d_AC_x + d_Am_y * d_AC_y + d_Am_z * d_AC_z) * rinv12
            var delcs    = cs - costh;  var delcssq = delcs * delcs
            var pcsinv   = bigc * delcssq + Float32(1.0); var pcsinvsq = pcsinv * pcsinv
            var pcs      = delcssq / pcsinv
            var facexp   = expg_Am * expg_AC
            var facrad   = bigb * facexp * pcs
            var frad1    = facrad * gsrainvsq_Am
            var facang   = big2b * facexp * delcs / pcsinvsq
            var facang12 = rinv12 * facang
            var csfacang = cs * facang
            var csfac1   = (Float32(1.0) / rsq_Am) * csfacang
            fxm += d_Am_x * (frad1 + csfac1) - d_AC_x * facang12
            fym += d_Am_y * (frad1 + csfac1) - d_AC_y * facang12
            fzm += d_Am_z * (frad1 + csfac1) - d_AC_z * facang12

    f_ptr[3 * m] += fxm; f_ptr[3 * m + 1] += fym; f_ptr[3 * m + 2] += fzm
    pe_ptr[m] += em


# ---------------------------------------------------------------------------
# PairVashishta — single struct, CPU and GPU dispatchers in the same file.
# Params stored once (flat List[Float64]); same kernel body on either backend.
# ---------------------------------------------------------------------------

struct PairVashishta(PairStyle):
    """Vashishta two-body + three-body potential.

    Parameters: flat List[Float64] of n_types³ × _VS entries — same layout the
    GPU device buffer uses, so the kernel body is identical on CPU and GPU.
    Owner-computes everywhere: thread for atom m writes only to f[m].
    """
    var n_types: Int
    var params:  List[Float64]   # flat: _VS * n_types^3 — single source of truth
    var _cutoff: Float64
    var _r0_max: Float64

    fn __init__(out self, n_types: Int):
        self.n_types = n_types
        self.params = List[Float64]()
        for _ in range(n_types * n_types * n_types * _VS):
            self.params.append(0.0)
        self._cutoff = 0.0
        self._r0_max = 0.0

    fn _flat_idx(self, i: Int, j: Int, k: Int) -> Int:
        return ((i * self.n_types + j) * self.n_types + k) * _VS

    fn set_param(mut self, itype: Int, jtype: Int, ktype: Int, p: VashishtaParam):
        """Set parameters for triplet (itype, jtype, ktype). Call for all n³ combinations."""
        var off = self._flat_idx(itype, jtype, ktype)
        self.params[off + _VS_BIGH]    = p.bigh
        self.params[off + _VS_ETA]     = p.eta
        self.params[off + _VS_ZI]      = p.zi
        self.params[off + _VS_ZJ]      = p.zj
        self.params[off + _VS_LAMBDA1] = p.lambda1
        self.params[off + _VS_BIGD]    = p.bigd
        self.params[off + _VS_LAMBDA4] = p.lambda4
        self.params[off + _VS_BIGW]    = p.bigw
        self.params[off + _VS_CUT]     = p.cut
        self.params[off + _VS_BIGB]    = p.bigb
        self.params[off + _VS_GAMMA]   = p.gamma
        self.params[off + _VS_R0]      = p.r0
        self.params[off + _VS_BIGC]    = p.bigc
        self.params[off + _VS_COSTH]   = p.costheta
        self.params[off + _VS_HETA]    = p.heta
        self.params[off + _VS_ZIZJ]    = p.zizj
        self.params[off + _VS_LAM1INV] = p.lam1inv
        self.params[off + _VS_MBIGD]   = p.mbigd
        self.params[off + _VS_LAM4INV] = p.lam4inv
        self.params[off + _VS_BIG2B]   = p.big2b
        self.params[off + _VS_BIG6W]   = p.big6w
        self.params[off + _VS_CUTSQ]   = p.cutsq
        self.params[off + _VS_CUTSQ2]  = p.cutsq2
        self.params[off + _VS_DVRC]    = p.dvrc
        self.params[off + _VS_C0]      = p.c0
        if p.cut > self._cutoff:
            self._cutoff = p.cut
        if p.r0 > self._r0_max:
            self._r0_max = p.r0

    fn cutoff(self) -> Float64:
        return self._cutoff

    fn short_cutoff(self) -> Float64:
        return self._r0_max

    # ---- CPU dispatch ----
    fn compute(mut self, mut atoms: Atoms, mut nlist: NeighborList) -> Float64:
        var nlocal = atoms.nlocal

        # type_id is List[Int] — convert once (nmax entries, small vs neighbor list).
        var tid_buf = List[Int32](capacity=atoms.nmax)
        for i in range(atoms.nmax):
            tid_buf.append(Int32(atoms.type_id[i]))

        # Neighbor lists are already List[Int32] — pass pointers directly, no copy.
        var pe_atom = List[Float64](capacity=nlocal)
        for _ in range(nlocal):
            pe_atom.append(0.0)

        var x_ptr    = atoms.x.unsafe_ptr()
        var f_ptr    = atoms.f.unsafe_ptr()
        var pe_ptr   = pe_atom.unsafe_ptr()
        var p_ptr    = self.params.unsafe_ptr()
        var tid_ptr  = tid_buf.unsafe_ptr()
        var off_ptr  = nlist.offsets.unsafe_ptr()
        var nb_ptr   = nlist.neighbors.unsafe_ptr()
        var soff_ptr = nlist.short_offsets.unsafe_ptr()
        var snb_ptr  = nlist.short_neighbors.unsafe_ptr()
        var n_types  = self.n_types

        # 2-body pass — pe_atom[m] gets the halved 2-body energy
        @parameter
        fn body_2b(i: Int):
            _vashishta_2body_body(i, x_ptr, f_ptr, pe_ptr, tid_ptr,
                                  p_ptr, off_ptr, nb_ptr, n_types)
        parallelize[body_2b](nlocal)

        # 3-body pass — pe_atom[m] += apex energy (m as apex of triplets)
        @parameter
        fn body_3b(m: Int):
            _vashishta_3body_body(m, nlocal, x_ptr, f_ptr, pe_ptr, tid_ptr,
                                  p_ptr, soff_ptr, snb_ptr, n_types)
        parallelize[body_3b](nlocal)

        var total: Float64 = 0.0
        for e in pe_atom:
            total += e
        return total

    # ---- GPU dispatch ----
    fn make_gpu_params(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.float32]:
        var n_floats = self.n_types * self.n_types * self.n_types * _VS
        var dev  = ctx.enqueue_create_buffer[DType.float32](n_floats)
        var host = ctx.enqueue_create_host_buffer[DType.float32](n_floats)
        for i in range(n_floats):
            host[i] = Float32(self.params[i])
        ctx.enqueue_copy(dev, host)
        ctx.synchronize()
        return dev^

    fn compute_gpu(
        self,
        mut atoms: GPUAtoms,
        read nlist: GPUNeighborList,
        read params_dev: DeviceBuffer[DType.float32],
        ctx: DeviceContext,
    ) raises -> Float64:
        var nlocal = atoms.nlocal
        var n_blocks = ceildiv(nlocal, _BLOCK_SIZE)

        # 2-body pass — writes pe_atom[i] = 0.5 * 2-body energy
        ctx.enqueue_function[_vashishta_2body_kernel, _vashishta_2body_kernel](
            atoms.x.unsafe_ptr(),
            atoms.f.unsafe_ptr(),
            atoms.pe_atom.unsafe_ptr(),
            atoms.type_id.unsafe_ptr(),
            params_dev.unsafe_ptr(),
            nlist.offsets.unsafe_ptr(),
            nlist.neighbors.unsafe_ptr(),
            self.n_types,
            nlocal,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

        # 3-body pass — accumulates apex energy into the same pe_atom[m]
        ctx.enqueue_function[_vashishta_3body_kernel, _vashishta_3body_kernel](
            atoms.x.unsafe_ptr(),
            atoms.f.unsafe_ptr(),
            atoms.pe_atom.unsafe_ptr(),
            atoms.type_id.unsafe_ptr(),
            params_dev.unsafe_ptr(),
            nlist.short_offsets.unsafe_ptr(),
            nlist.short_neighbors.unsafe_ptr(),
            self.n_types,
            nlocal,
            grid_dim  = n_blocks,
            block_dim = _BLOCK_SIZE,
        )

        return atoms.read_pe_to_cpu(ctx)
