from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.neighbor import NeighborList
from mojo_md.pair_style import PairStyle

# Flat layout (per (itype, jtype) pair): 6 floats. Same on CPU and GPU.
comptime _LJ_STRIDE: Int = 6
comptime _LJ_LJ1:    Int = 0
comptime _LJ_LJ2:    Int = 1
comptime _LJ_LJ3:    Int = 2
comptime _LJ_LJ4:    Int = 3
comptime _LJ_RC_SQ:  Int = 4
comptime _LJ_ESHIFT: Int = 5

comptime _BLOCK_SIZE: Int = 256


struct LJParams(ImplicitlyCopyable, Movable):
    """Precomputed Lennard-Jones coefficients for one (i,j) type pair."""
    var lj1: Float64
    var lj2: Float64
    var lj3: Float64
    var lj4: Float64
    var rc_sq: Float64
    var energy_shift: Float64

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


# ---------------------------------------------------------------------------
# Shared LJ force kernel body.
# Owner-computes: thread for atom i loops over neighbours and writes only
# to f[3*i .. 3*i+2] and pe_atom[i]. No atomics, parallel-safe on CPU and GPU.
# ---------------------------------------------------------------------------

@always_inline
fn _lj_force_body(
    i:           Int,
    x_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    f_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    pe_ptr:      UnsafePointer[Float64, MutAnyOrigin],   # nlocal — per-atom PE
    type_id_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    p_ptr:       UnsafePointer[Float64, MutAnyOrigin],
    off_ptr:     UnsafePointer[Int32,   MutAnyOrigin],
    nb_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    n_types:     Int,
):
    var xi = x_ptr[3 * i]
    var yi = x_ptr[3 * i + 1]
    var zi = x_ptr[3 * i + 2]
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
        var off = (n_types * itype + jtype) * _LJ_STRIDE

        var lj1    = p_ptr[off + _LJ_LJ1]
        var lj2    = p_ptr[off + _LJ_LJ2]
        var lj3    = p_ptr[off + _LJ_LJ3]
        var lj4    = p_ptr[off + _LJ_LJ4]
        var rc_sq  = p_ptr[off + _LJ_RC_SQ]
        var eshift = p_ptr[off + _LJ_ESHIFT]

        var dx = xi - x_ptr[3 * j]
        var dy = yi - x_ptr[3 * j + 1]
        var dz = zi - x_ptr[3 * j + 2]
        var rsq = dx * dx + dy * dy + dz * dz

        if rsq < rc_sq:
            var r2inv = 1.0 / rsq
            var r6inv = r2inv * r2inv * r2inv
            var fpair = (lj1 * r6inv - lj2) * r6inv * r2inv
            fxi += dx * fpair
            fyi += dy * fpair
            fzi += dz * fpair
            ei  += r6inv * (lj3 * r6inv - lj4) - eshift

    f_ptr[3 * i]     += fxi
    f_ptr[3 * i + 1] += fyi
    f_ptr[3 * i + 2] += fzi
    pe_ptr[i] = ei  # owner-computes — no atomic, summed on host afterwards


# GPU kernel — Float32 (Metal/MPS). Same algorithm as _lj_force_body above (Float64 CPU).
fn _lj_force_kernel(
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
        var jtype = Int(type_id_ptr[j])
        var off = (n_types * itype + jtype) * _LJ_STRIDE
        var lj1    = p_ptr[off + _LJ_LJ1];  var lj2    = p_ptr[off + _LJ_LJ2]
        var lj3    = p_ptr[off + _LJ_LJ3];  var lj4    = p_ptr[off + _LJ_LJ4]
        var rc_sq  = p_ptr[off + _LJ_RC_SQ]; var eshift = p_ptr[off + _LJ_ESHIFT]
        var dx = xi - x_ptr[3 * j]; var dy = yi - x_ptr[3 * j + 1]; var dz = zi - x_ptr[3 * j + 2]
        var rsq = dx * dx + dy * dy + dz * dz
        if rsq < rc_sq:
            var r2inv = Float32(1.0) / rsq
            var r6inv = r2inv * r2inv * r2inv
            var fpair = (lj1 * r6inv - lj2) * r6inv * r2inv
            fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair
            ei  += r6inv * (lj3 * r6inv - lj4) - eshift
    f_ptr[3 * i]     += fxi; f_ptr[3 * i + 1] += fyi; f_ptr[3 * i + 2] += fzi
    pe_ptr[i] = ei


# ---------------------------------------------------------------------------
# PairLJ — single struct, CPU and GPU dispatchers in the same file.
# Params stored once (flat List[Float64]). The GPU mirror is owned by the
# simulation (created via make_gpu_params) so PairLJ doesn't need to hold
# a DeviceContext in its constructor.
# ---------------------------------------------------------------------------

struct PairLJ(PairStyle):
    """Multi-type Lennard-Jones with full neighbor list."""
    var n_types: Int
    var params:  List[Float64]   # flat: _LJ_STRIDE * n_types^2 — single source of truth
    var _cutoff: Float64

    fn __init__(out self, n_types: Int):
        self.n_types = n_types
        self.params = List[Float64]()
        for _ in range(n_types * n_types * _LJ_STRIDE):
            self.params.append(0.0)
        self._cutoff = 0.0

    fn set_pair(
        mut self,
        itype: Int, jtype: Int,
        epsilon: Float64, sigma: Float64, rc: Float64,
    ):
        """Set LJ parameters for (itype, jtype). Symmetric — also sets (jtype, itype)."""
        var p = LJParams(epsilon, sigma, rc)
        self._write(itype, jtype, p)
        self._write(jtype, itype, p)
        if rc > self._cutoff:
            self._cutoff = rc

    fn _write(mut self, itype: Int, jtype: Int, p: LJParams):
        var off = (self.n_types * itype + jtype) * _LJ_STRIDE
        self.params[off + _LJ_LJ1]    = p.lj1
        self.params[off + _LJ_LJ2]    = p.lj2
        self.params[off + _LJ_LJ3]    = p.lj3
        self.params[off + _LJ_LJ4]    = p.lj4
        self.params[off + _LJ_RC_SQ]  = p.rc_sq
        self.params[off + _LJ_ESHIFT] = p.energy_shift

    fn cutoff(self) -> Float64:
        return self._cutoff

    fn short_cutoff(self) -> Float64:
        return 0.0

    # ---- CPU dispatch ----
    fn compute(mut self, mut atoms: Atoms, mut nlist: NeighborList) -> Float64:
        var nlocal = atoms.nlocal
        var pe_atom = List[Float64](capacity=nlocal)
        for _ in range(nlocal):
            pe_atom.append(0.0)

        # type_id is List[Int] — convert once (nmax entries, small vs neighbor list).
        var tid_buf = List[Int32](capacity=atoms.nmax)
        for i in range(atoms.nmax):
            tid_buf.append(Int32(atoms.type_id[i]))

        # Neighbor list is already List[Int32] — pass pointers directly, no copy.
        var x_ptr   = atoms.x.unsafe_ptr()
        var f_ptr   = atoms.f.unsafe_ptr()
        var pe_ptr  = pe_atom.unsafe_ptr()
        var p_ptr   = self.params.unsafe_ptr()
        var tid_ptr = tid_buf.unsafe_ptr()
        var off_ptr = nlist.offsets.unsafe_ptr()
        var nb_ptr  = nlist.neighbors.unsafe_ptr()
        var n_types = self.n_types

        @parameter
        fn body(i: Int):
            _lj_force_body(i, x_ptr, f_ptr, pe_ptr, tid_ptr, p_ptr,
                           off_ptr, nb_ptr, n_types)

        parallelize[body](nlocal)

        var total: Float64 = 0.0
        for e in pe_atom:
            total += e
        return total * 0.5  # full list double-counts

    # ---- GPU dispatch ----
    fn make_gpu_params(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.float32]:
        """Upload the flat params buffer to the device as Float32. Caller owns the result."""
        var n_floats = self.n_types * self.n_types * _LJ_STRIDE
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
        ctx.enqueue_function[_lj_force_kernel, _lj_force_kernel](
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
        return atoms.read_pe_to_cpu(ctx) * 0.5  # full list double-counts
