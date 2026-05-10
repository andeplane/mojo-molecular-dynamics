from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from mojo_md.neighbor import NeighborList


struct Atoms(Movable):
    """
    Per-atom data in Structure-of-Arrays layout.

    Indices 0..nlocal-1 are real (owned) atoms.
    Indices nlocal..nlocal+nghost-1 are ghost copies (PBC images or, later,
    halo atoms from neighboring MPI ranks). Positions and forces span the full
    nlocal+nghost range; velocities are nlocal-only (ghosts do not integrate).
    """
    var nlocal: Int
    var nghost: Int
    var nmax: Int          # allocated capacity (>= nlocal + nghost)

    var x: List[Float64]   # positions [3*nmax]
    var v: List[Float64]   # velocities [3*nlocal]
    var f: List[Float64]   # forces [3*nmax]  -- ghosts accumulate, then reverse-summed
    var mass: List[Float64] # [nmax]
    var type_id: List[Int]  # atom type 0-indexed [nmax]
    var tag: List[Int]      # global atom ID [nmax]
    var ghost_source: List[Int]  # ghost g → source local atom index [nghost]

    var box: List[Float64] # [lx, ly, lz] — orthogonal periodic box

    fn __init__(out self, nlocal: Int, lx: Float64, ly: Float64, lz: Float64):
        self.nlocal = nlocal
        self.nghost = 0
        self.nmax = nlocal + 128  # headroom for ghosts

        self.x = List[Float64](capacity=3 * self.nmax)
        self.v = List[Float64](capacity=3 * nlocal)
        self.f = List[Float64](capacity=3 * self.nmax)
        self.mass = List[Float64](capacity=self.nmax)
        self.type_id = List[Int](capacity=self.nmax)
        self.tag = List[Int](capacity=self.nmax)
        self.ghost_source = List[Int]()
        self.box = List[Float64](capacity=3)

        for _ in range(3 * self.nmax):
            self.x.append(0.0)
            self.f.append(0.0)
        for _ in range(3 * nlocal):
            self.v.append(0.0)
        for _ in range(self.nmax):
            self.mass.append(1.0)
            self.type_id.append(0)
            self.tag.append(0)

        self.box.append(lx)
        self.box.append(ly)
        self.box.append(lz)

    fn n(self) -> Int:
        """Total atoms including ghosts."""
        return self.nlocal + self.nghost

    fn zero_forces(mut self):
        """Zero forces for all local + ghost atoms."""
        for idx in range(3 * self.n()):
            self.f[idx] = 0.0

    fn grow(mut self, new_nmax: Int):
        """Grow buffers to accommodate more ghost atoms."""
        if new_nmax <= self.nmax:
            return
        var old_nmax = self.nmax
        self.nmax = new_nmax
        for _ in range(3 * (new_nmax - old_nmax)):
            self.x.append(0.0)
            self.f.append(0.0)
        for _ in range(new_nmax - old_nmax):
            self.mass.append(1.0)
            self.type_id.append(0)
            self.tag.append(0)

    fn kinetic_energy(self) -> Float64:
        """Sum of 0.5 * m * v^2 over local atoms."""
        var ke: Float64 = 0.0
        for i in range(self.nlocal):
            var vx = self.v[3 * i]
            var vy = self.v[3 * i + 1]
            var vz = self.v[3 * i + 2]
            ke += 0.5 * self.mass[i] * (vx * vx + vy * vy + vz * vz)
        return ke

    fn temperature(self, dof: Int = 0) -> Float64:
        """Instantaneous temperature in units where kB = 1."""
        var effective_dof = dof if dof > 0 else 3 * self.nlocal - 3
        if effective_dof <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy() / Float64(effective_dof)


fn minimum_image(dx: Float64, box_len: Float64) -> Float64:
    """Apply minimum image convention for one component."""
    var result = dx
    var half = 0.5 * box_len
    if result > half:
        result -= box_len
    elif result < -half:
        result += box_len
    return result


fn _floor(x: Float64) -> Float64:
    """Floor toward -infinity (unlike Int() which truncates toward zero)."""
    var n = Int(x)
    if Float64(n) > x:
        return Float64(n - 1)
    return Float64(n)


fn wrap_into_box(mut atoms: Atoms):
    """Wrap real atoms back into the primary box [0, L)."""
    for i in range(atoms.nlocal):
        var ix = 3 * i
        for d in range(3):
            var l = atoms.box[d]
            var pos = atoms.x[ix + d]
            pos -= l * _floor(pos / l)
            if pos < 0.0:
                pos += l
            if pos >= l:
                pos -= l
            atoms.x[ix + d] = pos


# ---------------------------------------------------------------------------
# GPU mirror of Atoms — same SoA layout, lives in DeviceBuffer instead of List.
# Created from `Atoms.upload(ctx)` and refreshed at every neighbor-list rebuild.
# Per-step kernels never touch the CPU side; only positions are round-tripped
# at rebuild_interval steps so the CPU can rebuild ghosts and the neighbor list.
# ---------------------------------------------------------------------------

struct GPUAtoms(Movable):
    """GPU-resident mirror of Atoms (DeviceBuffer SoA, Float32 for Metal/MPS compatibility).
    CPU uses Float64; casts happen at upload/download boundaries."""
    var nlocal: Int
    var nghost: Int
    var nmax: Int
    var x:        DeviceBuffer[DType.float32]   # 3*nmax
    var v:        DeviceBuffer[DType.float32]   # 3*nlocal
    var f:        DeviceBuffer[DType.float32]   # 3*nmax
    var pe_atom:  DeviceBuffer[DType.float32]   # nlocal
    var mass:     DeviceBuffer[DType.float32]   # nmax
    var type_id:  DeviceBuffer[DType.int32]     # nmax
    var tag:      DeviceBuffer[DType.int32]     # nmax
    var box:      DeviceBuffer[DType.float32]   # 3

    fn __init__(
        out self,
        nlocal: Int, nghost: Int, nmax: Int,
        var x: DeviceBuffer[DType.float32],
        var v: DeviceBuffer[DType.float32],
        var f: DeviceBuffer[DType.float32],
        var pe_atom: DeviceBuffer[DType.float32],
        var mass: DeviceBuffer[DType.float32],
        var type_id: DeviceBuffer[DType.int32],
        var tag: DeviceBuffer[DType.int32],
        var box: DeviceBuffer[DType.float32],
    ):
        self.nlocal = nlocal;  self.nghost = nghost;  self.nmax = nmax
        self.x = x^;  self.v = v^;  self.f = f^;  self.pe_atom = pe_atom^
        self.mass = mass^;  self.type_id = type_id^;  self.tag = tag^;  self.box = box^

    fn n(self) -> Int:
        return self.nlocal + self.nghost

    @staticmethod
    fn from_cpu(read atoms: Atoms, ctx: DeviceContext) raises -> GPUAtoms:
        """Allocate Float32 device buffers and upload all per-atom arrays (Float64→Float32)."""
        var nlocal = atoms.nlocal
        var nghost = atoms.nghost
        var nmax   = atoms.nmax

        var x_dev    = ctx.enqueue_create_buffer[DType.float32](3 * nmax)
        var v_dev    = ctx.enqueue_create_buffer[DType.float32](3 * nlocal)
        var f_dev    = ctx.enqueue_create_buffer[DType.float32](3 * nmax)
        var pe_dev   = ctx.enqueue_create_buffer[DType.float32](nlocal)
        var mass_dev = ctx.enqueue_create_buffer[DType.float32](nmax)
        var tid_dev  = ctx.enqueue_create_buffer[DType.int32](nmax)
        var tag_dev  = ctx.enqueue_create_buffer[DType.int32](nmax)
        var box_dev  = ctx.enqueue_create_buffer[DType.float32](3)

        var h_x    = ctx.enqueue_create_host_buffer[DType.float32](3 * nmax)
        var h_v    = ctx.enqueue_create_host_buffer[DType.float32](3 * nlocal)
        var h_f    = ctx.enqueue_create_host_buffer[DType.float32](3 * nmax)
        var h_mass = ctx.enqueue_create_host_buffer[DType.float32](nmax)
        var h_tid  = ctx.enqueue_create_host_buffer[DType.int32](nmax)
        var h_tag  = ctx.enqueue_create_host_buffer[DType.int32](nmax)
        var h_box  = ctx.enqueue_create_host_buffer[DType.float32](3)

        for i in range(3 * nmax):
            h_x[i] = Float32(atoms.x[i])
            h_f[i] = Float32(atoms.f[i])
        for i in range(3 * nlocal):
            h_v[i] = Float32(atoms.v[i])
        for i in range(nmax):
            h_mass[i] = Float32(atoms.mass[i])
            h_tid[i]  = Int32(atoms.type_id[i])
            h_tag[i]  = Int32(atoms.tag[i])
        h_box[0] = Float32(atoms.box[0])
        h_box[1] = Float32(atoms.box[1])
        h_box[2] = Float32(atoms.box[2])

        ctx.enqueue_copy(x_dev,    h_x)
        ctx.enqueue_copy(v_dev,    h_v)
        ctx.enqueue_copy(f_dev,    h_f)
        ctx.enqueue_copy(mass_dev, h_mass)
        ctx.enqueue_copy(tid_dev,  h_tid)
        ctx.enqueue_copy(tag_dev,  h_tag)
        ctx.enqueue_copy(box_dev,  h_box)
        ctx.synchronize()

        return GPUAtoms(
            nlocal, nghost, nmax,
            x_dev^, v_dev^, f_dev^, pe_dev^, mass_dev^, tid_dev^, tag_dev^, box_dev^,
        )

    fn refresh_from_cpu(mut self, read atoms: Atoms, ctx: DeviceContext) raises:
        """After a CPU-side ghost+nlist rebuild, re-upload x, type_id, tag (Float64→Float32).
        Reallocates device buffers if nmax grew so copy never exceeds buffer length."""
        self.nlocal = atoms.nlocal
        self.nghost = atoms.nghost

        # Reallocate device buffers if nmax grew (ghost atoms pushed us past old capacity).
        if atoms.nmax > self.nmax:
            self.nmax    = atoms.nmax
            self.x       = ctx.enqueue_create_buffer[DType.float32](3 * self.nmax)
            self.type_id = ctx.enqueue_create_buffer[DType.int32](self.nmax)
            self.tag     = ctx.enqueue_create_buffer[DType.int32](self.nmax)

        var h_x   = ctx.enqueue_create_host_buffer[DType.float32](3 * self.nmax)
        var h_tid = ctx.enqueue_create_host_buffer[DType.int32](self.nmax)
        var h_tag = ctx.enqueue_create_host_buffer[DType.int32](self.nmax)

        for i in range(3 * self.nmax):
            h_x[i] = Float32(atoms.x[i])
        for i in range(self.nmax):
            h_tid[i] = Int32(atoms.type_id[i])
            h_tag[i] = Int32(atoms.tag[i])

        ctx.enqueue_copy(self.x,       h_x)
        ctx.enqueue_copy(self.type_id, h_tid)
        ctx.enqueue_copy(self.tag,     h_tag)
        ctx.synchronize()

    fn read_positions_to_cpu(read self, mut atoms: Atoms, ctx: DeviceContext) raises:
        """GPU→CPU: copy real-atom positions (Float32→Float64) for CPU neighbor-list rebuild."""
        var h_x = ctx.enqueue_create_host_buffer[DType.float32](3 * self.nlocal)
        ctx.enqueue_copy(h_x, self.x)
        ctx.synchronize()
        for i in range(3 * self.nlocal):
            atoms.x[i] = Float64(h_x[i])

    fn read_pe_to_cpu(read self, ctx: DeviceContext) raises -> Float64:
        """GPU→CPU: read per-atom PE (Float32), sum on host as Float64."""
        var h_pe = ctx.enqueue_create_host_buffer[DType.float32](self.nlocal)
        ctx.enqueue_copy(h_pe, self.pe_atom)
        ctx.synchronize()
        var total: Float64 = 0.0
        for i in range(self.nlocal):
            total += Float64(h_pe[i])
        return total

    fn read_ke_to_cpu(read self, ctx: DeviceContext) raises -> Float64:
        """GPU→CPU: copy v and mass (Float32), compute KE on host as Float64."""
        var nlocal = self.nlocal
        var h_v    = ctx.enqueue_create_host_buffer[DType.float32](3 * nlocal)
        var h_mass = ctx.enqueue_create_host_buffer[DType.float32](nlocal)
        ctx.enqueue_copy(h_v,    self.v)
        ctx.enqueue_copy(h_mass, self.mass)
        ctx.synchronize()
        var ke: Float64 = 0.0
        for i in range(nlocal):
            var vx = Float64(h_v[3 * i])
            var vy = Float64(h_v[3 * i + 1])
            var vz = Float64(h_v[3 * i + 2])
            ke += 0.5 * Float64(h_mass[i]) * (vx*vx + vy*vy + vz*vz)
        return ke


struct GPUNeighborList(Movable):
    """GPU-resident CSR neighbor list. Built on CPU each rebuild_interval, uploaded here."""
    var nlocal: Int
    var offsets:         DeviceBuffer[DType.int32]
    var neighbors:       DeviceBuffer[DType.int32]
    var short_offsets:   DeviceBuffer[DType.int32]
    var short_neighbors: DeviceBuffer[DType.int32]

    fn __init__(
        out self,
        nlocal: Int,
        var offsets: DeviceBuffer[DType.int32],
        var neighbors: DeviceBuffer[DType.int32],
        var short_offsets: DeviceBuffer[DType.int32],
        var short_neighbors: DeviceBuffer[DType.int32],
    ):
        self.nlocal = nlocal
        self.offsets = offsets^
        self.neighbors = neighbors^
        self.short_offsets = short_offsets^
        self.short_neighbors = short_neighbors^

    @staticmethod
    fn from_cpu(read nlist: NeighborList, ctx: DeviceContext) raises -> GPUNeighborList:
        var nlocal = nlist.nlocal
        var n_off  = nlocal + 1
        var n_nb   = max(len(nlist.neighbors), 1)
        var n_snb  = max(len(nlist.short_neighbors), 1)

        var off_dev  = ctx.enqueue_create_buffer[DType.int32](n_off)
        var nb_dev   = ctx.enqueue_create_buffer[DType.int32](n_nb)
        var soff_dev = ctx.enqueue_create_buffer[DType.int32](n_off)
        var snb_dev  = ctx.enqueue_create_buffer[DType.int32](n_snb)

        var h_off  = ctx.enqueue_create_host_buffer[DType.int32](n_off)
        var h_nb   = ctx.enqueue_create_host_buffer[DType.int32](n_nb)
        var h_soff = ctx.enqueue_create_host_buffer[DType.int32](n_off)
        var h_snb  = ctx.enqueue_create_host_buffer[DType.int32](n_snb)

        for i in range(n_off):
            h_off[i]  = Int32(nlist.offsets[i])
            h_soff[i] = Int32(nlist.short_offsets[i])
        for i in range(len(nlist.neighbors)):
            h_nb[i] = Int32(nlist.neighbors[i])
        for i in range(len(nlist.short_neighbors)):
            h_snb[i] = Int32(nlist.short_neighbors[i])

        ctx.enqueue_copy(off_dev,  h_off)
        ctx.enqueue_copy(nb_dev,   h_nb)
        ctx.enqueue_copy(soff_dev, h_soff)
        ctx.enqueue_copy(snb_dev,  h_snb)
        ctx.synchronize()

        return GPUNeighborList(nlocal, off_dev^, nb_dev^, soff_dev^, snb_dev^)

    fn refresh_from_cpu(mut self, read nlist: NeighborList, ctx: DeviceContext) raises:
        """Re-upload after each CPU-side rebuild. Always reallocates device buffers
        since neighbor count changes each step (new ghost atoms, different occupancy)."""
        var nlocal = nlist.nlocal
        var n_off  = nlocal + 1
        var n_nb   = max(len(nlist.neighbors), 1)
        var n_snb  = max(len(nlist.short_neighbors), 1)
        self.nlocal = nlocal

        self.offsets         = ctx.enqueue_create_buffer[DType.int32](n_off)
        self.neighbors       = ctx.enqueue_create_buffer[DType.int32](n_nb)
        self.short_offsets   = ctx.enqueue_create_buffer[DType.int32](n_off)
        self.short_neighbors = ctx.enqueue_create_buffer[DType.int32](n_snb)

        var h_off  = ctx.enqueue_create_host_buffer[DType.int32](n_off)
        var h_nb   = ctx.enqueue_create_host_buffer[DType.int32](n_nb)
        var h_soff = ctx.enqueue_create_host_buffer[DType.int32](n_off)
        var h_snb  = ctx.enqueue_create_host_buffer[DType.int32](n_snb)
        for i in range(n_off):
            h_off[i]  = Int32(nlist.offsets[i])
            h_soff[i] = Int32(nlist.short_offsets[i])
        for i in range(len(nlist.neighbors)):
            h_nb[i] = Int32(nlist.neighbors[i])
        for i in range(len(nlist.short_neighbors)):
            h_snb[i] = Int32(nlist.short_neighbors[i])

        ctx.enqueue_copy(self.offsets,         h_off)
        ctx.enqueue_copy(self.neighbors,       h_nb)
        ctx.enqueue_copy(self.short_offsets,   h_soff)
        ctx.enqueue_copy(self.short_neighbors, h_snb)
        ctx.synchronize()
