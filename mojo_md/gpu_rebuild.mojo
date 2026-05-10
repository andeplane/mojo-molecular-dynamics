"""GPU-resident ghost rebuild and cell-list neighbor build.

This module owns the kernels that replace the per-rebuild CPU round-trip. After
init, no atom data crosses CPU↔GPU during a run (scalars like nghost still cross
once per rebuild — that's a 4-byte transfer).

Design:
- Ghost build: 1 thread per local atom, scan 26 PBC shifts, atomically allocate
  ghost slots via a single Int32 device counter. Source-atom index recorded for
  later reverse_comm fold.
- Cell list: fixed-stride 2D `cell_atoms[nc * MAX_ATOMS_PER_CELL]`. cell_count is
  the atomic counter. No prefix scan needed.
- Neighbor list: fixed-stride `neighbors[nlocal * MAX_NB]`. Each thread writes
  valid neighbors at low indices and a single `-1` sentinel after. Force kernels
  break on the first `-1`.

If any pre-allocated bound overflows at runtime, the offending kernel writes a
flag we read back; we abort with an actionable error.
"""

from std.atomic import Atomic
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt
from mojo_md.atom import GPUAtoms


comptime _BLOCK_SIZE: Int = 256
comptime MAX_NEIGHBORS_PER_ATOM:       Int = 128
comptime MAX_SHORT_NEIGHBORS_PER_ATOM: Int = 64
comptime MAX_ATOMS_PER_CELL:           Int = 256


# ---------------------------------------------------------------------------
# Wrap positions into primary box [0, L). Per-local-atom.
# ---------------------------------------------------------------------------

fn _wrap_kernel(
    x_ptr:  UnsafePointer[Float32, MutAnyOrigin],
    lx:     Float32,
    ly:     Float32,
    lz:     Float32,
    nlocal: Int,
):
    var i = Int(global_idx.x)
    if i >= nlocal:
        return
    var px = x_ptr[3 * i]
    var py = x_ptr[3 * i + 1]
    var pz = x_ptr[3 * i + 2]
    # x - L * floor(x / L), then clamp into [0, L)
    var nx = Float32(Int(px / lx))
    if Float32(nx) > px / lx: nx -= Float32(1.0)
    px -= lx * nx
    if px < Float32(0.0): px += lx
    if px >= lx: px -= lx
    var ny = Float32(Int(py / ly))
    if Float32(ny) > py / ly: ny -= Float32(1.0)
    py -= ly * ny
    if py < Float32(0.0): py += ly
    if py >= ly: py -= ly
    var nz = Float32(Int(pz / lz))
    if Float32(nz) > pz / lz: nz -= Float32(1.0)
    pz -= lz * nz
    if pz < Float32(0.0): pz += lz
    if pz >= lz: pz -= lz
    x_ptr[3 * i]     = px
    x_ptr[3 * i + 1] = py
    x_ptr[3 * i + 2] = pz


fn wrap_into_box_gpu(mut atoms: GPUAtoms, lx: Float32, ly: Float32, lz: Float32, ctx: DeviceContext) raises:
    var nlocal = atoms.nlocal
    var n_blocks = ceildiv(nlocal, _BLOCK_SIZE)
    ctx.enqueue_function[_wrap_kernel, _wrap_kernel](
        atoms.x.unsafe_ptr(), lx, ly, lz, nlocal,
        grid_dim  = n_blocks,
        block_dim = _BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Ghost atom rebuild — 1 thread per local atom, atomic ghost slot allocation.
# ---------------------------------------------------------------------------

fn _ghost_build_kernel(
    x_ptr:        UnsafePointer[Float32, MutAnyOrigin],
    type_id_ptr:  UnsafePointer[Int32,   MutAnyOrigin],
    tag_ptr:      UnsafePointer[Int32,   MutAnyOrigin],
    mass_ptr:     UnsafePointer[Float32, MutAnyOrigin],
    source_idx_ptr: UnsafePointer[Int32, MutAnyOrigin],
    nghost_ptr:   UnsafePointer[Int32,   MutAnyOrigin],   # 1-element atomic counter
    overflow_ptr: UnsafePointer[Int32,   MutAnyOrigin],   # 1-element flag
    nlocal:       Int,
    nmax:         Int,
    lx:           Float32,
    ly:           Float32,
    lz:           Float32,
    rc:           Float32,
):
    var i = Int(global_idx.x)
    if i >= nlocal:
        return
    var xi = x_ptr[3 * i]
    var yi = x_ptr[3 * i + 1]
    var zi = x_ptr[3 * i + 2]
    var itid  = type_id_ptr[i]
    var itag  = tag_ptr[i]
    var imass = mass_ptr[i]
    var ghost_cap = nmax - nlocal

    for sx in range(-1, 2):
        for sy in range(-1, 2):
            for sz in range(-1, 2):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                var gx = xi + Float32(sx) * lx
                var gy = yi + Float32(sy) * ly
                var gz = zi + Float32(sz) * lz
                if gx >= -rc and gx < lx + rc and \
                   gy >= -rc and gy < ly + rc and \
                   gz >= -rc and gz < lz + rc:
                    var g = Int(Atomic.fetch_add(nghost_ptr, Int32(1)))
                    if g >= ghost_cap:
                        # Overflow — signal host and bail out
                        _ = Atomic.fetch_add(overflow_ptr, Int32(1))
                        return
                    var gi = nlocal + g
                    x_ptr[3 * gi]     = gx
                    x_ptr[3 * gi + 1] = gy
                    x_ptr[3 * gi + 2] = gz
                    type_id_ptr[gi]   = itid
                    tag_ptr[gi]       = itag
                    mass_ptr[gi]      = imass
                    source_idx_ptr[g] = Int32(i)


fn _zero_int_kernel(p: UnsafePointer[Int32, MutAnyOrigin], n: Int):
    var i = Int(global_idx.x)
    if i < n:
        p[i] = Int32(0)


fn rebuild_ghosts_gpu(
    mut atoms:    GPUAtoms,
    nghost_dev:   DeviceBuffer[DType.int32],     # length 1
    overflow_dev: DeviceBuffer[DType.int32],     # length 1
    lx: Float32, ly: Float32, lz: Float32, rc: Float32,
    ctx: DeviceContext,
) raises:
    # zero the two scalars
    ctx.enqueue_function[_zero_int_kernel, _zero_int_kernel](
        nghost_dev.unsafe_ptr(), 1, grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[_zero_int_kernel, _zero_int_kernel](
        overflow_dev.unsafe_ptr(), 1, grid_dim=1, block_dim=1,
    )

    var n_blocks = ceildiv(atoms.nlocal, _BLOCK_SIZE)
    ctx.enqueue_function[_ghost_build_kernel, _ghost_build_kernel](
        atoms.x.unsafe_ptr(),
        atoms.type_id.unsafe_ptr(),
        atoms.tag.unsafe_ptr(),
        atoms.mass.unsafe_ptr(),
        atoms.source_idx.unsafe_ptr(),
        nghost_dev.unsafe_ptr(),
        overflow_dev.unsafe_ptr(),
        atoms.nlocal,
        atoms.nmax,
        lx, ly, lz, rc,
        grid_dim  = n_blocks,
        block_dim = _BLOCK_SIZE,
    )

    # Read back nghost (1-int transfer — allowed by spec) and overflow flag.
    var h_ng = ctx.enqueue_create_host_buffer[DType.int32](1)
    var h_of = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(h_ng, nghost_dev)
    ctx.enqueue_copy(h_of, overflow_dev)
    ctx.synchronize()
    if Int(h_of[0]) != 0:
        raise Error("GPU ghost build overflowed: nmax too small for this run. Increase ghost_headroom.")
    atoms.nghost = Int(h_ng[0])


# ---------------------------------------------------------------------------
# Cell-list scatter — 1 thread per atom (local + ghost).
# Atomically increments cell_count[cid] and writes atom index into the slot.
# ---------------------------------------------------------------------------

fn _cell_scatter_kernel(
    x_ptr:        UnsafePointer[Float32, MutAnyOrigin],
    cell_count:   UnsafePointer[Int32,   MutAnyOrigin],
    cell_atoms:   UnsafePointer[Int32,   MutAnyOrigin],
    overflow_ptr: UnsafePointer[Int32,   MutAnyOrigin],
    n_total:      Int,
    nc_x:         Int,
    nc_y:         Int,
    nc_z:         Int,
    cell_lx:      Float32,
    cell_ly:      Float32,
    cell_lz:      Float32,
    max_per_cell: Int,
):
    var i = Int(global_idx.x)
    if i >= n_total:
        return
    var px = x_ptr[3 * i]
    var py = x_ptr[3 * i + 1]
    var pz = x_ptr[3 * i + 2]
    var cx = Int(px / cell_lx)
    var cy = Int(py / cell_ly)
    var cz = Int(pz / cell_lz)
    if cx < 0: cx = 0
    if cy < 0: cy = 0
    if cz < 0: cz = 0
    if cx >= nc_x: cx = nc_x - 1
    if cy >= nc_y: cy = nc_y - 1
    if cz >= nc_z: cz = nc_z - 1
    var cid = (cx * nc_y + cy) * nc_z + cz
    var pos = Int(Atomic.fetch_add(cell_count + cid, Int32(1)))
    if pos < max_per_cell:
        cell_atoms[cid * max_per_cell + pos] = Int32(i)
    else:
        _ = Atomic.fetch_add(overflow_ptr, Int32(1))


# ---------------------------------------------------------------------------
# Neighbor find — 1 thread per local atom. Walks 27 neighboring cells, writes
# valid neighbors at low indices, ends with one `-1` sentinel.
# ---------------------------------------------------------------------------

fn _nlist_build_kernel(
    x_ptr:           UnsafePointer[Float32, MutAnyOrigin],
    cell_count:      UnsafePointer[Int32,   MutAnyOrigin],
    cell_atoms:      UnsafePointer[Int32,   MutAnyOrigin],
    neighbors:       UnsafePointer[Int32,   MutAnyOrigin],
    short_neighbors: UnsafePointer[Int32,   MutAnyOrigin],
    overflow_ptr:    UnsafePointer[Int32,   MutAnyOrigin],
    nlocal:          Int,
    nc_x:            Int,
    nc_y:            Int,
    nc_z:            Int,
    cell_lx:         Float32,
    cell_ly:         Float32,
    cell_lz:         Float32,
    cutsq:           Float32,
    short_cutsq:     Float32,
    max_per_cell:    Int,
    max_nb:          Int,
    max_snb:         Int,
):
    var i = Int(global_idx.x)
    if i >= nlocal:
        return
    var xi = x_ptr[3 * i]
    var yi = x_ptr[3 * i + 1]
    var zi = x_ptr[3 * i + 2]
    var ci_x = Int(xi / cell_lx)
    var ci_y = Int(yi / cell_ly)
    var ci_z = Int(zi / cell_lz)
    if ci_x < 0: ci_x = 0
    if ci_y < 0: ci_y = 0
    if ci_z < 0: ci_z = 0
    if ci_x >= nc_x: ci_x = nc_x - 1
    if ci_y >= nc_y: ci_y = nc_y - 1
    if ci_z >= nc_z: ci_z = nc_z - 1

    var nb_base   = i * max_nb
    var snb_base  = i * max_snb
    var nb_count  = 0
    var snb_count = 0
    var has_short = short_cutsq > Float32(0.0)

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                var nx = ci_x + dx
                var ny = ci_y + dy
                var nz = ci_z + dz
                # periodic wrap of cell index
                if nx < 0: nx += nc_x
                if nx >= nc_x: nx -= nc_x
                if ny < 0: ny += nc_y
                if ny >= nc_y: ny -= nc_y
                if nz < 0: nz += nc_z
                if nz >= nc_z: nz -= nc_z
                var ncid = (nx * nc_y + ny) * nc_z + nz
                var n_in_cell = Int(cell_count[ncid])
                if n_in_cell > max_per_cell:
                    n_in_cell = max_per_cell
                for k in range(n_in_cell):
                    var j = Int(cell_atoms[ncid * max_per_cell + k])
                    if j == i:
                        continue
                    var ddx = xi - x_ptr[3 * j]
                    var ddy = yi - x_ptr[3 * j + 1]
                    var ddz = zi - x_ptr[3 * j + 2]
                    var rsq = ddx * ddx + ddy * ddy + ddz * ddz
                    if rsq < cutsq:
                        if nb_count < max_nb:
                            neighbors[nb_base + nb_count] = Int32(j)
                            nb_count += 1
                        else:
                            _ = Atomic.fetch_add(overflow_ptr, Int32(1))
                    if has_short and rsq < short_cutsq:
                        if snb_count < max_snb:
                            short_neighbors[snb_base + snb_count] = Int32(j)
                            snb_count += 1
                        else:
                            _ = Atomic.fetch_add(overflow_ptr, Int32(1))

    # sentinel — single -1 after the last valid entry (stale data beyond is fine)
    if nb_count < max_nb:
        neighbors[nb_base + nb_count] = Int32(-1)
    if has_short and snb_count < max_snb:
        short_neighbors[snb_base + snb_count] = Int32(-1)


fn nlist_build_gpu(
    mut atoms:       GPUAtoms,
    cell_count_dev:  DeviceBuffer[DType.int32],
    cell_atoms_dev:  DeviceBuffer[DType.int32],
    neighbors_dev:   DeviceBuffer[DType.int32],
    short_neighbors_dev: DeviceBuffer[DType.int32],
    overflow_dev:    DeviceBuffer[DType.int32],
    nc_x: Int, nc_y: Int, nc_z: Int,
    cell_lx: Float32, cell_ly: Float32, cell_lz: Float32,
    cutsq: Float32, short_cutsq: Float32,
    ctx: DeviceContext,
) raises:
    var nc = nc_x * nc_y * nc_z
    var n_total = atoms.n()

    # zero cell_count, overflow
    var blk_nc = ceildiv(nc, _BLOCK_SIZE)
    ctx.enqueue_function[_zero_int_kernel, _zero_int_kernel](
        cell_count_dev.unsafe_ptr(), nc, grid_dim=blk_nc, block_dim=_BLOCK_SIZE,
    )
    ctx.enqueue_function[_zero_int_kernel, _zero_int_kernel](
        overflow_dev.unsafe_ptr(), 1, grid_dim=1, block_dim=1,
    )

    var blk_nt = ceildiv(n_total, _BLOCK_SIZE)
    ctx.enqueue_function[_cell_scatter_kernel, _cell_scatter_kernel](
        atoms.x.unsafe_ptr(),
        cell_count_dev.unsafe_ptr(),
        cell_atoms_dev.unsafe_ptr(),
        overflow_dev.unsafe_ptr(),
        n_total,
        nc_x, nc_y, nc_z,
        cell_lx, cell_ly, cell_lz,
        MAX_ATOMS_PER_CELL,
        grid_dim  = blk_nt,
        block_dim = _BLOCK_SIZE,
    )

    var blk_nl = ceildiv(atoms.nlocal, _BLOCK_SIZE)
    ctx.enqueue_function[_nlist_build_kernel, _nlist_build_kernel](
        atoms.x.unsafe_ptr(),
        cell_count_dev.unsafe_ptr(),
        cell_atoms_dev.unsafe_ptr(),
        neighbors_dev.unsafe_ptr(),
        short_neighbors_dev.unsafe_ptr(),
        overflow_dev.unsafe_ptr(),
        atoms.nlocal,
        nc_x, nc_y, nc_z,
        cell_lx, cell_ly, cell_lz,
        cutsq, short_cutsq,
        MAX_ATOMS_PER_CELL,
        MAX_NEIGHBORS_PER_ATOM,
        MAX_SHORT_NEIGHBORS_PER_ATOM,
        grid_dim  = blk_nl,
        block_dim = _BLOCK_SIZE,
    )

    # Check overflow once — single 4-byte readback per rebuild.
    var h_of = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(h_of, overflow_dev)
    ctx.synchronize()
    if Int(h_of[0]) != 0:
        raise Error("GPU cell/neighbor build overflowed: increase MAX_ATOMS_PER_CELL or MAX_NEIGHBORS_PER_ATOM.")
