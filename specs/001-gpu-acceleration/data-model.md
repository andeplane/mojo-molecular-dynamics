# Data Model: Native GPU Acceleration

**Branch**: `001-gpu-acceleration` | **Date**: 2026-05-10

---

## GPUAtoms (`atom_gpu.mojo`)

GPU-resident mirror of the CPU `Atoms` struct. All hot-path arrays live in `DeviceBuffer` on the GPU. Scalar counters (`nlocal`, `nghost`, `nmax`) remain on the host.

| Field | Type | Description |
|-------|------|-------------|
| `nlocal` | `Int` | Number of real (owned) atoms |
| `nghost` | `Int` | Number of ghost copies |
| `nmax` | `Int` | Allocated capacity (≥ nlocal + nghost) |
| `x` | `DeviceBuffer[DType.float64]` | Positions, len = 3 × nmax, SoA |
| `v` | `DeviceBuffer[DType.float64]` | Velocities, len = 3 × nlocal |
| `f` | `DeviceBuffer[DType.float64]` | Forces, len = 3 × nmax (ghosts accumulate) |
| `mass` | `DeviceBuffer[DType.float64]` | Per-atom mass, len = nmax |
| `type_id` | `DeviceBuffer[DType.int32]` | 0-indexed atom type, len = nmax |
| `tag` | `DeviceBuffer[DType.int32]` | Global atom ID, len = nmax |
| `box` | `DeviceBuffer[DType.float64]` | `[lx, ly, lz]`, len = 3 |
| `source_idx` | `DeviceBuffer[DType.int32]` | `source_idx[g]` = local atom index that ghost `g` was copied from; len = nghost; populated at ghost build |
| `ctx` | `DeviceContext` | Owns all buffers; used for kernel launches and synchronisation |

**State transitions**:
- Created by `GPUAtoms.from_cpu(atoms: Atoms, ctx: DeviceContext)` — uploads all arrays via `ctx.enqueue_copy`.
- Updated in-place by GPU kernels (integrator, force, zero_forces, reverse_comm).
- Read back to CPU only at: print/save steps (scalars — PE, KE) and at program end.
- Positions + ghost data refreshed from CPU after neighbour-list rebuild (CPU→GPU copy of `x[0:3*nmax]`, `type_id`, `tag`, `source_idx`).

---

## GPUNeighborList (`atom_gpu.mojo` or `neighbor.mojo`)

GPU-resident copy of the CSR neighbour list. Built on CPU (see research.md Decision 4), then uploaded.

| Field | Type | Description |
|-------|------|-------------|
| `nlocal` | `Int` | Host-side count; matches GPUAtoms.nlocal |
| `offsets` | `DeviceBuffer[DType.int32]` | CSR row offsets, len = nlocal + 1 |
| `neighbors` | `DeviceBuffer[DType.int32]` | Flat neighbour indices, len = total pairs |
| `short_offsets` | `DeviceBuffer[DType.int32]` | Short-range offsets (Vashishta 3-body), len = nlocal + 1 |
| `short_neighbors` | `DeviceBuffer[DType.int32]` | Short-range neighbour indices |

**State transitions**:
- Built from CPU `NeighborList` via `GPUNeighborList.from_cpu(nlist: NeighborList, ctx: DeviceContext)`.
- Replaced entirely at each neighbour rebuild step (new CPU list built, uploaded).
- Read-only on GPU during force kernel execution.

---

## GPUIntegrator trait (`integrator_gpu.mojo`)

Parallel to the CPU `Integrator` trait, but operates on `GPUAtoms`.

```
trait GPUIntegrator:
    fn half_step_v(mut self, mut atoms: GPUAtoms, dt: Float64)
    fn full_step_x(mut self, mut atoms: GPUAtoms, dt: Float64)
```

**`VelocityVerletGPU`**: implements both steps as GPU kernels (one thread per atom i). Stateless — kernels launched on `atoms.ctx`.

---

## GPUPairStyle trait (`pair_style.mojo` extension or new file)

```
trait GPUPairStyle:
    fn compute_gpu(mut self, mut atoms: GPUAtoms, read nlist: GPUNeighborList) -> Float64
    fn cutoff(self) -> Float64
    fn short_cutoff(self) -> Float64
```

`Float64` return is the PE sum (GPU reduction → scalar transferred to host).

**`PairLJGPU`** (`pair_lj_gpu.mojo`): holds same `LJParams` list as `PairLJ`. Force kernel: one thread per atom i, reads `offsets[i:i+2]`, loops neighbours, accumulates force in registers, writes once. PE accumulated via atomic add to a single `DeviceBuffer[DType.float64]` of length 1.

**`PairVashishtaGPU`** (`pair_vashishta_gpu.mojo`): 2-body kernel identical pattern to LJ. 3-body kernel: one thread per apex i, loops (j, k) pairs from short list, adds to `f[i]` in registers, atomically adds to `f[j]` and `f[k]` in global memory.

---

## SimulationGPU (`simulation_gpu.mojo`)

```
struct SimulationGPU[P: GPUPairStyle, I: GPUIntegrator]:
    gpu_atoms: GPUAtoms
    gpu_nlist: GPUNeighborList
    cpu_atoms: Atoms           # kept for ghost/nlist rebuild
    cpu_nlist: NeighborList    # kept for ghost/nlist rebuild
    ghosts: GhostBuilder       # CPU ghost builder (unchanged)
    pair: P
    integrator: I
    dt, skin, rebuild_interval, step: ...
    ctx: DeviceContext
```

**Per-step flow**:
```
non-rebuild step:
  [GPU] half_step_v → full_step_x → zero_forces → pair.compute_gpu → reverse_comm_gpu → half_step_v
  [GPU→CPU] scalars only at print_interval (PE, KE reduction)

rebuild step (every rebuild_interval):
  [GPU→CPU] copy x[0:3*nlocal] → cpu_atoms.x  (positions of real atoms only)
  [CPU]     wrap_into_box → rebuild_ghosts → nlist.build
  [CPU→GPU] copy x[0:3*nmax], type_id, tag, source_idx, offsets, neighbors
  then continue as non-rebuild step
```

---

## BenchmarkResult (`bench.mojo`)

In-memory record per run, printed as a table row.

| Field | Type | Description |
|-------|------|-------------|
| `pair_style` | `String` | `"LJ"` or `"Vashishta"` |
| `n_atoms` | `Int` | System size |
| `backend` | `String` | `"CPU"` or `"GPU"` |
| `n_steps` | `Int` | Steps completed before timeout |
| `elapsed_s` | `Float64` | Wall-clock seconds |
| `matom_steps_s` | `Float64` | `n_atoms × n_steps / elapsed_s / 1e6` |
| `timed_out` | `Bool` | True if run was skipped due to prior timeout |
| `unavailable` | `Bool` | True if GPU not present |

**Size ladder**: `[1000, 2500, 5000, 10000, 25000, 50000, 100000, 125000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]`

**Timeout logic**: Track `cpu_timed_out: Bool` per pair style. If elapsed for a CPU run exceeds 30 s, mark current and all larger sizes as `TIMEOUT` for that pair style + CPU. GPU runs always attempted (GPU is fast; at 10M atoms GPU may still be under 30 s).

---

## Entities summary

| Entity | File | Relationship |
|--------|------|-------------|
| `GPUAtoms` | `atom_gpu.mojo` | GPU mirror of `Atoms`; owns `DeviceContext` |
| `GPUNeighborList` | `atom_gpu.mojo` | GPU mirror of `NeighborList`; uploaded from CPU |
| `VelocityVerletGPU` | `integrator_gpu.mojo` | GPU impl of `GPUIntegrator`; stateless |
| `PairLJGPU` | `pair_lj_gpu.mojo` | GPU impl of `GPUPairStyle`; holds `LJParams` |
| `PairVashishtaGPU` | `pair_vashishta_gpu.mojo` | GPU impl of `GPUPairStyle`; holds `VashishtaParam` |
| `SimulationGPU` | `simulation_gpu.mojo` | Drives the GPU loop; bridges to CPU for rebuilds |
| `BenchmarkResult` | `bench.mojo` | In-memory result record for reporting |
