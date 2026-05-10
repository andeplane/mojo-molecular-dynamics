# Research: Native GPU Acceleration

**Branch**: `001-gpu-acceleration` | **Date**: 2026-05-10

---

## Decision 1: GPU data representation for `Atoms`

**Decision**: Create a parallel `GPUAtoms` struct holding `DeviceBuffer[DType.float64]` for `x`, `v`, `f`, `mass` and `DeviceBuffer[DType.int64]` for `type_id`, `tag`. Keep the SoA layout identical to the CPU `Atoms` struct. The CPU `Atoms` struct is **not modified**.

**Rationale**: SoA is already in place and is the optimal GPU memory access pattern — adjacent threads loading `x[3i], x[3i+1], x[3i+2]` for consecutive `i` achieve coalesced global memory reads. Creating a parallel struct avoids any risk of breaking the CPU path and keeps compile-time generic dispatch clean (`Simulation[P,I]` vs `SimulationGPU[P,I]`).

**Alternatives considered**:
- Union of CPU/GPU buffers in one `Atoms` struct: rejected because Mojo `comptime if has_accelerator()` means GPU types don't compile on CPU-only builds; merging them would require more conditional compilation throughout the codebase.
- AoS layout: rejected — coalescing worse, no upsides for this kernel pattern.

---

## Decision 2: GPU kernel structure for force loops

**Decision**: One GPU thread per local atom `i`. Each thread loads `offsets[i]` and `offsets[i+1]` from the neighbour list, iterates its neighbour range, accumulates force into a register, then writes the result to `f[3i], f[3i+1], f[3i+2]` once. Block size = 256 (safe default for all NVIDIA and AMD targets).

**Rationale**: The full neighbour list (both `i→j` and `j→i` stored) was designed explicitly for this pattern — no write conflicts, no atomics needed on `f[i]`. This directly matches the `parallelize[update_force](nlocal)` CPU pattern but mapped to GPU warps.

**Alternatives considered**:
- One warp per atom i (warp reduction over neighbours): more complex, marginal benefit only for systems with very large neighbour counts (> ~200); not needed here.
- Half neighbour list with atomics on `f[j]`: rejected — the full list was deliberately chosen to avoid this; switching would require redesigning the existing list builder.

---

## Decision 3: Vashishta 3-body kernel and atomic forces

**Decision**: The 3-body kernel runs with one thread per apex atom `i`. Forces on neighbour atoms `j` and `k` are accumulated via `atomic_add` on the GPU `f` buffer. The 2-body kernel follows the same one-thread-per-i pattern as LJ with no atomics.

**Rationale**: The 3-body loop over (i, j, k) triples has `j` and `k` shared across different `i` threads — write conflicts are unavoidable without atomics. The apex-per-thread mapping minimises the number of atomic operations (one atomic write per `j` and one per `k` per triple), which is acceptable given the smaller system sizes where Vashishta is typically used.

**Alternatives considered**:
- Serial 3-body, GPU 2-body only: leaves significant performance on the table; also complicates the data residency model since f[i] would need to be on both CPU and GPU simultaneously.
- Warp-level reduction per i: more complex implementation; saves atomics but the coordination overhead may not pay off until very large neighbour counts.

---

## Decision 4: Neighbour list build stays on CPU (Phase 1)

**Decision**: The cell-list neighbour list is built on CPU for this PR. Atom positions are copied GPU→CPU before each rebuild (`every rebuild_interval steps`), ghosts are rebuilt and the neighbour list is recomputed on CPU, then new ghost positions and the full neighbour CSR arrays are copied CPU→GPU. This is the one acknowledged CPU round-trip per `rebuild_interval` steps.

**Rationale**: The cell-list builder involves per-cell atomic counters, sorting, and prefix sums — significantly more complex to port to GPU. For the benchmark sizes targeted (up to 10M atoms), the rebuild overhead every 10 steps is ~10% of total runtime and does not dominate. The spec explicitly allows transfers that are "absolutely necessary."

**Alternatives considered**:
- GPU neighbour list (sorted binning + prefix scan): correct design, but out of scope for this PR; documented in AGENTS.md as a follow-on task.
- Keep atoms always on CPU, copy to GPU every step: rejected — violates the spec's zero-copy-per-step constraint and defeats the performance goal.

---

## Decision 5: Ghost reverse-comm on GPU

**Decision**: During ghost build, record `source_idx[g] = i` (the local atom index that ghost `g` was copied from) as a `DeviceBuffer[DType.int32]`. The `reverse_comm` GPU kernel iterates over ghost indices `g ∈ [nlocal, nlocal+nghost)` and atomically adds `f[g*3..g*3+2]` into `f[source_idx[g]*3..source_idx[g]*3+2]`.

**Rationale**: This replaces the O(N_ghost × N_local) tag scan in the CPU version with an O(N_ghost) kernel, and eliminates the CPU transfer for reverse-comm. The source index is cheap to record at ghost build time.

**Alternatives considered**:
- Tag-based lookup on GPU (hash map): more complex; not needed since source_idx can be populated deterministically during ghost build.
- Keep reverse-comm on CPU: would require copying all ghost forces back to CPU every step — unacceptable.

---

## Decision 6: GPU availability detection and --gpu flag

**Decision**: Use `comptime if has_accelerator()` to gate all GPU-specific code (types, kernels, `SimulationGPU`). The `--gpu` argument is parsed at runtime in `main.mojo`. If `--gpu` is passed but `has_accelerator()` is false at compile time, the binary exits with a clear error at startup. GPU-warning logic (when no `--gpu` is given but GPU is present) is also gated by `comptime if has_accelerator()` so the check is only compiled into GPU-capable binaries.

**Rationale**: `comptime if` ensures GPU types aren't compiled into CPU-only builds (Mojo requirement). The runtime `--gpu` flag gives users explicit control without requiring a separate binary.

**Alternatives considered**:
- Two separate binaries (mojo-md-cpu, mojo-md-gpu): cleaner separation but worse UX; one binary with a flag is the standard CLI approach.
- Auto-detect and use GPU by default: rejected — explicit opt-in avoids surprise performance changes and makes benchmarking deterministic.

---

## Decision 7: Benchmark script design

**Decision**: `bench.mojo` is a standalone entry point. It iterates over:
- Pair styles: `{LJ, Vashishta}`
- Backends: `{CPU, GPU}` (GPU skipped silently if unavailable)
- Atom counts: `{1000, 2500, 5000, 10000, 25000, 50000, 100000, 125000, 250000, 500000, 1000000, 2500000, 5000000, 10000000}`

For each combination: run a fixed number of steps (enough to measure; at most 30 s wall-clock). Measure elapsed time, compute `n_atoms × n_steps / elapsed_s / 1e6` → Matom·steps/s. Print results as a table. Saves optional CSV.

The timeout is enforced via a pre-run estimate: if a previous CPU run at a smaller size already hit timeout, skip larger sizes for that backend/pair-style combination (no partial-run timeout needed; size ladder grows fast enough that a single run is either well under 30 s or will far exceed it).

**Rationale**: A hard 30 s syscall timeout in Mojo would require OS-level signal handling; the skip-on-timeout approach is simpler and still correct — once a configuration takes > 30 s, all larger sizes will too.

**Alternatives considered**:
- Thread-based timeout: portable but complex in Mojo; skip-on-extrapolation is simpler and correct for a monotonically increasing workload.
- Fixed step count only: doesn't give fair comparison at different scales; wall-clock-normalised Matom·steps/s is the right metric.

---

## Decision 8: Energy/KE readback (print_interval)

**Decision**: At each `print_interval` step, PE and KE scalars are computed on GPU (reduction kernels or summing on host after GPU→CPU copy of the velocity/force arrays). For simplicity in Phase 1, copy only the scalar result (PE returned from `pair.compute` as a Float64 accumulation on GPU, then transferred to host; KE computed via a GPU reduction over `v[]`).

**Rationale**: Scalar transfers are O(1) and do not affect the per-step data-residency guarantee. PE is a natural output of the force kernel; KE requires a parallel reduction but that is a simple pattern.

**Alternatives considered**:
- Full array readback every print step: unnecessary overhead; only scalars needed.
