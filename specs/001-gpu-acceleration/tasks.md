# Tasks: Native GPU Acceleration

**Input**: Design documents from `specs/001-gpu-acceleration/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/cli.md ✓

**Total tasks**: 26  
**Parallelisable**: 10 tasks marked [P]

---

## Phase 1: Setup — GPU Trait Definitions

**Purpose**: Define the GPU-side trait interfaces that all subsequent GPU types must conform to. No existing files are modified here.

- [ ] T001 [P] Define `GPUPairStyle` trait (mirrors `PairStyle` but accepts `GPUAtoms` + `GPUNeighborList`, returns `Float64` PE) in `pair_lj_gpu.mojo` preamble or new `gpu_pair_style.mojo`
- [ ] T002 [P] Define `GPUIntegrator` trait (`half_step_v` + `full_step_x` over `GPUAtoms`) as preamble in `integrator_gpu.mojo`

**Checkpoint**: Both traits compile; no GPU hardware required at this step.

---

## Phase 2: Foundational — Core GPU Data Structures

**Purpose**: `GPUAtoms` and `GPUNeighborList` are required by every GPU kernel and the simulation driver. Must be complete before any Phase 3 work begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Implement `GPUAtoms` struct with `DeviceBuffer[DType.float64]` fields (`x`, `v`, `f`, `mass`) and `DeviceBuffer[DType.int32]` fields (`type_id`, `tag`, `source_idx`), plus `DeviceContext ctx`, scalar counters `nlocal`/`nghost`/`nmax`, and `from_cpu(atoms: Atoms, ctx: DeviceContext) -> GPUAtoms` constructor that uploads all arrays via `ctx.enqueue_copy` in `atom_gpu.mojo`
- [ ] T004 Implement `GPUAtoms.zero_forces()` GPU kernel (one thread per element of `f[0:3*n()]`, block_size=256) in `atom_gpu.mojo`
- [ ] T005 Implement `GPUAtoms.reverse_comm_gpu()` GPU kernel (one thread per ghost g; atomically adds `f[g*3..g*3+2]` into `f[source_idx[g]*3..source_idx[g]*3+2]`) in `atom_gpu.mojo`
- [ ] T006 Implement `GPUNeighborList` struct with `DeviceBuffer[DType.int32]` fields (`offsets`, `neighbors`, `short_offsets`, `short_neighbors`) and `from_cpu(nlist: NeighborList, ctx: DeviceContext) -> GPUNeighborList` constructor in `atom_gpu.mojo`

**Checkpoint**: `GPUAtoms.from_cpu` and `GPUNeighborList.from_cpu` compile and upload without error; `zero_forces` kernel executes without segfault on a small test system.

---

## Phase 3: User Story 1 — GPU-Accelerated Simulation Run (Priority: P1) 🎯 MVP

**Goal**: Full end-to-end GPU simulation loop for both LJ and Vashishta via `mojo run main.mojo --gpu`.

**Independent Test**: Run `mojo run main.mojo --gpu --demo lj` on a GPU-equipped machine; simulation completes, energy is printed, and GPU correctness tests pass (relative energy error < 1e-5 vs CPU reference).

- [ ] T007 [P] [US1] Implement `VelocityVerletGPU.half_step_v` GPU kernel (one thread per atom i: `v[3i..] += 0.5*dt * f[3i..] / mass[i]`) in `integrator_gpu.mojo`
- [ ] T008 [P] [US1] Implement `VelocityVerletGPU.full_step_x` GPU kernel (one thread per atom i: `x[3i..] += dt * v[3i..]`) in `integrator_gpu.mojo`
- [ ] T009 [US1] Implement `PairLJGPU.compute_gpu` kernel (one thread per atom i; loop over `offsets[i]:offsets[i+1]` neighbours; accumulate force in registers; write once to `f[3i..]`; PE accumulated via `atomic_add` into a single-element `DeviceBuffer[DType.float64]`) in `pair_lj_gpu.mojo`; kernel returns PE scalar after `ctx.synchronize()`
- [ ] T010 [US1] Implement `PairVashishtaGPU` 2-body kernel (same 1-thread-per-i pattern as LJ, using full neighbour list) in `pair_vashishta_gpu.mojo`
- [ ] T011 [US1] Implement `PairVashishtaGPU` 3-body kernel (one thread per apex i; loop over `(j, k)` pairs from short neighbour list; accumulate `f[i]` in registers; atomically add to `f[j]` and `f[k]` in GPU global memory) in `pair_vashishta_gpu.mojo`
- [ ] T012 [US1] Implement `SimulationGPU[P: GPUPairStyle, I: GPUIntegrator]` struct and `run()` loop in `simulation_gpu.mojo`: non-rebuild steps are fully GPU-resident; at every `rebuild_interval`-th step, copy real-atom positions `x[0:3*nlocal]` GPU→CPU, run `wrap_into_box` + `rebuild_ghosts` + `nlist.build` on CPU, then copy updated ghost+nlist data CPU→GPU before continuing
- [ ] T013 [US1] Add `--gpu` flag parsing to `main.mojo`; gate GPU dispatch with `comptime if has_accelerator()`; route `--gpu` + LJ demo to `SimulationGPU[PairLJGPU, VelocityVerletGPU]` and `--gpu` + Vashishta demo to `SimulationGPU[PairVashishtaGPU, VelocityVerletGPU]`; `-in` config loading also respects `--gpu` flag via updated `run_from_file` path
- [ ] T014 [P] [US1] Write GPU integrator correctness test: run 100 steps of VelocityVerlet on CPU and VelocityVerletGPU on GPU with identical initial conditions; assert positions and velocities agree element-wise within 1e-10 in `test/test_gpu_integrator.mojo`
- [ ] T015 [P] [US1] Write LJ GPU correctness test: run 200-step argon LJ demo on CPU and GPU; assert final total energy matches within relative error 1e-5 in `test/test_gpu_lj.mojo`
- [ ] T016 [P] [US1] Write Vashishta GPU correctness test: run 100-step SiO₂ demo on CPU and GPU; assert final total energy within relative error 1e-5 in `test/test_gpu_vashishta.mojo`

**Checkpoint**: `mojo run main.mojo --gpu --demo lj` and `--demo vashishta` both complete; GPU test suite passes; no per-step CPU transfers except at rebuild steps.

---

## Phase 4: User Story 2 — CPU-Only Run With GPU Warning (Priority: P2)

**Goal**: When the simulator is run without `--gpu` on a GPU-equipped machine, print a stderr warning recommending the flag. Error gracefully when `--gpu` is passed but no GPU exists.

**Independent Test**: On a GPU machine, run `mojo run main.mojo --demo lj` (no `--gpu`) and verify the warning appears on stderr; then run `mojo run main.mojo --gpu --demo lj` and verify no warning.

- [ ] T017 [US2] Implement GPU detection and warning/error branches in `main.mojo` inside `comptime if has_accelerator()` block: (a) if `--gpu` absent → print `[mojo-md] WARNING: A GPU was detected. For significantly better performance, add the --gpu flag.` to stderr; (b) outside the `comptime if` block, if `--gpu` was requested → print `[mojo-md] ERROR: --gpu requested but no compatible GPU was detected.` to stderr and exit with non-zero status

**Checkpoint**: Warning appears exactly once per CPU run on a GPU machine; no warning on CPU-only machines; error + non-zero exit on `--gpu` with no GPU.

---

## Phase 5: User Story 3 — Benchmark Script (Priority: P3)

**Goal**: `mojo run bench.mojo` iterates over LJ + Vashishta × CPU + GPU × full size ladder; prints Matom·steps/s table; respects 30-second timeout; optionally writes CSV.

**Independent Test**: Run `mojo run bench.mojo` on a GPU machine; verify table is printed, all GPU rows have Matom·steps/s values, large CPU rows are marked TIMEOUT, and the speedup column is populated.

- [ ] T018 [US3] Create `bench.mojo` with: size ladder constant `[1000, 2500, 5000, 10000, 25000, 50000, 100000, 125000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]`; `BenchmarkResult` struct (pair_style, n_atoms, backend, n_steps, elapsed_s, matom_steps_s, timed_out, unavailable); `--csv <path>` flag parsing; per-pair-style `cpu_timed_out: Bool` tracker (skip all larger sizes once a CPU run is too slow); main loop skeleton
- [ ] T019 [US3] Implement LJ CPU benchmark runs in `bench.mojo`: construct FCC Ar lattice at each size, run `Simulation[PairLJ, VelocityVerlet]` for enough steps to fill ~5 s (or until 30 s threshold hit for this size), record elapsed and compute Matom·steps/s; skip larger sizes if timed out
- [ ] T020 [US3] Implement LJ GPU benchmark runs in `bench.mojo` (inside `comptime if has_accelerator()`): same system construction as CPU run, use `SimulationGPU[PairLJGPU, VelocityVerletGPU]`; mark rows `unavailable=true` if no GPU
- [ ] T021 [US3] Implement Vashishta CPU + GPU benchmark runs in `bench.mojo` (random Si/O placement scaled to target N); apply same timeout-skip logic and GPU unavailability handling
- [ ] T022 [US3] Implement stdout table formatter in `bench.mojo`: columns — Pair style, N atoms, Backend, Steps, Matom·steps/s, Speedup (GPU/CPU or N/A or TIMEOUT); print header + separator + one row per result; print TIMEOUT and UNAVAILABLE in the Matom·steps/s column as literals
- [ ] T023 [P] [US3] Implement `--csv` output in `bench.mojo`: write `pair_style,n_atoms,backend,n_steps,elapsed_s,matom_steps_s,timed_out,unavailable` CSV when path provided

**Checkpoint**: `mojo run bench.mojo` produces a complete table; `mojo run bench.mojo --csv out.csv` also writes the CSV; all timeout/unavailable cases handled without crash.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T024 [P] Update `AGENTS.md`: add new GPU file list (`atom_gpu.mojo`, `integrator_gpu.mojo`, `pair_lj_gpu.mojo`, `pair_vashishta_gpu.mojo`, `simulation_gpu.mojo`, `bench.mojo`) to Project layout table; mark GPU porting notes section as complete with implementation status per component
- [ ] T025 [P] Update `CLAUDE.md` project structure section with actual GPU file paths (replacing the placeholder `src/`, `tests/` template)
- [ ] T026 Run `quickstart.md` validation end-to-end: `--gpu --demo lj`, `--gpu --demo vashishta`, `mojo run bench.mojo`, and all GPU test files; confirm output matches quickstart.md examples

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Traits)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 ✓ — **blocks all user story work**
- **Phase 3 (US1)**: Depends on Phase 2 ✓ — GPU simulation end-to-end
- **Phase 4 (US2)**: Depends on Phase 1 ✓ and T013 (--gpu flag parsing infrastructure) — otherwise independent of US1 internals
- **Phase 5 (US3)**: Depends on Phase 3 (needs `SimulationGPU` + `PairLJGPU` + `PairVashishtaGPU`) and Phase 2 (CPU `Simulation` already exists)
- **Phase 6 (Polish)**: Depends on all prior phases complete

### User Story Dependencies

- **US1 (P1)**: Can begin after Phase 2 — no dependency on US2/US3
- **US2 (P2)**: Depends only on T013 (--gpu flag parsing); can be done concurrently with rest of US1
- **US3 (P3)**: Depends on US1 complete (needs `SimulationGPU`, `PairLJGPU`, `PairVashishtaGPU`)

### Within Phase 3 (US1)

T007 + T008 (integrator kernels) can run in parallel with each other and before T012.  
T009 (LJ kernel) can run in parallel with T010+T011 (Vashishta kernels).  
T012 (SimulationGPU) depends on T004 (zero_forces), T005 (reverse_comm), T007, T008, T009 or T010+T011.  
T013 (main.mojo dispatch) depends on T012.  
T014, T015, T016 (tests) can run in parallel after T012 + T013.

### Parallel Opportunities

```bash
# Phase 1 — run together:
T001, T002

# Phase 2 — sequential (each depends on prior):
T003 → T004 → T005 → T006

# Phase 3 — parallel batches:
# Batch A (after Phase 2):
T007, T008, T009, T010   # all parallel (different files/functions)
# Batch B (after T009 ready):
T011                      # depends on T010 (same file, 3-body after 2-body)
# Batch C (after T007, T008, T009, T011):
T012                      # SimulationGPU needs all kernels
# Batch D (after T012):
T013                      # main.mojo dispatch
# Batch E (after T013):
T014, T015, T016          # GPU tests — all parallel

# Phase 5 — sequential within each pair style:
T018 → T019 → T020 → T021 → T022
T023 [P] can be added any time after T018

# Phase 6 — fully parallel:
T024, T025 (docs); T026 last (validation)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only — LJ)

1. Complete Phase 1 (T001, T002) — traits
2. Complete Phase 2 (T003–T006) — GPUAtoms/GPUNeighborList
3. Complete T007, T008 — integrator kernels
4. Complete T009 — LJ force kernel
5. Complete T012 — SimulationGPU loop
6. Complete T013 — --gpu flag dispatch
7. **STOP and VALIDATE**: Run `mojo run main.mojo --gpu --demo lj` and T015 test
8. Vashishta (T010, T011, T016) can follow once LJ is green

### Incremental Delivery

1. Phase 1 + 2 → GPU data structures ready
2. Phase 3 → GPU simulation works for both pair styles (main deliverable)
3. Phase 4 → Warning UX
4. Phase 5 → Benchmark numbers
5. Phase 6 → Docs and validation

---

## Notes

- `[P]` = parallelisable (different files, no shared dependencies)
- `[US1/US2/US3]` = maps to user story in spec.md
- No tests have been requested beyond the GPU correctness checks in Phase 3 (FR-011)
- GPU tests (T014–T016) require GPU hardware; skip in CPU-only CI environments
- Commit after each checkpoint to keep the branch bisectable
- The one acknowledged CPU round-trip per `rebuild_interval` steps (neighbor list rebuild) is by design — see research.md Decision 4
