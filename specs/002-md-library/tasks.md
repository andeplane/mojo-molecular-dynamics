# Tasks: MD Library Interface

**Input**: Design documents from `specs/002-md-library/`  
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Organization**: Tasks grouped by user story for independent implementation and testing.  
**Tests**: Not TDD — existing 7 test files are updated (import paths only, no logic changes) as part of the refactor.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no conflicts)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup

**Purpose**: Create package directory. No source files moved or changed yet.

- [X] T001 Create `mojo_md/` directory at repo root (the Mojo package container)

---

## Phase 2: Foundational — Move Modules into Package

**Purpose**: Physically relocate all library modules into `mojo_md/` and fix intra-package imports. Blocking prerequisite for all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. Work in dependency order within the phase.

### Group A — Atom (no internal deps)

- [X] T002 Move `atom.mojo` → `mojo_md/atom.mojo`; verify it imports only `std.math` (no internal deps to fix)

### Group B — Atom dependents (parallel, after T002)

- [X] T003 [P] Move `ghost.mojo` → `mojo_md/ghost.mojo`; update `from atom import` → `from mojo_md.atom import`
- [X] T004 [P] Move `neighbor.mojo` → `mojo_md/neighbor.mojo`; update `from atom import` → `from mojo_md.atom import`
- [X] T005 [P] Move `integrator.mojo` → `mojo_md/integrator.mojo`; update `from atom import` → `from mojo_md.atom import`
- [X] T006 [P] Move `random_utils.mojo` → `mojo_md/random_utils.mojo`; update `from atom import` → `from mojo_md.atom import`

### Group C — Pair style + implementations (after T002, T004)

- [X] T007 Move `pair_style.mojo` → `mojo_md/pair_style.mojo`; update imports of `atom`, `neighbor`
- [X] T008 [P] Move `pair_lj.mojo` → `mojo_md/pair_lj.mojo`; update imports of `atom`, `neighbor`, `pair_style` (after T007)
- [X] T009 [P] Move `pair_vashishta.mojo` → `mojo_md/pair_vashishta.mojo`; update imports of `atom`, `neighbor`, `pair_style` (after T007)

### Group D — Simulation driver + IO (after all above)

- [X] T010 Move `simulation.mojo` → `mojo_md/simulation.mojo`; update all imports (`atom`, `ghost`, `integrator`, `neighbor`, `pair_style`)
- [X] T011 Move `sim_io.mojo` → `mojo_md/sim_io.mojo`; update all imports (`atom`, `simulation`, `pair_lj`, `pair_vashishta`, `integrator`, `random_utils`)

**Checkpoint**: All 10 modules live in `mojo_md/` with correct intra-package imports. Root contains only `main.mojo`, `test/`, `examples/`, `mojo_md/`.

---

## Phase 3: User Story 1 — Programmatic Simulation Setup (Priority: P1)

**Goal**: External Mojo code can `from mojo_md import Simulation, Atoms, PairLJ` and run a full simulation.

**Independent Test**: `mojo run -I . examples/library_usage.mojo` completes with positive KE printed.

### Implementation for User Story 1

- [X] T012 [US1] Write `mojo_md/__init__.mojo` re-exporting all public symbols per `specs/002-md-library/contracts/library-api.md` — `Atoms`, `Simulation`, `PairStyle`, `Integrator`, `PairLJ`, `PairVashishta`, `VashishtaParam`, `VelocityVerlet`, `NeighborList`, `init_velocities_mb`, `wrap_into_box`
- [X] T013 [US1] Update `main.mojo` imports from flat (`from atom import`, `from simulation import`, etc.) to package imports (`from mojo_md import Atoms, Simulation, ...`)
- [X] T014 [P] [US1] Update imports in `test/test_atom.mojo` to use `from mojo_md import Atoms, wrap_into_box, minimum_image`
- [X] T015 [P] [US1] Update imports in `test/test_ghost.mojo` — `GhostBuilder` is internal; import from `mojo_md.ghost` directly rather than the top-level package
- [X] T016 [P] [US1] Update imports in `test/test_neighbor.mojo` to use `from mojo_md import Atoms, NeighborList`
- [X] T017 [P] [US1] Update imports in `test/test_lj.mojo` to use `from mojo_md import Atoms, PairLJ, NeighborList`
- [X] T018 [P] [US1] Update imports in `test/test_vashishta.mojo` to use `from mojo_md import Atoms, PairVashishta, VashishtaParam, NeighborList`
- [X] T019 [P] [US1] Update imports in `test/test_integrator.mojo` to use `from mojo_md import Atoms, VelocityVerlet`
- [X] T020 [US1] Update imports in `test/test_integration.mojo` to use `from mojo_md import ...` (all types it uses)
- [X] T021 [US1] Run all 7 tests and confirm they pass: `for t in atom ghost neighbor lj vashishta integrator integration; do mojo run -I . "test/test_${t}.mojo"; done`

**Checkpoint**: `from mojo_md import Simulation, Atoms, PairLJ, VelocityVerlet` works from any Mojo file given `-I /path/to/repo`. All 7 existing tests pass.

---

## Phase 4: User Story 2 — Post-Step State Access (Priority: P2)

**Goal**: Callers can invoke `sim.step()` in a manual loop and read `sim.atoms` state between steps.

**Independent Test**: Write a 10-step loop using `sim.step()`, print KE each iteration, verify values differ across steps and remain positive.

### Implementation for User Story 2

- [X] T022 [US2] Rename `var step: Int` → `var step_count: Int` in `mojo_md/simulation.mojo` and update all internal references (`self.step` → `self.step_count`, print logic in `run()`) — required to avoid field/method name collision before adding the `step()` method
- [X] T023 [US2] Add `fn step(mut self) -> Float64` method to `mojo_md/simulation.mojo` — executes one full velocity-Verlet timestep (half-v, full-x, conditional nlist rebuild, zero-forces, pair.compute, reverse-comm, half-v), increments `self.step_count`, and returns PE
- [X] T024 [US2] Refactor `Simulation.run()` in `mojo_md/simulation.mojo` to delegate to `self.step()` rather than duplicating the timestep logic inline — verify `run()` output is numerically identical to pre-refactor for both the argon (LJ) and SiO₂ (Vashishta) demos
- [X] T025 [US2] Add `examples/step_loop.mojo` demonstrating the manual step loop pattern from `specs/002-md-library/quickstart.md` — sets up argon LJ, calls `sim.step()` 100 times, prints PE + KE each step

**Checkpoint**: `sim.step()` exists, returns PE, and `sim.atoms` is readable between calls. `sim.run()` produces identical results to the pre-refactor standalone for both pair styles.

---

## Phase 5: User Story 3 — Custom Pair Style / Integrator Plug-in (Priority: P3)

**Goal**: A user-defined struct implementing `PairStyle` in an external file compiles and runs with `Simulation`.

**Independent Test**: `mojo run -I . examples/custom_pair.mojo` compiles and runs using a user-defined `HarmonicPair` without modifying any `mojo_md/` source file.

### Implementation for User Story 3

- [X] T026 [US3] Verify `PairStyle` and `Integrator` traits are re-exported in `mojo_md/__init__.mojo` (should be covered by T012; confirm explicitly)
- [X] T027 [US3] Verify `NeighborList` is re-exported in `mojo_md/__init__.mojo` so custom `PairStyle` implementors can type-annotate the `nlist` parameter of `compute()`
- [X] T028 [US3] Create `examples/custom_pair.mojo` implementing `HarmonicPair` from `specs/002-md-library/quickstart.md` — imports only from `mojo_md`, not from submodules; runs a 4-atom simulation for 50 steps and prints final KE

**Checkpoint**: `examples/custom_pair.mojo` compiles and runs using only `from mojo_md import ...`; no `mojo_md/` files are modified.

---

## Phase 6: User Story 4 — Preserve Standalone Executable (Priority: P4)

**Goal**: The CLI entry point (`main.mojo`) continues to work identically after the refactor.

**Independent Test**: `mojo run main.mojo --demo lj` produces the same step/PE/KE output as the pre-refactor version.

### Implementation for User Story 4

*(main.mojo imports were updated in T013; this phase is verification only)*

- [X] T029 [US4] Run `mojo run main.mojo --demo lj` and confirm output matches the expected argon demo format (step / PE / KE lines + "Argon demo complete" message)
- [X] T030 [US4] Run `mojo run main.mojo --demo vashishta` and confirm output matches expected SiO₂ demo format
- [X] T031 [US4] Run `mojo run main.mojo -in examples/argon.json` and confirm it completes without error
- [X] T032 [US4] Run `mojo run main.mojo -in examples/sio2.json` and confirm it completes without error

**Checkpoint**: All four CLI invocations work. The standalone is fully backward compatible.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Usage examples, documentation, and final end-to-end verification.

- [X] T033 [P] Create `examples/library_usage.mojo` — the canonical minimal 15-line LJ simulation from `specs/002-md-library/quickstart.md`; run with `mojo run -I . examples/library_usage.mojo` to confirm it works (covers SC-001)
- [X] T034 [P] Create `examples/vashishta_usage.mojo` — library-consumer example using `PairVashishta` and `VashishtaParam` directly (satisfies SC-006 for bundled many-body pair style); run with `mojo run -I . examples/vashishta_usage.mojo`
- [X] T035 [P] Update `README.md`: add a "Library Usage" section showing `from mojo_md import ...`, the 15-line example, and the `-I /path/to/repo` flag for consumers
- [X] T036 [P] Update `AGENTS.md` project layout section to reflect the new `mojo_md/` package structure and updated import paths
- [X] T037 Run full end-to-end verification: all 7 tests + both demos + both JSON configs + all 3 example files (library_usage, vashishta_usage, custom_pair)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1; within Phase 2, work in Groups A → B/C → D
- **Phase 3 (US1)**: Depends on Phase 2 completion — BLOCKS all other user stories
- **Phase 4 (US2)**: Depends on Phase 3 (package must be importable before adding step())
- **Phase 5 (US3)**: Depends on Phase 3 (traits must be exported); can overlap with Phase 4
- **Phase 6 (US4)**: Depends on Phase 3 (main.mojo imports updated); can overlap with Phases 4–5
- **Phase 7 (Polish)**: Depends on Phases 3–6 complete

### Within Phase 2 (Foundational)

```
T001 (create dir)
└── T002 (atom.mojo — no deps)
    ├── T003 [P] (ghost)
    ├── T004 [P] (neighbor)
    ├── T005 [P] (integrator)
    └── T006 [P] (random_utils)
        └── T007 (pair_style — needs atom + neighbor T004)
            ├── T008 [P] (pair_lj)
            └── T009 [P] (pair_vashishta)
                └── T010 (simulation — needs all above)
                    └── T011 (sim_io — needs simulation)
```

### Within Phase 3 (US1)

- T012 (`__init__.mojo`) must come first — everything else imports from it
- T013 (main.mojo) can follow T012
- T014–T019 (test file import updates) are all parallel to each other after T012
- T020 (test_integration) after T014–T019
- T021 (run all tests) last

### Within Phase 4 (US2)

- T022 (rename field) must come before T023 (add method) — name collision otherwise
- T023 before T024 (refactor run() to use step())
- T025 (step_loop.mojo) after T023

### Parallel Opportunities

- Phase 2 Group B (T003–T006): 4 files, all parallel after T002
- Phase 2 Group C (T008–T009): parallel after T007
- Phase 3 test updates (T014–T019): 6 test files, all parallel after T012
- Phases 5 + 6 can be worked simultaneously once Phase 3 is done
- Phase 7 (T033–T036): all four parallel

---

## Parallel Example: Phase 2 Group B

```bash
# After T002 completes, launch simultaneously:
Task: "Move ghost.mojo → mojo_md/ghost.mojo; update from atom import"       # T003
Task: "Move neighbor.mojo → mojo_md/neighbor.mojo; update from atom import" # T004
Task: "Move integrator.mojo → mojo_md/integrator.mojo; update imports"      # T005
Task: "Move random_utils.mojo → mojo_md/random_utils.mojo; update imports"  # T006
```

## Parallel Example: Phase 3 Test Updates

```bash
# After T012 (__init__.mojo) completes, launch simultaneously:
Task: "Update test/test_atom.mojo imports"        # T014
Task: "Update test/test_ghost.mojo imports"       # T015
Task: "Update test/test_neighbor.mojo imports"    # T016
Task: "Update test/test_lj.mojo imports"          # T017
Task: "Update test/test_vashishta.mojo imports"   # T018
Task: "Update test/test_integrator.mojo imports"  # T019
```

---

## Implementation Strategy

### Execution Order

1. Phase 1 + 2 — package scaffolding (foundational, blocks everything)
2. Phase 3 (US1) — importable package, all tests pass
3. Phase 4 (US2) — step-by-step control
4. Phase 5 (US3) — custom pair styles and integrators
5. Phase 6 (US4) — CLI backward compatibility verified
6. Phase 7 — examples, docs, final end-to-end run

All phases are required. The library is not complete until T037 passes.

---

## Notes

- [P] tasks = different files, no write conflicts — safe to run simultaneously
- Intra-package import fix pattern: `from atom import X` → `from mojo_md.atom import X` (or `from .atom import X` if Mojo 0.26.x supports relative imports — verify early in T002)
- `GhostBuilder` is internal (not in `__init__.mojo`); tests that use it import from `mojo_md.ghost` directly (see T015)
- T022 (rename `step` field to `step_count`) is mandatory before T023 — Mojo does not allow a field and method with the same name
- T024 must verify Vashishta numerical equivalence, not just LJ — both pair styles go through `run()`
- Commit after each phase checkpoint, not after every individual task
