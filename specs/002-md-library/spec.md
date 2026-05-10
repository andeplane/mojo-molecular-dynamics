# Feature Specification: MD Library Interface

**Feature Branch**: `002-md-library`  
**Created**: 2026-05-10  
**Status**: Draft  
**Input**: User description: "I want the mojo MD code to be a library that can be run inside another codebase, not only as a standalone. make a good library with decent interface"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Programmatic Simulation Setup (Priority: P1)

A researcher or developer working in their own Mojo codebase imports the MD library and runs a full molecular dynamics simulation in a few lines of code — no JSON config file, no CLI, no subprocess. They construct atoms, choose a pair style, and call `run()` directly from their own `main()` function.

**Why this priority**: This is the core value of making MD a library. Everything else builds on this. Without it, nothing else matters.

**Independent Test**: Import the library from a separate Mojo file, construct an LJ argon simulation, run 100 steps, and verify that the returned kinetic energy is a positive finite number.

**Acceptance Scenarios**:

1. **Given** an external Mojo file that imports the MD library, **When** the user constructs `Atoms`, `PairLJ`, and `Simulation` and calls `sim.run(100)`, **Then** the simulation completes without error and returns thermodynamic values.
2. **Given** a simulation configured entirely in code, **When** it runs, **Then** the results are numerically identical to the equivalent standalone run with the same parameters.
3. **Given** an external codebase with no MD-specific files present, **When** the library is referenced via a standard Mojo package path, **Then** all required types and functions are accessible through import statements.

---

### User Story 2 - Post-Step State Access (Priority: P2)

A developer wants to inspect or extract simulation data at each timestep from their own code — positions, velocities, forces, and energies — so they can feed them into analysis routines, visualizers, or custom stopping criteria.

**Why this priority**: Reading simulation state is essential for any non-trivial usage of the library. Without it, the caller is blind to what the simulation is doing.

**Independent Test**: Run a simulation for 10 steps, extracting kinetic energy after each step, and verify the values change over time as expected from a thermalizing system.

**Acceptance Scenarios**:

1. **Given** a running simulation, **When** the caller invokes a single-step method instead of `run(N)`, **Then** it can read atom positions and energies between steps.
2. **Given** a completed run, **When** the caller accesses the simulation's atom data, **Then** all particle data (positions, velocities, forces, masses, types) is readable.
3. **Given** a user who needs custom logic per step, **When** they use the step-by-step interface, **Then** no internal state is hidden or inaccessible.

---

### User Story 3 - Custom Pair Style / Integrator Plug-in (Priority: P3)

An advanced user implements their own pair potential or time integrator conforming to the library's traits and passes it directly to `Simulation` — the library compiles and runs it without any modification to library source files.

**Why this priority**: The existing code is already generic over pair style and integrator traits. Making this composability explicit in the public interface is a key library quality differentiator.

**Independent Test**: Implement a trivial no-op pair style conforming to the `PairStyle` trait in an external file, pass it to `Simulation`, run one step, and verify compilation succeeds and no forces are applied.

**Acceptance Scenarios**:

1. **Given** a user-defined struct that implements the `PairStyle` trait, **When** it is passed to `Simulation`, **Then** it compiles and runs correctly.
2. **Given** a user-defined struct that implements the `Integrator` trait, **When** it is passed to `Simulation`, **Then** it is used for the integration loop.
3. **Given** the trait definitions published as part of the library's public API, **When** a user reads them, **Then** the required methods are clearly specified with documented semantics.

---

### User Story 4 - Preserve Standalone Executable (Priority: P4)

Existing users who run the tool via `mojo run main.mojo -in config.json` or `--demo lj` continue to do so without any changes to their workflow after the library refactor.

**Why this priority**: Backward compatibility matters. Existing users and CI pipelines must not break.

**Independent Test**: Run `mojo run main.mojo --demo lj` after the refactor and verify the same output as before.

**Acceptance Scenarios**:

1. **Given** the refactored codebase, **When** a user runs the CLI with `-in config.json`, **Then** it behaves identically to the pre-refactor version.
2. **Given** the refactored codebase, **When** all existing tests are run, **Then** they all pass without modification.

---

### Edge Cases

- What happens when a user imports the library but provides an invalid number of atom types to a pair style?
- How does the library behave when `Atoms` is constructed with zero particles?
- How does the package handle naming conflicts if the user's codebase has its own `atom` or `simulation` module?
- What if the caller accesses atom data while ghost atoms are stale (between neighbor list rebuilds)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The library MUST be structured as a Mojo package so external codebases can import all public types with a single import path (e.g., `from mojo_md import Simulation, Atoms, PairLJ`).
- **FR-002**: Users MUST be able to construct and run a complete simulation entirely in code, without any JSON configuration file or CLI arguments.
- **FR-003**: The library MUST expose a single-step execution method on `Simulation` (in addition to `run(N)`) so callers can inspect or modify state between timesteps.
- **FR-004**: The library MUST expose the `PairStyle` and `Integrator` traits as public API so users can implement their own and compose them with `Simulation`.
- **FR-005**: The library MUST allow read access to all atom data (positions, velocities, forces, masses, type IDs, tags) after any step or run completes.
- **FR-006**: The existing `main.mojo` standalone entry point MUST continue to work unchanged after the library restructure.
- **FR-007**: All existing tests MUST pass after the library restructure without rewriting test logic or assertions. Mechanical import-path updates (e.g., `from atom import` → `from mojo_md import`) are permitted.
- **FR-008**: The library MUST include at least one usage example showing the minimum code required to start a simulation (the canonical "getting started" reference).

### Key Entities

- **Atoms**: Particle data container — positions, velocities, forces, masses, type IDs. Public and readable by library consumers.
- **Simulation**: The main driver, generic over pair style and integrator. Primary entry point for library consumers.
- **PairStyle** (trait): Defines the interface for interatomic potentials. Must be public.
- **Integrator** (trait): Defines the interface for time integration schemes. Must be public.
- **PairLJ**: Lennard-Jones pair style — bundled implementation, exported from the library.
- **PairVashishta**: Vashishta three-body pair style — bundled implementation, exported from the library.
- **VelocityVerlet**: Standard velocity-Verlet integrator — bundled implementation, exported from the library.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An external Mojo codebase can set up and run a simulation in 15 lines of code or fewer, using only import statements (no copy-pasting of source files).
- **SC-002**: All existing tests pass without modification after the restructure.
- **SC-003**: Simulation results (energies, final positions) from library usage match standalone results to floating-point precision for identical inputs.
- **SC-004**: A user-defined pair style or integrator can be plugged in and compiled without modifying any library source file.
- **SC-005**: The standalone CLI (`-in`, `--demo`) continues to work identically after the restructure.
- **SC-006**: All exported types and traits have at least one usage example accessible to users of the library.

## Assumptions

- The library will be consumed by Mojo codebases only (no Python or C bindings) in this version; cross-language bindings are out of scope.
- The package name will be `mojo_md`; the exact name can be adjusted during planning if Mojo tooling conventions differ.
- Mojo's package system (a directory with an `__init__.mojo` re-exporting public symbols) is the standard mechanism; no separate build manifest is required beyond what the Mojo toolchain supports.
- The single-step interface (FR-003) is achieved by exposing a `step()` method on `Simulation`, not a callback or closure mechanism.
- Scientific correctness is validated by comparing against the existing standalone run, not against an external reference implementation.
- GPU acceleration (tracked in feature 001) is out of scope for this feature; the library interface design should not preclude it but need not enable it.
