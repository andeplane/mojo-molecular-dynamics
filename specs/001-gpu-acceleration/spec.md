# Feature Specification: Native GPU Acceleration

**Feature Branch**: `001-gpu-acceleration`  
**Created**: 2026-05-10  
**Status**: Draft  
**Input**: User description: "Now I want to add native GPU acceleration. it should be with --gpu flag etc, and if you don't do it show a warning / suggestion to run WITH GPU acceleration. We must run the ENTIRE thing on GPU so there is no copy to CPU during a time step (other than ABSOLUTELY necessary such as save state etc). We also need a benchmark script that runs vashishta and LJ to show scale."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - GPU-Accelerated Simulation Run (Priority: P1)

A researcher runs a Lennard-Jones or Vashishta simulation with the `--gpu` flag. Force calculation, time integration, and ghost force reverse-communication execute entirely on the GPU. The neighbour-list and ghost rebuild (which occurs every N time steps, not every step) requires a brief CPU round-trip to position data; all other steps are GPU-resident. At the end of the run, final state is printed to stdout.

**Why this priority**: This is the core value proposition of the feature. Without end-to-end GPU execution, the feature does not exist.

**Independent Test**: Can be fully tested by running `main --gpu` with a small LJ system and verifying that a trajectory is produced and the simulation completes without error.

**Acceptance Scenarios**:

1. **Given** a simulation configuration targeting LJ or Vashishta, **When** the user invokes the simulator with `--gpu`, **Then** the simulation runs to completion using GPU-resident data structures throughout, producing correct output identical (within floating-point tolerance) to the CPU reference.
2. **Given** a GPU-accelerated run is in progress, **When** a neighbour-list rebuild step is reached, **Then** only real-atom positions are transferred CPU↔GPU (not the full simulation state), and GPU execution resumes immediately after the rebuild.
3. **Given** a GPU-accelerated run is in progress, **When** the simulation completes, **Then** final PE and KE are printed to stdout and the program exits cleanly.

---

### User Story 2 - CPU-Only Run With GPU Warning (Priority: P2)

A researcher runs the simulator without the `--gpu` flag on a machine where a compatible GPU is detected. The simulator runs on CPU as before, but prints a clear warning at startup recommending GPU acceleration and showing the exact flag to use. On machines with no detectable GPU, no warning is shown.

**Why this priority**: Discoverability. Users who have GPU hardware should know the GPU path exists and how to use it without reading documentation.

**Independent Test**: Can be fully tested by running `main` without `--gpu` on a GPU-equipped machine and verifying a warning line appears on stderr before the first time-step output; and by running the same on a CPU-only machine and verifying no warning appears.

**Acceptance Scenarios**:

1. **Given** the simulator is invoked without `--gpu` and a compatible GPU is detected, **When** the simulation starts, **Then** a warning is printed to stderr that reads something like: `[WARNING] Running on CPU. For better performance, use the --gpu flag.`
2. **Given** the simulator is invoked without `--gpu` and no compatible GPU is detected, **When** the simulation starts, **Then** no GPU warning is printed and the simulation runs silently on CPU.
3. **Given** the warning is printed, **When** the simulation continues, **Then** all existing CPU behavior is unchanged and no regression occurs.

---

### User Story 3 - Benchmark Script: LJ and Vashishta GPU vs CPU Scaling (Priority: P3)

A developer or researcher runs a benchmark script that executes both the Lennard-Jones and Vashishta pair styles across a fixed ladder of system sizes (1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 125k, … up to 10 million atoms) on both CPU and GPU. Each individual run is subject to a 30-second wall-clock timeout so that slow CPU runs at large sizes are cut off automatically rather than blocking the benchmark indefinitely. Results are printed as a table showing million atom-timesteps per second (Matom·steps/s) for each combination of pair style, system size, and backend.

**Why this priority**: Demonstrates the value of GPU acceleration quantitatively across the full size range relevant to production use. The timeout ensures the benchmark is practical to run even on CPU-only hardware or during CI.

**Independent Test**: Can be fully tested by running the benchmark script on a GPU-equipped machine and verifying a complete table is printed for both pair styles, that timed-out runs are marked as such rather than crashing, and that all GPU rows report a Matom·steps/s figure.

**Acceptance Scenarios**:

1. **Given** the benchmark script is invoked, **When** it runs, **Then** it iterates over every combination of {LJ, Vashishta} × {CPU, GPU} × {1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 125k, … 10M atoms}, executing each run with a 30-second timeout.
2. **Given** a run completes within the timeout, **When** results are printed, **Then** the output shows the system size, pair style, backend, and throughput in million atom-timesteps per second (Matom·steps/s).
3. **Given** a run exceeds the 30-second timeout, **When** results are printed, **Then** the row is marked as `TIMEOUT` and the script continues to the next combination without error.
4. **Given** benchmark results are collected, **When** the script completes, **Then** a summary table shows GPU vs CPU speedup factors (or `N/A` for timed-out CPU rows) for each pair style and system size.
5. **Given** the GPU is unavailable, **When** the benchmark script is invoked, **Then** it runs CPU-only benchmarks (with the same timeout), marks GPU rows as `UNAVAILABLE`, and still produces a valid output table.

---

### Edge Cases

- What happens when `--gpu` is passed but no compatible GPU is detected? The simulator must print a clear error message, not crash silently.
- What happens when save-state I/O is the bottleneck? The benchmark should be able to run with saves disabled or at reduced frequency to isolate compute performance.
- What happens if the system size exceeds GPU memory? A clear, actionable error message should be produced.
- How does the GPU path handle the random number generation used by some initial configurations?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a `--gpu` command-line flag that enables GPU-resident execution for the full simulation loop.
- **FR-002**: When `--gpu` is active, all per-time-step data (atom positions, velocities, forces, neighbor lists) MUST remain on the GPU throughout the time step — no CPU copies between time steps.
- **FR-003**: CPU-to-GPU data transfer MUST only occur at: (a) initialization, when the initial state is loaded onto the GPU; (b) neighbour-list rebuild steps (every `rebuild_interval` time steps), where real-atom positions are briefly copied CPU↔GPU to reconstruct the ghost list and neighbour list on CPU; and (c) output/save events, where scalar results (PE, KE) are read back. No other per-step CPU transfers are permitted.
- **FR-004**: When the simulator is invoked without `--gpu` and a compatible GPU is detected at runtime, it MUST print a warning to stderr recommending GPU acceleration and showing the correct flag syntax. No warning is shown on machines without a detectable GPU.
- **FR-005**: The GPU path MUST support both the Lennard-Jones (LJ) and Vashishta pair styles.
- **FR-006**: If `--gpu` is specified but no compatible GPU is detected, the simulator MUST print a clear error and exit with a non-zero status code rather than silently falling back to CPU.
- **FR-007**: A benchmark script MUST be provided that runs LJ and Vashishta across a fixed size ladder — 1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 125k atoms, continuing in reasonable steps up to 10 million atoms — on both CPU and GPU (when available).
- **FR-008**: Each individual benchmark run MUST be subject to a 30-second wall-clock timeout. Runs that exceed the timeout MUST be recorded as `TIMEOUT` and the script MUST continue to the next combination without error or manual intervention.
- **FR-009**: Benchmark output MUST report throughput in million atom-timesteps per second (Matom·steps/s) for each completed run, plus GPU vs CPU speedup factor (or `N/A` / `TIMEOUT` where applicable).
- **FR-010**: Benchmark results MUST be printed as a human-readable table and optionally saved to a CSV file for downstream analysis.
- **FR-011**: The GPU simulation results MUST be numerically equivalent to CPU results within acceptable floating-point tolerance (to be defined during planning based on precision used).
- **FR-012**: *(Deferred — out of scope for this PR)* Save-state / trajectory-writing is not yet implemented in the CPU path. When that feature is added, the GPU path MUST honor save-state intervals without suppressing GPU execution for subsequent time steps.

### Key Entities

- **SimulationState**: Positions, velocities, forces, and auxiliary per-atom data — must live on GPU memory during time steps.
- **NeighborList**: Per-atom neighbor indices used for force calculation — consumed by GPU force kernels. The list is rebuilt on CPU (using a CPU copy of atom positions) at every `rebuild_interval` steps; indices are then uploaded to GPU. Full GPU-side neighbour-list construction is a future optimisation.
- **PairStyle (LJ / Vashishta)**: Force computation kernels — must execute natively on GPU.
- **Integrator**: Velocity-Verlet (or equivalent) update — must execute on GPU.
- **BenchmarkResult**: Pair style, system size, mode (CPU/GPU), wall-clock time, speedup factor.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A simulation with `--gpu` runs to completion without any per-time-step CPU-GPU data transfers (verifiable via profiling or instrumentation).
- **SC-002**: GPU-accelerated runs produce final energies and positions that match CPU reference runs within a defined tolerance (e.g., relative energy error < 1e-5 for double precision).
- **SC-003**: The benchmark script demonstrates at least 5× speedup over CPU for systems of 50,000+ atoms on a modern GPU for the LJ pair style.
- **SC-004**: The CPU-only path shows the GPU warning on every run where `--gpu` is not specified and a GPU is present — never on GPU-absent machines — with zero false positives or omissions.
- **SC-005**: The benchmark script runs to completion for both LJ and Vashishta on a reference machine, producing a full table with Matom·steps/s figures for all completed runs and `TIMEOUT` markers for runs that exceeded 30 seconds.
- **SC-006**: Error messages for missing GPU are actionable — the user can understand the problem and what to do next without consulting documentation.

## Assumptions

- The project targets systems with a single GPU; multi-GPU support is out of scope for this feature.
- Double-precision floating point is used throughout; mixed-precision is not considered in this spec.
- The existing CPU implementation remains unchanged and continues to work exactly as before (the GPU path is additive).
- Save-state frequency is assumed to be low relative to the number of time steps (e.g., every 100–1000 steps), so the cost of CPU-GPU transfer at save points is acceptable.
- The benchmark script is a standalone script in the project (e.g., `bench/benchmark.mojo` or a shell script), not integrated into the main simulation binary.
- GPU support requires building on a GPU-equipped machine (Mojo's `comptime if has_accelerator()` gates GPU types at compile time). A binary compiled on a GPU machine includes both CPU and GPU execution paths; the `--gpu` flag selects between them at runtime.
- The benchmark size ladder is fixed at: 1k, 2.5k, 5k, 10k, 25k, 50k, 100k, 125k atoms, then continuing in roughly doubling steps up to 10 million atoms. The exact intermediate steps above 125k are to be finalized during planning.
