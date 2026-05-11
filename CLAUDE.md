# mojo-md Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-05-10

## Active Technologies

- Mojo 0.26.x + `std.gpu`, `std.gpu.host`, `layout` (TileTensor), `std.sys.has_accelerator`, `std.algorithm.parallelize` (CPU path unchanged); Python interop for `sim_io.mojo` (JSON parsing)

## Project Structure

```text
mojo_md/        — importable Mojo package (atom, simulation, pair styles, etc.)
main.mojo       — CLI entry point (--demo lj/vashishta, --gpu flag)
bench.mojo      — Benchmark script (LJ + Vashishta × CPU + GPU)
examples/       — usage examples (library_usage.mojo, step_loop.mojo, etc.)
test/           — unit and integration tests
specs/          — feature specs and contracts
```

## Commands

```bash
# CPU tests (all platforms)
for t in atom ghost neighbor lj vashishta integrator integration; do
  mojo run -I . "test/test_${t}.mojo"
done

# GPU tests (GPU hardware required)
for t in gpu_integrator gpu_lj gpu_vashishta; do
  mojo run -I . "test/test_${t}.mojo"
done

# Demos
mojo run main.mojo --demo lj
mojo run main.mojo --demo vashishta
mojo run main.mojo --gpu --demo lj
mojo run main.mojo --gpu --demo vashishta

# Benchmark
mojo run bench.mojo
mojo run bench.mojo --csv results.csv
```

## Code Style

Mojo 0.26.x: Follow standard conventions

## Recent Changes

- 001-gpu-acceleration: Added GPU path — `GPUAtoms`, `SimulationGPU`, GPU kernels for LJ, Vashishta, integrator; unified CPU+GPU kernel bodies via `enqueue_function`
- 002-md-library: Restructured flat files into `mojo_md/` package with `from mojo_md.xxx import ...` imports; added `step()` API to `Simulation`

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
