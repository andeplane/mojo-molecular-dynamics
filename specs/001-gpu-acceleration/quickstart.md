# Quickstart: GPU Acceleration

**Branch**: `001-gpu-acceleration` | **Date**: 2026-05-10

---

## Prerequisites

- Mojo 0.26.x installed and on PATH (via `uv pip install mojo` + `.venv` activated)
- A compatible GPU (NVIDIA or AMD) — required only for GPU path
- Project root: `mojo-md/`

---

## Running a simulation on GPU

```bash
# Argon (LJ) demo on GPU
mojo run main.mojo --gpu --demo lj

# SiO₂ (Vashishta) demo on GPU
mojo run main.mojo --gpu --demo vashishta

# JSON config on GPU
mojo run main.mojo --gpu -in examples/argon.json
```

If you omit `--gpu` on a machine with a GPU, you'll see a warning:

```
[mojo-md] WARNING: A GPU was detected. For significantly better performance, add the --gpu flag.
```

If you pass `--gpu` on a machine without a GPU:

```
[mojo-md] ERROR: --gpu requested but no compatible GPU was detected.
```

---

## Running the benchmark

```bash
# Print results table to stdout
mojo run bench.mojo

# Also save as CSV
mojo run bench.mojo --csv results.csv
```

Expected output format:

```
Pair style  |  N atoms  |  Backend  |  Steps  |  Matom·steps/s  |  Speedup
------------|-----------|-----------|---------|-----------------|----------
LJ          |     1,000 |  CPU      |  1000   |       12.34     |   —
LJ          |     1,000 |  GPU      |  1000   |       45.67     |  3.70×
...
LJ          |   100,000 |  CPU      | TIMEOUT |      —          |   —
LJ          |   100,000 |  GPU      |   500   |      380.10     |   N/A
```

---

## Running the test suite

```bash
# Existing CPU tests (unchanged)
for t in atom ghost neighbor lj vashishta integrator integration; do
  mojo run -I . "test/test_${t}.mojo"
done

# GPU tests (requires GPU hardware)
mojo run -I . test/test_gpu_integrator.mojo
mojo run -I . test/test_gpu_lj.mojo
mojo run -I . test/test_gpu_vashishta.mojo
```

GPU tests compare final energies and positions against the CPU reference; they pass if relative energy error < 1e-5.

---

## Key files

| File | Purpose |
|------|---------|
| `atom_gpu.mojo` | `GPUAtoms` struct (DeviceBuffer SoA) |
| `integrator_gpu.mojo` | `VelocityVerletGPU` kernel |
| `pair_lj_gpu.mojo` | LJ GPU pair style |
| `pair_vashishta_gpu.mojo` | Vashishta GPU pair style |
| `simulation_gpu.mojo` | `SimulationGPU[P, I]` loop driver |
| `bench.mojo` | Benchmark entry point |
| `main.mojo` | Entry point — now handles `--gpu` flag |
