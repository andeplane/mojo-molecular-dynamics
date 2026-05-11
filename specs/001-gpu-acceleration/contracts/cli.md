# CLI Contract: Extended Interface (GPU Acceleration)

**Branch**: `001-gpu-acceleration` | **Date**: 2026-05-10

---

## Existing interface (unchanged)

```
mojo run main.mojo -in <config.json>
mojo run main.mojo --demo lj
mojo run main.mojo --demo vashishta
```

---

## New flag: `--gpu`

```
mojo run main.mojo --gpu -in <config.json>
mojo run main.mojo --gpu --demo lj
mojo run main.mojo --gpu --demo vashishta
```

`--gpu` may appear anywhere in the argument list. It is combinable with `-in` or `--demo`.

### Behaviour

| Condition | Behaviour |
|-----------|-----------|
| `--gpu` present, GPU detected | Run `SimulationGPU`; no warning |
| `--gpu` present, no GPU detected | Print error to stderr, exit non-zero |
| `--gpu` absent, GPU detected | Run `Simulation` (CPU); print warning to stderr |
| `--gpu` absent, no GPU detected | Run `Simulation` (CPU); no warning |

### Warning format (stderr)

```
[mojo-md] WARNING: A GPU was detected. For significantly better performance, add the --gpu flag.
```

### Error format (stderr)

```
[mojo-md] ERROR: --gpu requested but no compatible GPU was detected.
```

---

## Benchmark entry point: `bench.mojo`

```
mojo run bench.mojo [--csv <output.csv>]
```

| Flag | Description |
|------|-------------|
| `--csv <path>` | Optional. If provided, writes results to a CSV file in addition to stdout table. |

### Output: stdout table

```
Pair style  |  N atoms  |  Backend  |  Steps  |  Matom·steps/s  |  Speedup
------------|-----------|-----------|---------|-----------------|----------
LJ          |     1,000 |  CPU      |  1000   |       12.34     |   —
LJ          |     1,000 |  GPU      |  1000   |       45.67     |  3.70×
LJ          |    10,000 |  CPU      |   500   |        8.20     |   —
LJ          |    10,000 |  GPU      |   500   |      120.50     | 14.70×
LJ          |   100,000 |  CPU      |   TIMEOUT  |      —      |   —
LJ          |   100,000 |  GPU      |   200   |      380.10     |   N/A
...
```

- `Speedup` is `GPU Matom·steps/s ÷ CPU Matom·steps/s`; shown as `N/A` if CPU row is `TIMEOUT`.
- `TIMEOUT` rows: run was not executed because a smaller size already exceeded 30 s on that backend.
- `UNAVAILABLE` rows (GPU column): printed when no GPU was detected at run time.

### Output: CSV format (when `--csv` provided)

```csv
pair_style,n_atoms,backend,n_steps,elapsed_s,matom_steps_s,timed_out,unavailable
LJ,1000,CPU,1000,0.081,12.34,false,false
LJ,1000,GPU,1000,0.022,45.67,false,false
...
```
