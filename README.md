# mojo-molecular-dynamics

A molecular dynamics (MD) simulator written in [Mojo](https://www.modular.com/mojo) 1.0.

Implements the LAMMPS-style simulation loop with compile-time generic pair styles and integrators, ghost-atom periodic boundary conditions, and a parallelised force kernel designed for GPU porting.

## Features

- **Structure-of-Arrays atom storage** — cache-friendly, SIMD-ready
- **Ghost atoms for PBC** — LAMMPS-style halo exchange, MPI-ready design
- **Full neighbor list (CSR format)** — thread-safe parallel force evaluation
- **Pair styles**: Lennard-Jones (multi-type) and Vashishta (2- and 3-body)
- **Velocity Verlet integrator** — time-reversible, symplectic
- **Compile-time generics** — `Simulation[P: PairStyle, I: Integrator]`
- **JSON input files** — drive simulations without touching Mojo code
- **~50 unit and integration tests**

## Requirements

Install Mojo into a Python virtual environment using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv .venv
source .venv/bin/activate
uv pip install mojo
```

## Library Usage

Import `mojo_md` from any Mojo file by passing `-I /path/to/repo`:

```mojo
from mojo_md import (
    Atoms, Simulation, PairLJ, VelocityVerlet, init_velocities_mb
)

fn main():
    var atoms = Atoms(4, 10.0, 10.0, 10.0)
    atoms.x[0] = 1.0; atoms.x[1] = 1.0; atoms.x[2] = 1.0
    atoms.x[3] = 5.0; atoms.x[4] = 1.0; atoms.x[5] = 1.0
    atoms.x[6] = 1.0; atoms.x[7] = 5.0; atoms.x[8] = 1.0
    atoms.x[9] = 5.0; atoms.x[10] = 5.0; atoms.x[11] = 5.0
    for i in range(4):
        atoms.mass[i] = 39.948; atoms.type_id[i] = 0; atoms.tag[i] = i

    var pair = PairLJ(1)
    pair.set_pair(0, 0, 0.01040, 3.4, 8.5)
    init_velocities_mb(atoms, 0.0081)

    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, VelocityVerlet()^, dt=0.002
    )
    sim.run(1000, print_interval=100)
    print("Final KE =", sim.atoms.kinetic_energy())
```

```bash
mojo run -I /path/to/mojo-md your_script.mojo
```

For a step-by-step loop with per-step state access, use `sim.step()`:

```mojo
for _ in range(100):
    var pe = sim.step()
    var ke = sim.atoms.kinetic_energy()
    print("PE =", pe, "KE =", ke)
```

See `examples/` for `library_usage.mojo`, `step_loop.mojo`, `custom_pair.mojo`, and `vashishta_usage.mojo`.

## Quick start

```bash
source .venv/bin/activate

# Built-in demos: 108-atom Ar FCC (LJ) + 9-atom SiO₂ (Vashishta)
mojo run main.mojo --demo lj
mojo run main.mojo --demo vashishta

# Drive from a JSON config
mojo run main.mojo -in examples/argon.json
mojo run main.mojo -in examples/sio2.json

# Run the test suite (-I . so test files can import the mojo_md package)
for t in atom ghost neighbor lj vashishta integrator integration; do
  mojo run -I . "test/test_${t}.mojo"
done
```

## Project layout

```
mojo-md/
├── mojo_md/                 — Importable Mojo package
│   ├── __init__.mojo        — Re-exports full public API
│   ├── atom.mojo            — Atoms struct (SoA layout), PBC utilities
│   ├── ghost.mojo           — Ghost atom builder + reverse-comm force fold
│   ├── neighbor.mojo        — Cell-list neighbor list (CSR)
│   ├── pair_style.mojo      — PairStyle trait
│   ├── pair_lj.mojo         — Lennard-Jones (parallelised, energy-shifted)
│   ├── pair_vashishta.mojo  — Vashishta 2+3-body potential
│   ├── integrator.mojo      — Integrator trait + VelocityVerlet
│   ├── simulation.mojo      — Generic Simulation[P, I] loop driver
│   ├── random_utils.mojo    — LCG RNG, Box-Muller, Maxwell-Boltzmann init
│   └── sim_io.mojo          — JSON config loader (via Python interop)
├── main.mojo                — CLI entry point + built-in demos
├── examples/
│   ├── argon.json           — 108-atom Ar FCC, LJ, 1 000 steps
│   ├── sio2.json            — 9-atom SiO₂, Vashishta, 200 steps
│   ├── library_usage.mojo   — Minimal 15-line LJ example
│   ├── step_loop.mojo       — Manual step() loop with per-step KE/PE
│   ├── custom_pair.mojo     — User-defined HarmonicPair style
│   └── vashishta_usage.mojo — Library usage with Vashishta potential
└── test/                    — Unit and integration tests (~50 cases)
```

## JSON input format

```json
{
  "box": [16.215, 16.215, 16.215],
  "atom_types": [{"name": "Ar", "mass": 39.948}],
  "lattice": {"type": "fcc", "a": 5.405, "nx": 3, "ny": 3, "nz": 3, "atom_type": "Ar"},
  "velocities": {"type": "maxwell_boltzmann", "temperature": 0.0081, "seed": 42},
  "pair_style": "lj",
  "lj_pairs": [{"types": ["Ar","Ar"], "epsilon": 0.01040, "sigma": 3.4, "cutoff": 8.5}],
  "run": {"timestep": 0.002, "nsteps": 1000, "skin": 0.3, "rebuild_every": 10, "print_every": 100}
}
```

Supported `pair_style` values: `"lj"`, `"vashishta"`. See `examples/sio2.json` for the Vashishta triplet schema.

## Design notes

### Parallelisation

Two-body forces are computed in parallel over the outer atom loop using `from algorithm import parallelize`. Each thread writes only to `f[i]`, so no atomics are needed — the full neighbor list stores both i→j and j→i, so each thread owns its output slice.

Three-body (Vashishta) forces are computed serially: the apex-j-k loop applies Newton's third law by writing to `f[j]` and `f[k]`, which would require atomic float additions for safe parallel execution. A GPU port would use `atomic_add` on the force arrays.

### Units

LAMMPS *metal* units throughout: Å, eV, atomic mass units, ps.

### Neighbor list and ghost rebuild

Ghosts and the neighbor list are rebuilt every `rebuild_every` steps with a skin distance. Energy conservation degrades if atoms move more than `skin/2` between rebuilds; `skin ≥ 0.3 Å` and `rebuild_every ≤ 20` is safe for most liquid/solid systems near equilibrium.
