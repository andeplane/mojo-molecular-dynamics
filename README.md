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

Install [Magic](https://docs.modular.com/magic/) (Modular's package manager):

```bash
curl -ssL https://magic.modular.com | bash
```

## Quick start

```bash
# Built-in demos: 108-atom Ar FCC (LJ) + 9-atom SiO₂ (Vashishta)
magic run mojo run main.mojo

# Drive from a JSON config
magic run mojo run main.mojo examples/argon.json
magic run mojo run main.mojo examples/sio2.json

# Run the test suite
magic run mojo test test/
# or via the project task shortcut
magic run test
```

## Project layout

```
mojo-md/
├── atom.mojo           — Atoms struct (SoA layout), PBC utilities
├── ghost.mojo          — Ghost atom builder + reverse-comm force fold
├── neighbor.mojo       — Cell-list neighbor list (CSR)
├── pair_style.mojo     — PairStyle trait
├── pair_lj.mojo        — Lennard-Jones (parallelised, energy-shifted)
├── pair_vashishta.mojo — Vashishta 2+3-body potential
├── integrator.mojo     — Integrator trait + VelocityVerlet
├── simulation.mojo     — Generic Simulation[P, I] loop driver
├── random_utils.mojo   — LCG RNG, Box-Muller, Maxwell-Boltzmann init
├── io.mojo             — JSON config loader (via Python interop)
├── main.mojo           — Entry point + built-in demos
├── examples/
│   ├── argon.json      — 108-atom Ar FCC, LJ, 1 000 steps
│   └── sio2.json       — 9-atom SiO₂, Vashishta, 200 steps
└── test/               — Unit and integration tests (~50 cases)
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
