# Library API Contract: mojo_md

**Type**: Mojo Package  
**Import**: `from mojo_md import <Symbol>`  
**Package path**: `mojo_md/` (directory at repo root, contains `__init__.mojo`)  
**Feature**: 002-md-library  
**Date**: 2026-05-10

---

## Import Surface

All symbols below are importable from the top-level package:

```mojo
from mojo_md import (
    Atoms,
    Simulation,
    PairStyle,
    Integrator,
    PairLJ,
    PairVashishta,
    VashishtaParam,
    VelocityVerlet,
    NeighborList,       # needed only when implementing custom PairStyle
    init_velocities_mb,
    wrap_into_box,
)
```

---

## Atoms Contract

```mojo
struct Atoms(Movable):
    var nlocal: Int
    var nghost: Int
    var x: List[Float64]      # [3 * (nlocal + nghost)]
    var v: List[Float64]      # [3 * nlocal]
    var f: List[Float64]      # [3 * (nlocal + nghost)]
    var mass: List[Float64]   # [nlocal + nghost]
    var type_id: List[Int]    # [nlocal + nghost]
    var tag: List[Int]        # [nlocal + nghost]
    var box: List[Float64]    # [lx, ly, lz]

    fn __init__(out self, nlocal: Int, lx: Float64, ly: Float64, lz: Float64)
    fn n(self) -> Int                          # nlocal + nghost
    fn kinetic_energy(self) -> Float64
    fn temperature(self, dof: Int = 0) -> Float64
    fn zero_forces(mut self)
```

**Invariants**:
- `x`, `f` arrays always have capacity `≥ 3 * (nlocal + nghost)` after any `Simulation` operation
- `v` array always has capacity `≥ 3 * nlocal`
- Caller must not write to ghost entries in `v` (indices `nlocal..n()-1` do not exist in `v`)
- Forces in `f` are only valid immediately after `Simulation.step()` or `Simulation.run()` completes

---

## Simulation Contract

```mojo
struct Simulation[P: PairStyle, I: Integrator](Movable):
    var atoms: Atoms
    var step: Int

    fn __init__(
        out self,
        var atoms: Atoms,
        var pair: P,
        var integrator: I,
        dt: Float64,
        skin: Float64 = 0.3,
        rebuild_interval: Int = 10,
    )

    fn step(mut self) -> Float64
    # Advances one timestep. Returns potential energy (eV in metal units).
    # After return, sim.atoms reflects the new positions/velocities/forces.

    fn run(mut self, nsteps: Int, print_interval: Int = 100)
    # Calls step() nsteps times. Prints thermo output every print_interval steps.
```

**Invariants**:
- `step()` always leaves forces consistent with the updated positions (no stale forces)
- After construction, the simulation is already at step 0 with forces computed
- `sim.step` counter increments by 1 per `step()` call
- `run()` is exactly equivalent to a caller loop over `step()` with the same print logic

---

## PairStyle Trait Contract

```mojo
trait PairStyle(Movable, ImplicitlyDestructible):
    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64
    # PRE:  atoms.f is zeroed by the caller
    # POST: atoms.f[3*i .. 3*i+2] contains accumulated force on atom i
    # Returns total potential energy (caller handles ghost force reverse-comm)

    fn cutoff(self) -> Float64
    # Returns the outer interaction cutoff radius.
    # Must be stable (same value for the lifetime of the simulation).

    fn short_cutoff(self) -> Float64
    # Returns the inner cutoff for the 3-body short neighbor list.
    # Return 0.0 if the potential has no 3-body term.
```

**Contract for implementors**:
- Do NOT zero forces inside `compute()` — the caller does this
- `cutoff()` and `short_cutoff()` must return the same values throughout a simulation run
- For 2-body potentials: each pair (i,j) appears twice in the full neighbor list; divide summed energy by 2
- For 3-body potentials: each triplet (i,j,k) appears 6 times; divide 3-body energy by 6

---

## Integrator Trait Contract

```mojo
trait Integrator(Movable, ImplicitlyDestructible):
    fn half_step_v(mut self, mut atoms: Atoms, dt: Float64)
    # v[i] += 0.5 * dt * f[i] / mass[i]  for i in 0..nlocal
    # Must not touch ghost atom velocities (they don't integrate)

    fn full_step_x(mut self, mut atoms: Atoms, dt: Float64)
    # x[i] += dt * v[i]  for i in 0..nlocal
    # Positions may drift outside the box; ghost rebuild handles wrapping
```

**Contract for implementors**:
- Both methods operate only on indices `0..atoms.nlocal`
- `half_step_v` reads `atoms.f` and `atoms.mass`, writes `atoms.v`
- `full_step_x` reads `atoms.v`, writes `atoms.x`

---

## NeighborList Contract (for custom PairStyle implementors)

```mojo
struct NeighborList:
    var offsets: List[Int]          # length nlocal+1 (CSR row offsets)
    var neighbors: List[Int]        # flat neighbor indices
    var offsets_short: List[Int]    # short-list CSR offsets
    var neighbors_short: List[Int]  # short-list neighbor indices
```

**Iteration pattern for atom `i`**:
```mojo
for idx in range(nlist.offsets[i], nlist.offsets[i+1]):
    var j = nlist.neighbors[idx]
    # process pair (i, j)
```

---

## Utility Functions Contract

```mojo
fn init_velocities_mb(mut atoms: Atoms, temperature: Float64)
# Assigns Maxwell-Boltzmann velocities to real atoms at given temperature.
# Removes center-of-mass drift. Units: metal (eV/amu → Å/ps).

fn wrap_into_box(mut atoms: Atoms)
# Wraps real atom positions back into the primary box [0, L).
# Automatically called during ghost rebuild; available for manual use.
```

---

## Versioning Policy

- This is v0 (pre-release). API stability is not guaranteed between minor versions.
- Trait method signatures (`PairStyle`, `Integrator`) are most stable.
- Field access on `Atoms` is stable.
- Internal fields of `Simulation` (e.g., `nlist`, `ghosts`) are not part of the public contract.
