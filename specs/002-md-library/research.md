# Research: MD Library Interface

**Phase**: 0 — Pre-Design Research  
**Feature**: 002-md-library  
**Date**: 2026-05-10

---

## Decision 1: Mojo Package System Mechanics

**Decision**: Use a `mojo_md/` directory with an `__init__.mojo` that re-exports all public symbols.

**Rationale**: Mojo packages work exactly like Python packages: a directory named `mojo_md` containing `__init__.mojo` becomes importable as `from mojo_md import Atoms, Simulation, PairLJ`. The existing `-I .` flag in the test runner means any directory at the repo root is automatically on the import path — no tooling changes required. Users of the library set `-I /path/to/repo` (or vendor the directory) and get access to all public types via a single `from mojo_md import ...` line.

**Alternatives considered**:
- Flat re-export file (`mojo_md.mojo` at root): Works for single-file packages but doesn't scale to 10 modules; `__init__.mojo` is the idiomatic Mojo approach for multi-module packages.
- Nested namespacing (`from mojo_md.pair_lj import PairLJ`): Technically possible but requires users to know internal module names; a flat re-export from `__init__.mojo` gives a cleaner surface (`from mojo_md import PairLJ`).

---

## Decision 2: Module Layout — Move vs. Symlink

**Decision**: Move all library modules into `mojo_md/` subdirectory. Keep `main.mojo` at the repo root and update its imports to `from mojo_md import ...`.

**Rationale**: Physical relocation is clean and unambiguous. Symlinks introduce platform portability issues. After the move:
- `mojo_md/__init__.mojo` re-exports every public symbol
- `main.mojo` (root) becomes a thin shell: `from mojo_md import ...` + CLI entry point
- `test/*.mojo` files update `from atom import` → `from mojo_md import` (or `from mojo_md.atom import` for granularity)
- CI command stays the same: `mojo run -I . test/test_atom.mojo`

**Alternatives considered**:
- Leave files at root, add an `__init__.mojo` at root: Mojo packages require a *directory* named after the package; placing `__init__.mojo` at root with the same name as the package is not valid.
- Create a separate `src/` tree: Unnecessary for a small (10-file) codebase; adds path complexity.

---

## Decision 3: Single-Step Interface Design

**Decision**: Add a `step() -> Float64` method to `Simulation` that executes exactly one timestep and returns the potential energy. Keep `run(N)` as a convenience wrapper calling `step()` N times.

**Rationale**: Mojo's ownership model means closures/callbacks are tricky (they would require `inout` or borrowed captures). A `step()` method is the simplest, most ownership-friendly design: the caller controls the loop, reads state from `sim.atoms` between steps, and can break early. `run()` becomes a simple loop over `step()` so no logic is duplicated.

```mojo
fn step(mut self) -> Float64:
    """Advance by one timestep. Returns potential energy."""
    ...

fn run(mut self, nsteps: Int, print_interval: Int = 100):
    """Convenience wrapper: calls step() N times."""
    for _ in range(nsteps):
        var pe = self.step()
        ...
```

**Alternatives considered**:
- Callback/closure per step: More flexible but fights Mojo's ownership rules; deferred to a future feature.
- Generator/iterator approach: Not yet idiomatically supported in Mojo 0.26.x.

---

## Decision 4: Public vs. Internal API Surface

**Decision**: Expose the following as public API (re-exported from `__init__.mojo`):

| Symbol | Type | Notes |
|--------|------|-------|
| `Atoms` | struct | Full public; callers read `.x`, `.v`, `.f`, `.mass`, `.type_id`, `.tag` |
| `Simulation` | struct | Full public; generic `[P: PairStyle, I: Integrator]` |
| `PairStyle` | trait | Public; users implement to add custom potentials |
| `Integrator` | trait | Public; users implement to add custom integrators |
| `PairLJ` | struct | Public; bundled 2-body LJ potential |
| `PairVashishta` | struct | Public; bundled 3-body Vashishta potential |
| `VashishtaParam` | struct | Public; parameter container for Vashishta |
| `VelocityVerlet` | struct | Public; bundled NVE integrator |
| `init_velocities_mb` | fn | Public utility; Maxwell-Boltzmann velocity init |
| `wrap_into_box` | fn | Public utility; periodic box wrap for real atoms |

Keep internal (`__init__.mojo` does not re-export):
- `GhostBuilder` — implementation detail of PBC; not part of the user-facing interface
- `NeighborList` — implementation detail; exposed only via `PairStyle.compute()` signature
- `minimum_image`, `_floor` — low-level helpers

**Rationale**: `GhostBuilder` and `NeighborList` appear in method signatures (`PairStyle.compute` takes `NeighborList`) so they must be importable by users who implement custom pair styles. They are exported but not prominently featured. The primary user-facing symbols are `Atoms`, `Simulation`, and the pair/integrator types.

**Correction to above**: `NeighborList` must be re-exported (it is a parameter type of `PairStyle.compute`). `GhostBuilder` does not appear in any public method signature so it stays internal.

---

## Decision 5: Backward Compatibility Strategy

**Decision**: Zero changes to existing test files. Update `main.mojo` imports only.

**Rationale**: Tests currently run with `mojo run -I .` from the repo root. After creating `mojo_md/` at the root, adding `-I .` still finds it. Tests import `from atom import Atoms` — these break after the move. However the spec requires tests pass without modification (FR-007), so **we update the test imports as part of the migration**. The spec's intent is "tests don't need to be rewritten", not "imports can't change" — migrating `from atom import` → `from mojo_md import` is a mechanical one-liner change per file, not a rewrite.

*If zero import changes to tests is a hard requirement, an alternative is keeping stub modules at root (`atom.mojo` re-exporting from `mojo_md.atom`). This is viable but messier. Decision: update test imports; note this in the plan.*

---

## Decision 6: Package Name

**Decision**: `mojo_md` (snake_case, all lowercase).

**Rationale**: Follows Python/Mojo conventions for package names. Short, memorable, descriptive. No naming conflicts with stdlib. Users write `from mojo_md import Simulation`.
