# Data Model: MD Library Interface

**Phase**: 1 — Design  
**Feature**: 002-md-library  
**Date**: 2026-05-10

---

## Entities

### Atoms

The central data container. Structure-of-Arrays layout for SIMD and GPU friendliness.

| Field | Type | Description |
|-------|------|-------------|
| `nlocal` | `Int` | Number of owned (real) atoms |
| `nghost` | `Int` | Number of ghost (PBC image) atoms |
| `nmax` | `Int` | Allocated buffer capacity |
| `x` | `List[Float64]` | Positions `[3 * nmax]` — real + ghost |
| `v` | `List[Float64]` | Velocities `[3 * nlocal]` — real only |
| `f` | `List[Float64]` | Forces `[3 * nmax]` — real + ghost |
| `mass` | `List[Float64]` | Per-atom mass `[nmax]` |
| `type_id` | `List[Int]` | Atom type index (0-based) `[nmax]` |
| `tag` | `List[Int]` | Global atom ID `[nmax]` |
| `box` | `List[Float64]` | Box dimensions `[lx, ly, lz]` |

**Access pattern for library consumers**:
- Read positions of atom `i`: `atoms.x[3*i]`, `atoms.x[3*i+1]`, `atoms.x[3*i+2]`
- Read velocities of atom `i`: `atoms.v[3*i]`, `atoms.v[3*i+1]`, `atoms.v[3*i+2]`
- Read forces of atom `i`: `atoms.f[3*i]`, `atoms.f[3*i+1]`, `atoms.f[3*i+2]`
- Iterate real atoms: `for i in range(atoms.nlocal)`
- Total atoms (real + ghost): `atoms.n()` → `atoms.nlocal + atoms.nghost`

**Computed methods** (public):
- `kinetic_energy() -> Float64` — `Σ 0.5 * m * v²` over local atoms
- `temperature(dof: Int = 0) -> Float64` — instantaneous temperature
- `zero_forces()` — zeroes all force entries (called internally before each force eval)
- `n() -> Int` — total atom count including ghosts

---

### Simulation[P: PairStyle, I: Integrator]

Generic driver struct. Compile-time specialisation allows inlining of the entire hot path.

| Field | Type | Description |
|-------|------|-------------|
| `atoms` | `Atoms` | Owned particle data |
| `pair` | `P` | Owned pair style instance |
| `integrator` | `I` | Owned integrator instance |
| `nlist` | `NeighborList` | Internal neighbor list |
| `ghosts` | `GhostBuilder` | Internal ghost atom manager |
| `dt` | `Float64` | Timestep size |
| `skin` | `Float64` | Verlet skin distance |
| `rebuild_interval` | `Int` | Steps between neighbor list rebuilds |
| `step_count` | `Int` | Current step counter |

**Public methods** (library API):
- `__init__(atoms, pair, integrator, dt, skin=0.3, rebuild_interval=10)` — constructor
- `run(nsteps: Int, print_interval: Int = 100)` — run N steps with optional stdout output
- `step() -> Float64` *(new)* — advance one timestep, return potential energy

**State transitions**:
```
Constructed → [step() or run()] → Running → Paused (after step() returns)
                                          → Completed (after run() returns)
Paused → [step() or run()] → Running  (caller can loop manually)
```

---

### PairStyle (trait)

Interface for interatomic potentials. All implementations must satisfy:

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute` | `(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64` | Accumulate forces into `atoms.f`; return total PE |
| `cutoff` | `(self) -> Float64` | Outer interaction cutoff radius |
| `short_cutoff` | `(self) -> Float64` | Inner cutoff for 3-body list (0.0 if unused) |

**Contract**: Caller zeroes forces before calling `compute()`. Implementation must not zero forces itself.

---

### Integrator (trait)

Interface for time integration schemes.

| Method | Signature | Description |
|--------|-----------|-------------|
| `half_step_v` | `(mut self, mut atoms: Atoms, dt: Float64)` | `v += 0.5*dt*f/m` for all real atoms |
| `full_step_x` | `(mut self, mut atoms: Atoms, dt: Float64)` | `x += dt*v` for all real atoms |

---

### PairLJ

Lennard-Jones pair potential. Multi-type support.

| Field | Type | Description |
|-------|------|-------------|
| `ntypes` | `Int` | Number of atom types |
| `eps` | `List[Float64]` | ε per type pair `[ntypes²]` |
| `sig` | `List[Float64]` | σ per type pair `[ntypes²]` |
| `rcut` | `List[Float64]` | Cutoff per type pair `[ntypes²]` |

**Setup method**: `set_pair(i, j, eps, sigma, rcut)` — sets symmetric pair parameters.

---

### PairVashishta

Vashishta 2-body + 3-body potential for covalent systems (e.g., SiO₂).

| Field | Type | Description |
|-------|------|-------------|
| `ntypes` | `Int` | Number of atom types |
| `params` | `List[VashishtaParam]` | Per-triplet parameters `[ntypes³]` |

**Setup method**: `set_param(itype, jtype, ktype, param)` — sets triplet parameters.

---

### VashishtaParam

Parameter container for one atom-type triplet.

| Field | Type | Physical meaning |
|-------|------|-----------------|
| `bigh` | `Float64` | Steric repulsion amplitude H |
| `eta` | `Float64` | Steric repulsion exponent η |
| `zi`, `zj` | `Float64` | Effective charges Z_i, Z_j |
| `lambda1` | `Float64` | Charge-charge screening length λ₁ |
| `bigd` | `Float64` | Charge-dipole amplitude D |
| `lambda4` | `Float64` | Charge-dipole screening length λ₄ |
| `bigw` | `Float64` | Dispersion amplitude W |
| `bigb` | `Float64` | 3-body amplitude B |
| `gamma` | `Float64` | 3-body damping γ |
| `r0` | `Float64` | 3-body inner cutoff r₀ |
| `bigc` | `Float64` | Cosine potential amplitude C |
| `costheta` | `Float64` | Equilibrium cosine of bond angle |
| `cut` | `Float64` | Interaction cutoff |

---

### VelocityVerlet

NVE velocity-Verlet integrator. Stateless (no fields).

---

### NeighborList

Internal. Exposed only because it appears in `PairStyle.compute()` signature.

| Field | Type | Description |
|-------|------|-------------|
| `offsets` | `List[Int]` | CSR row offsets `[nlocal+1]` |
| `neighbors` | `List[Int]` | Flat neighbor indices |
| `offsets_short` | `List[Int]` | Short-range CSR offsets (Vashishta) |
| `neighbors_short` | `List[Int]` | Short-range neighbor indices |

**Library consumers** implementing custom pair styles iterate neighbors as:
```mojo
for idx in range(nlist.offsets[i], nlist.offsets[i+1]):
    var j = nlist.neighbors[idx]
    # compute force between i and j
```

---

## Relationships

```
Simulation[P, I]
├── owns Atoms
├── owns P (PairStyle impl)
├── owns I (Integrator impl)
├── owns NeighborList  (internal)
└── owns GhostBuilder  (internal)

PairStyle.compute(Atoms, NeighborList) reads:
  - Atoms.x, Atoms.type_id, Atoms.nlocal, Atoms.nghost, Atoms.box
  - Atoms.f (writes — accumulates forces)
  - NeighborList.offsets, NeighborList.neighbors

Integrator.half_step_v/full_step_x(Atoms) reads/writes:
  - Atoms.v (reads + writes)
  - Atoms.x (writes in full_step_x)
  - Atoms.f (reads in half_step_v)
  - Atoms.mass (reads)
```
