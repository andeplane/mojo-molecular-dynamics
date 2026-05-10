# Quickstart: mojo_md Library

**Feature**: 002-md-library  
**Date**: 2026-05-10

---

## Installation

Add the `mojo_md/` directory to your Mojo import path:

```bash
mojo run -I /path/to/mojo-md your_script.mojo
# or if mojo-md is a sibling directory:
mojo run -I .. your_script.mojo
```

---

## Minimal example — Argon LJ simulation

```mojo
from mojo_md import (
    Atoms, Simulation, PairLJ, VelocityVerlet, init_velocities_mb
)

fn main():
    # 1. Create atoms: 4 argon atoms in a 10 Å box
    var atoms = Atoms(4, 10.0, 10.0, 10.0)
    atoms.x[0] = 1.0; atoms.x[1] = 1.0; atoms.x[2] = 1.0
    atoms.x[3] = 5.0; atoms.x[4] = 1.0; atoms.x[5] = 1.0
    atoms.x[6] = 1.0; atoms.x[7] = 5.0; atoms.x[8] = 1.0
    atoms.x[9] = 5.0; atoms.x[10] = 5.0; atoms.x[11] = 5.0
    for i in range(4):
        atoms.mass[i] = 39.948  # argon in amu
        atoms.type_id[i] = 0
        atoms.tag[i] = i

    # 2. Configure LJ potential (ε=0.0104 eV, σ=3.4 Å, cutoff=8.5 Å)
    var pair = PairLJ(1)  # 1 atom type
    pair.set_pair(0, 0, 0.01040, 3.4, 8.5)

    # 3. Initialize Maxwell-Boltzmann velocities at T ≈ 94 K
    init_velocities_mb(atoms, 0.0081)

    # 4. Build simulation
    var integrator = VelocityVerlet()
    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.002
    )

    # 5a. Run 1000 steps (prints thermo every 100 steps)
    sim.run(1000, print_interval=100)

    # 5b. — OR — step manually to access state between steps
    for _ in range(100):
        var pe = sim.step()
        var ke = sim.atoms.kinetic_energy()
        print("PE =", pe, "KE =", ke, "TE =", pe + ke)
```

---

## Custom pair style

```mojo
from mojo_md import Atoms, NeighborList, PairStyle, Simulation, VelocityVerlet

struct HarmonicPair(PairStyle):
    var k: Float64
    var r0: Float64

    fn __init__(out self, k: Float64, r0: Float64):
        self.k = k
        self.r0 = r0

    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        var pe: Float64 = 0.0
        for i in range(atoms.nlocal):
            for idx in range(nlist.offsets[i], nlist.offsets[i+1]):
                var j = nlist.neighbors[idx]
                var dx = atoms.x[3*j]   - atoms.x[3*i]
                var dy = atoms.x[3*j+1] - atoms.x[3*i+1]
                var dz = atoms.x[3*j+2] - atoms.x[3*i+2]
                var r = (dx*dx + dy*dy + dz*dz) ** 0.5
                var dr = r - self.r0
                var f_mag = -self.k * dr / r
                atoms.f[3*i]   += f_mag * dx
                atoms.f[3*i+1] += f_mag * dy
                atoms.f[3*i+2] += f_mag * dz
                pe += 0.5 * self.k * dr * dr
        return pe * 0.5  # full list double-counts

    fn cutoff(self) -> Float64: return self.r0 + 2.0
    fn short_cutoff(self) -> Float64: return 0.0
```

---

## Reading atom data after a run

```mojo
sim.run(500)

# Positions of atom 0
var x0 = sim.atoms.x[0]
var y0 = sim.atoms.x[1]
var z0 = sim.atoms.x[2]

# Kinetic energy
var ke = sim.atoms.kinetic_energy()

# Temperature
var T = sim.atoms.temperature()

# Iterate all real atoms
for i in range(sim.atoms.nlocal):
    var vx = sim.atoms.v[3*i]
    var vy = sim.atoms.v[3*i+1]
    var vz = sim.atoms.v[3*i+2]
```
