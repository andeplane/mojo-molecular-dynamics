# mojo-md — Agent Reference

A molecular dynamics (MD) simulator written in Mojo. Implements the LAMMPS-style
simulation loop with compile-time generic pair styles and integrators, ghost-atom
periodic boundary conditions, and a parallelised force kernel designed for GPU porting.

---

## Project layout

```
mojo-md/
├── atom.mojo           — Atoms struct (SoA layout), PBC wrap utilities
├── ghost.mojo          — Ghost atom builder + reverse-comm force fold
├── neighbor.mojo       — Cell-list full neighbor list (CSR format)
├── pair_style.mojo     — PairStyle trait definition
├── pair_lj.mojo        — Lennard-Jones potential (multi-type, parallelised)
├── pair_vashishta.mojo — Vashishta 2+3-body potential
├── integrator.mojo     — Integrator trait + VelocityVerlet implementation
├── simulation.mojo     — Generic Simulation[P, I] loop driver
├── random_utils.mojo   — LCG RNG, Box-Muller, Maxwell-Boltzmann init
├── sim_io.mojo         — JSON config loader (via Python interop)
├── main.mojo           — Entry point + built-in demos
├── examples/
│   ├── argon.json      — 108-atom Ar FCC, LJ, 1000 steps
│   └── sio2.json       — 9-atom SiO₂, Vashishta, 200 steps
└── test/               — ~50 unit and integration tests
    ├── test_atom.mojo
    ├── test_ghost.mojo
    ├── test_neighbor.mojo
    ├── test_lj.mojo
    ├── test_vashishta.mojo
    ├── test_integrator.mojo
    └── test_integration.mojo
```

---

## Architecture

### Data model — `atom.mojo`

`Atoms` is a Structure-of-Arrays (SoA) struct, cache-friendly and SIMD-ready:

- `x[3*nmax]` — positions for local + ghost atoms
- `v[3*nlocal]` — velocities (local atoms only)
- `f[3*nmax]` — forces (local + ghost; ghosts accumulate, then reverse-summed)
- `mass[nmax]`, `type_id[nmax]`, `tag[nmax]` — per-atom metadata
- `box[3]` — orthogonal periodic box `[lx, ly, lz]`
- Indices `0..nlocal-1` are real atoms; `nlocal..n()-1` are ghost copies.

### Neighbor list — `neighbor.mojo`

Full CSR neighbor list built via a cell/bin algorithm:

- **Full list**: each pair (i,j) stored twice → force loop parallelises over i
  with no write conflicts; energy divided by 2 to correct double-counting.
- **Short list**: second list for r < r0_max (Vashishta 3-body only).
- GPU-friendly layout: kernel for atom i loads `offsets[i]`, streams `neighbors[offsets[i]:offsets[i+1]]`.

### Ghost atoms — `ghost.mojo`

LAMMPS-style PBC halo exchange for a single process:

1. Wrap real atoms into primary box.
2. Create shifted ghost copies for atoms near any face (within `cutoff+skin`).
3. After force eval, `reverse_comm` folds ghost forces back onto source real atoms (matched by `tag`).

Abstraction matches MPI halo exchange — adding MPI only requires replacing
`rebuild_ghosts` / `reverse_comm` with MPI variants.

### Pair styles — `pair_style.mojo`, `pair_lj.mojo`, `pair_vashishta.mojo`

`PairStyle` trait requires:

```mojo
fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64
fn cutoff(self) -> Float64
fn short_cutoff(self) -> Float64
```

- **PairLJ**: multi-type LJ, energy-shifted at cutoff, parallelised over i via
  `from std.algorithm import parallelize`.
- **PairVashishta**: 2-body + 3-body (apex-j-k), 3-body loop is currently serial
  (requires `atomic_add` on force arrays for safe GPU/parallel execution).

### Integrator — `integrator.mojo`

`Integrator` trait with `half_step_v` and `full_step_x`. `VelocityVerlet` splits
into two half-steps around force evaluation. Both steps parallelised with
`parallelize[fn](nlocal)`.

### Simulation loop — `simulation.mojo`

`Simulation[P: PairStyle, I: Integrator]` — compile-time generics let the
compiler inline and auto-vectorise the hot path.

Each timestep:
1. `half_step_v` (old forces)
2. `full_step_x`
3. Rebuild ghosts + neighbor list every `rebuild_interval` steps
4. `zero_forces` → `pair.compute()` → `reverse_comm`
5. `half_step_v` (new forces)

---

## Units

LAMMPS *metal* units: Å (distance), eV (energy), amu (mass), ps (time).

---

## Running the simulator

```bash
# Setup
uv venv .venv && source .venv/bin/activate && uv pip install mojo

# Run a JSON config
mojo run main.mojo -in examples/argon.json
mojo run main.mojo -in examples/sio2.json

# Built-in demos
mojo run main.mojo --demo lj
mojo run main.mojo --demo vashishta

# Test suite (run from project root)
for t in atom ghost neighbor lj vashishta integrator integration; do
  mojo run -I . "test/test_${t}.mojo"
done
```

---

## CI

GitHub Actions (`.github/workflows/ci.yml`), runs on `macos-latest`:

1. Install uv + Mojo via `uv pip install mojo` into `.venv`
2. Run all 7 test files with `mojo run -I .`
3. Smoke-test both JSON demos

Current Mojo version targeted: **0.26.x** (see recent CI fix commits).

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| Full neighbor list (both i→j and j→i) | Force loop parallelises over i with no atomic writes |
| SoA layout for `Atoms` | Cache-friendly; SIMD-ready; matches GPU memory access patterns |
| Compile-time generics `Simulation[P, I]` | Compiler can inline + auto-vectorise entire hot path |
| Ghost atoms abstract PBC | Identical interface to MPI halo exchange — parallel extension is a drop-in |
| Vashishta 3-body loop is serial | Writes to f[j] and f[k]; needs `atomic_add` for GPU/parallel |

---

## GPU porting notes

The codebase is explicitly designed for GPU porting:

- **Neighbor list** (`neighbor.mojo`): CSR layout noted as "GPU-friendly" in docstring.
- **LJ force loop** (`pair_lj.mojo`): already `parallelize`d over i; maps directly
  to a GPU kernel where each thread handles one atom i.
- **Vashishta 3-body** (`pair_vashishta.mojo`): the apex-j-k loop writes to f[j]
  and f[k]; GPU port requires `atomic_add` on force arrays.
- **`Atoms.zero_forces`**: trivially maps to a GPU memset or parallel fill kernel.
- **Integrator** (`integrator.mojo`): both half-steps are already parallelised;
  each atom update is fully independent — one thread per atom on GPU.

---

## GPU concepts from the docs (what to use and optimise)

> These are the key primitives from the Mojo GPU API that apply to this project.
> **Always check the docs for more details and for APIs not listed here — there may be more relevant utilities.**
> Primary reference: https://mojolang.org/docs/manual/gpu/intro-tutorial
> Further reading: https://mojolang.org/docs/manual/gpu/fundamentals/ and https://mojolang.org/docs/manual/gpu/block-and-warp/

### Core API types

| Type | Import | Purpose |
|------|--------|---------|
| `DeviceContext` | `std.gpu.host` | Represents the GPU; owns memory, queues kernels, synchronises |
| `HostBuffer` | `std.gpu.host` | CPU-side pinned buffer; must be explicitly copied to device |
| `DeviceBuffer` | `std.gpu.host` | GPU global-memory buffer; accessible by all threads in all blocks |
| `TileTensor` | `layout` | Multi-dimensional view over a `DeviceBuffer` with layout control (row-major, tiled, etc.) |

### Kernel function pattern

```mojo
from std.gpu import block_dim, block_idx, thread_idx, global_idx
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.sys import has_accelerator
from std.math import ceildiv

comptime block_size = 256
comptime num_blocks = ceildiv(N, block_size)

def my_kernel(data: TileTensor[DType.float32, ..., MutAnyOrigin]):
    var tid = block_idx.x * block_dim.x + thread_idx.x  # global thread index
    # or equivalently: var tid = global_idx.x
    if tid < N:   # bounds check — last block may be partial
        data[tid] = ...

def main() raises:
    comptime if not has_accelerator():
        print("No GPU")
    else:
        ctx = DeviceContext()
        buf = ctx.enqueue_create_buffer[DType.float32](N)
        tensor = TileTensor(buf, row_major[N]())
        ctx.enqueue_function[my_kernel, my_kernel](
            tensor, grid_dim=num_blocks, block_dim=block_size
        )
        ctx.synchronize()
```

### Thread hierarchy

```
Grid
└── Thread blocks  (block_idx.x / .y / .z)
    └── Warps (32 or 64 threads, GPU-dependent; scheduled together on a SM)
        └── Threads (thread_idx.x / .y / .z)
```

- `grid_dim` and `block_dim` can be 1-D, 2-D, or 3-D tuples.
- **Block size must be a multiple of warp size** (32 NVIDIA, 64 AMD) for full SM utilisation.
- Threads in the same block can use shared memory and synchronise; threads in different blocks cannot communicate directly.
- Typical sweet spots: `block_size = 256` (NVIDIA default-safe); max 1024 threads/block on A100.

### Memory model

| Memory | Scope | Latency | How to use in Mojo |
|--------|-------|---------|-------------------|
| Global memory | All threads, all blocks | High (~400–800 cycles) | `DeviceBuffer` / `TileTensor` |
| Shared memory | Threads within one block | Low (~20 cycles) | Declared with `__shared__` (see fundamentals docs) |
| Registers | One thread | Lowest | Local `var` inside kernel |
| Host (CPU) DRAM | CPU only | N/A from GPU | `HostBuffer`; must `enqueue_copy` to/from device |

Coalesced global memory access (adjacent threads read/write adjacent addresses) is critical for throughput — SoA layout in `Atoms` directly enables this.

### Asynchronous execution model

- Every `ctx.enqueue_*` call is non-blocking; operations are queued and run in order.
- Call `ctx.synchronize()` before reading results back on the CPU or at program end.
- Copy operations: `ctx.enqueue_copy(dst_buf=..., src_buf=...)` handles host↔device and device↔device.

### GPU availability guard

```mojo
from std.sys import has_accelerator

comptime if not has_accelerator():
    print("No compatible GPU found")
else:
    # GPU code here
```

Use `comptime if` so the GPU branch is only compiled when a GPU is present; otherwise the compiler rejects GPU-specific types.

### What to optimise in this project (mapped to doc concepts)

| Component | Current state | GPU approach |
|-----------|--------------|-------------|
| LJ force loop (`pair_lj.mojo`) | `parallelize` over i on CPU | GPU kernel: 1 thread per atom i; `tid = global_idx.x`; bounds-check with `if tid < nlocal` |
| `Atoms.zero_forces` | Serial loop | GPU kernel: 1 thread per element, writes `f[3*tid]=f[3*tid+1]=f[3*tid+2]=0` |
| Velocity Verlet half-steps (`integrator.mojo`) | `parallelize` on CPU | GPU kernel: 1 thread per atom, fully independent |
| Atom positions full-step (`integrator.mojo`) | `parallelize` on CPU | Same as above |
| Neighbor list build (`neighbor.mojo`) | Serial cell loop | Harder; requires atomic cell counters or sorted binning on GPU — do last |
| Vashishta 3-body (`pair_vashishta.mojo`) | Serial | Needs `atomic_add` on `f[j]` and `f[k]`; see block-and-warp docs for warp-level reductions |
| `Atoms` data | CPU `List[Float64]` | Replace with `DeviceBuffer[DType.float64]` + `TileTensor`; keep SoA layout (already correct) |
| Ghost reverse-comm | O(N_ghost × N_local) tag scan | Can be a GPU kernel once atoms live on device; match by `tag[g]` |

### Suggested port order

1. Move `Atoms` arrays to `DeviceBuffer` (`x`, `v`, `f`, `mass`, `type_id`, `tag`).
2. GPU kernel for `zero_forces`.
3. GPU kernel for integrator half-steps and full step.
4. GPU kernel for LJ `compute` (the highest-value target — already structured for it).
5. GPU kernel for Vashishta 2-body; serial 3-body with `atomic_add` for correctness first.
6. GPU neighbor list build (most complex; consider keeping on CPU with `enqueue_copy` each rebuild).

---

## Mojo documentation links

### GPU programming
- **GPU intro tutorial**: https://mojolang.org/docs/manual/gpu/intro-tutorial
- **Mojo GPU manual**: https://docs.modular.com/mojo/manual/gpu/
- **MAX GPU kernels**: https://docs.modular.com/max/tutorials/gpu-functions/

### Language features used in this project
- **Traits**: https://docs.modular.com/mojo/manual/traits
- **Ownership & lifetimes**: https://docs.modular.com/mojo/manual/ownership
- **Parallelism (`parallelize`)**: https://docs.modular.com/mojo/stdlib/algorithm/parallelize/
- **Parameter generics**: https://docs.modular.com/mojo/manual/parameters/
- **Structs**: https://docs.modular.com/mojo/manual/structs
- **SIMD types**: https://docs.modular.com/mojo/stdlib/builtin/simd/

### Standard library
- **stdlib overview**: https://docs.modular.com/mojo/stdlib/
- **algorithm module**: https://docs.modular.com/mojo/stdlib/algorithm/
- **math module**: https://docs.modular.com/mojo/stdlib/math/

### Reference
- **Mojo language manual (root)**: https://docs.modular.com/mojo/manual/
- **Changelog / release notes**: https://docs.modular.com/mojo/changelog
