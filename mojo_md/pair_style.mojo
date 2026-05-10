from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_md.atom import Atoms, GPUAtoms, GPUNeighborList
from mojo_md.neighbor import NeighborList


trait PairStyle(Movable, ImplicitlyDestructible):
    """Interface for pair (and many-body) potentials.

    Each implementation provides matching CPU and GPU paths that share their
    kernel body — see pair_lj.mojo / pair_vashishta.mojo. compute() returns
    total potential energy and accumulates forces into atoms.f. The caller
    is responsible for zeroing forces before calling.

    For the GPU path the simulation driver owns the device-side parameter
    buffer (created via make_gpu_params) and passes it to compute_gpu so the
    pair style itself can be constructed without a DeviceContext.
    """

    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        ...

    fn cutoff(self) -> Float64:
        ...

    fn short_cutoff(self) -> Float64:
        """Cutoff for 3-body short neighbor list (0 if not needed)."""
        ...

    fn make_gpu_params(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.float64]:
        """Upload the pair's flat params buffer to device memory. Caller owns the result."""
        ...

    fn compute_gpu(
        self,
        mut atoms: GPUAtoms,
        read nlist: GPUNeighborList,
        read params_dev: DeviceBuffer[DType.float64],
        ctx: DeviceContext,
    ) raises -> Float64:
        """GPU twin of compute(). Same algorithm — same kernel body."""
        ...
