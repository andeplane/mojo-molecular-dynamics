from atom import Atoms
from neighbor import NeighborList


trait PairStyle(Movable):
    """
    Interface for pair (and many-body) potentials.

    compute() accumulates forces into atoms.f and returns total potential
    energy. The caller is responsible for zeroing forces before calling.

    Using the full neighbor list, each pair (i,j) is visited twice — once
    when i is the center atom, once when j is. Implementations must divide
    the summed energy by the appropriate overcounting factor (2 for 2-body,
    3 for 3-body triplets).

    The full-list design makes compute() trivially parallelizable over i:
    each iteration writes only to atoms.f[3*i .. 3*i+2].
    """
    fn compute(mut self, mut atoms: Atoms, read nlist: NeighborList) -> Float64:
        ...

    fn cutoff(self) -> Float64:
        ...

    fn short_cutoff(self) -> Float64:
        """Cutoff for 3-body short neighbor list (0 if not needed)."""
        ...
