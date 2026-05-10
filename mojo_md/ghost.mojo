from mojo_md.atom import Atoms, wrap_into_box


struct GhostBuilder(Movable):
    """
    Generates ghost (image) atoms for periodic boundary conditions.

    For a single-process run, every real atom near a boundary gets one or more
    ghost copies shifted by ±lx, ±ly, ±lz. After force evaluation, forces
    accumulated on ghost atoms are reverse-summed back to the source real atom
    (reverse_comm). This abstraction is identical to what MPI domain
    decomposition uses for halo exchange, so adding true parallel MPI later only
    requires replacing rebuild_ghosts / reverse_comm with MPI variants.
    """
    var cutoff_with_skin: Float64

    fn __init__(out self, cutoff_with_skin: Float64):
        self.cutoff_with_skin = cutoff_with_skin

    fn rebuild_ghosts(mut self, mut atoms: Atoms):
        """
        1. Wrap real atoms into primary box.
        2. Reset ghost count.
        3. For each real atom within cutoff_with_skin of any face, create shifted
           copies for all 26 image directions that overlap the box+skin region.
        """
        wrap_into_box(atoms)
        atoms.nghost = 0

        var lx = atoms.box[0]
        var ly = atoms.box[1]
        var lz = atoms.box[2]
        var rc = self.cutoff_with_skin

        for i in range(atoms.nlocal):
            var xi = atoms.x[3 * i]
            var yi = atoms.x[3 * i + 1]
            var zi = atoms.x[3 * i + 2]

            for sx in range(-1, 2):
                for sy in range(-1, 2):
                    for sz in range(-1, 2):
                        if sx == 0 and sy == 0 and sz == 0:
                            continue
                        var gx = xi + Float64(sx) * lx
                        var gy = yi + Float64(sy) * ly
                        var gz = zi + Float64(sz) * lz
                        # Only add ghost if its shifted position overlaps the box+skin
                        if (gx >= -rc and gx < lx + rc and
                            gy >= -rc and gy < ly + rc and
                            gz >= -rc and gz < lz + rc):
                            self._add_ghost(atoms, i, gx, gy, gz)

    fn _add_ghost(mut self, mut atoms: Atoms, source: Int, gx: Float64, gy: Float64, gz: Float64):
        """Append a ghost atom copied from source real atom."""
        var g = atoms.nlocal + atoms.nghost
        if g >= atoms.nmax:
            atoms.grow(atoms.nmax * 2)

        atoms.x[3 * g] = gx
        atoms.x[3 * g + 1] = gy
        atoms.x[3 * g + 2] = gz
        atoms.f[3 * g] = 0.0
        atoms.f[3 * g + 1] = 0.0
        atoms.f[3 * g + 2] = 0.0
        atoms.mass[g] = atoms.mass[source]
        atoms.type_id[g] = atoms.type_id[source]
        atoms.tag[g] = atoms.tag[source]  # same global ID as source
        atoms.nghost += 1

    fn reverse_comm(mut self, mut atoms: Atoms):
        """
        Accumulate forces from ghost atoms back onto their source real atoms.

        For PBC on a single process, the source atom is identified by matching
        tag (global ID). This O(N_ghost * N_local) scan is cheap for typical
        ghost counts; a production MPI version replaces this with a reverse
        halo exchange.
        """
        for g in range(atoms.nlocal, atoms.n()):
            var g_tag = atoms.tag[g]
            var fx = atoms.f[3 * g]
            var fy = atoms.f[3 * g + 1]
            var fz = atoms.f[3 * g + 2]
            if fx == 0.0 and fy == 0.0 and fz == 0.0:
                continue
            # Find source local atom with matching tag
            for i in range(atoms.nlocal):
                if atoms.tag[i] == g_tag:
                    atoms.f[3 * i] += fx
                    atoms.f[3 * i + 1] += fy
                    atoms.f[3 * i + 2] += fz
                    break
