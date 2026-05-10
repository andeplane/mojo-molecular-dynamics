from math import sqrt


struct Atoms:
    """
    Per-atom data in Structure-of-Arrays layout.

    Indices 0..nlocal-1 are real (owned) atoms.
    Indices nlocal..nlocal+nghost-1 are ghost copies (PBC images or, later,
    halo atoms from neighboring MPI ranks). Positions and forces span the full
    nlocal+nghost range; velocities are nlocal-only (ghosts do not integrate).
    """
    var nlocal: Int
    var nghost: Int
    var nmax: Int          # allocated capacity (>= nlocal + nghost)

    var x: List[Float64]   # positions [3*nmax]
    var v: List[Float64]   # velocities [3*nlocal]
    var f: List[Float64]   # forces [3*nmax]  -- ghosts accumulate, then reverse-summed
    var mass: List[Float64] # [nmax]
    var type_id: List[Int]  # atom type 0-indexed [nmax]
    var tag: List[Int]      # global atom ID [nmax]

    var box: List[Float64] # [lx, ly, lz] — orthogonal periodic box

    fn __moveinit__(out self, owned other: Atoms):
        self.nlocal = other.nlocal
        self.nghost = other.nghost
        self.nmax = other.nmax
        self.x = other.x^
        self.v = other.v^
        self.f = other.f^
        self.mass = other.mass^
        self.type_id = other.type_id^
        self.tag = other.tag^
        self.box = other.box^

    fn __init__(out self, nlocal: Int, lx: Float64, ly: Float64, lz: Float64):
        self.nlocal = nlocal
        self.nghost = 0
        self.nmax = nlocal + 128  # headroom for ghosts

        self.x = List[Float64](capacity=3 * self.nmax)
        self.v = List[Float64](capacity=3 * nlocal)
        self.f = List[Float64](capacity=3 * self.nmax)
        self.mass = List[Float64](capacity=self.nmax)
        self.type_id = List[Int](capacity=self.nmax)
        self.tag = List[Int](capacity=self.nmax)
        self.box = List[Float64](capacity=3)

        for _ in range(3 * self.nmax):
            self.x.append(0.0)
            self.f.append(0.0)
        for _ in range(3 * nlocal):
            self.v.append(0.0)
        for _ in range(self.nmax):
            self.mass.append(1.0)
            self.type_id.append(0)
            self.tag.append(0)

        self.box.append(lx)
        self.box.append(ly)
        self.box.append(lz)

    fn n(self) -> Int:
        """Total atoms including ghosts."""
        return self.nlocal + self.nghost

    fn zero_forces(mut self):
        """Zero forces for all local + ghost atoms."""
        for idx in range(3 * self.n()):
            self.f[idx] = 0.0

    fn grow(mut self, new_nmax: Int):
        """Grow buffers to accommodate more ghost atoms."""
        if new_nmax <= self.nmax:
            return
        var old_nmax = self.nmax
        self.nmax = new_nmax
        for _ in range(3 * (new_nmax - old_nmax)):
            self.x.append(0.0)
            self.f.append(0.0)
        for _ in range(new_nmax - old_nmax):
            self.mass.append(1.0)
            self.type_id.append(0)
            self.tag.append(0)

    fn kinetic_energy(self) -> Float64:
        """Sum of 0.5 * m * v^2 over local atoms."""
        var ke: Float64 = 0.0
        for i in range(self.nlocal):
            var vx = self.v[3 * i]
            var vy = self.v[3 * i + 1]
            var vz = self.v[3 * i + 2]
            ke += 0.5 * self.mass[i] * (vx * vx + vy * vy + vz * vz)
        return ke

    fn temperature(self, dof: Int = 0) -> Float64:
        """Instantaneous temperature in units where kB = 1."""
        var effective_dof = dof if dof > 0 else 3 * self.nlocal - 3
        if effective_dof <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy() / Float64(effective_dof)


fn minimum_image(dx: Float64, box_len: Float64) -> Float64:
    """Apply minimum image convention for one component."""
    var result = dx
    var half = 0.5 * box_len
    if result > half:
        result -= box_len
    elif result < -half:
        result += box_len
    return result


fn _floor(x: Float64) -> Float64:
    """Floor toward -infinity (unlike Int() which truncates toward zero)."""
    var n = Int(x)
    if Float64(n) > x:
        return Float64(n - 1)
    return Float64(n)


fn wrap_into_box(mut atoms: Atoms):
    """Wrap real atoms back into the primary box [0, L)."""
    for i in range(atoms.nlocal):
        var ix = 3 * i
        for d in range(3):
            var l = atoms.box[d]
            var pos = atoms.x[ix + d]
            pos -= l * _floor(pos / l)
            if pos < 0.0:
                pos += l
            if pos >= l:
                pos -= l
            atoms.x[ix + d] = pos
