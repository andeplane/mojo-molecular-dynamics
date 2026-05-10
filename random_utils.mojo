from std.math import sqrt, log, cos, pi
from atom import Atoms

comptime TWO_PI: Float64 = 2.0 * pi


fn _lcg(mut state: Int) -> Float64:
    """Linear congruential generator. Returns uniform float in (0, 1)."""
    state = (state * 1664525 + 1013904223) & 0x7FFFFFFF
    return Float64(state) / Float64(0x7FFFFFFF)


fn _gauss(mut state: Int) -> Float64:
    """Box-Muller transform. Returns one standard normal sample."""
    var u1 = _lcg(state)
    while u1 <= 1e-30:
        u1 = _lcg(state)
    var u2 = _lcg(state)
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2)


fn init_velocities_mb(mut atoms: Atoms, temperature: Float64, seed: Int = 42):
    """
    Assign velocities from a Maxwell-Boltzmann distribution.

    temperature is in units where kB = 1 (e.g. eV for metal units).
    Net momentum is removed after sampling.
    """
    var n = atoms.nlocal
    var state = seed

    var px: Float64 = 0.0
    var py: Float64 = 0.0
    var pz: Float64 = 0.0
    var total_mass: Float64 = 0.0

    for i in range(n):
        var std = sqrt(temperature / atoms.mass[i])
        var vx = std * _gauss(state)
        var vy = std * _gauss(state)
        var vz = std * _gauss(state)
        atoms.v[3 * i]     = vx
        atoms.v[3 * i + 1] = vy
        atoms.v[3 * i + 2] = vz
        px += atoms.mass[i] * vx
        py += atoms.mass[i] * vy
        pz += atoms.mass[i] * vz
        total_mass += atoms.mass[i]

    # Remove net momentum (zero centre-of-mass drift)
    var inv_m = 1.0 / total_mass
    for i in range(n):
        atoms.v[3 * i]     -= px * inv_m
        atoms.v[3 * i + 1] -= py * inv_m
        atoms.v[3 * i + 2] -= pz * inv_m
