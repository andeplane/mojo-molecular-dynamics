from algorithm import parallelize
from atom import Atoms


trait Integrator:
    """
    Interface for time integrators.

    The velocity Verlet algorithm splits naturally into two half-steps around
    the force evaluation. Implementations of both half-steps are parallelized
    over local atoms since each atom's update is independent.
    """
    fn half_step_v(mut self, mut atoms: Atoms, dt: Float64):
        """
        v[i] += 0.5 * dt * f[i] / mass[i]  for i in 0..nlocal.
        Called both before and after force evaluation each timestep.
        """
        ...

    fn full_step_x(mut self, mut atoms: Atoms, dt: Float64):
        """
        x[i] += dt * v[i]  for i in 0..nlocal.
        Positions may drift outside the box; wrapping happens during ghost rebuild.
        """
        ...


struct VelocityVerlet(Integrator):
    """
    Standard velocity Verlet / leapfrog integrator (NVE ensemble).

    Integration sequence per timestep:
      1. half_step_v  — advance velocities by dt/2 using old forces
      2. full_step_x  — advance positions by dt using new velocities
      3. [ghost rebuild + neighbor list rebuild if needed]
      4. [force evaluation: zeros f, calls pair.compute()]
      5. half_step_v  — advance velocities by dt/2 using new forces

    Energy conservation is second-order in dt.
    """

    fn __moveinit__(out self, owned other: VelocityVerlet):
        pass

    fn __init__(out self):
        pass

    fn half_step_v(mut self, mut atoms: Atoms, dt: Float64):
        var half_dt = 0.5 * dt
        var nlocal = atoms.nlocal

        @parameter
        fn update_v(i: Int):
            var inv_m = 1.0 / atoms.mass[i]
            atoms.v[3 * i]     += half_dt * atoms.f[3 * i]     * inv_m
            atoms.v[3 * i + 1] += half_dt * atoms.f[3 * i + 1] * inv_m
            atoms.v[3 * i + 2] += half_dt * atoms.f[3 * i + 2] * inv_m

        parallelize[update_v](nlocal)

    fn full_step_x(mut self, mut atoms: Atoms, dt: Float64):
        var nlocal = atoms.nlocal

        @parameter
        fn update_x(i: Int):
            atoms.x[3 * i]     += dt * atoms.v[3 * i]
            atoms.x[3 * i + 1] += dt * atoms.v[3 * i + 1]
            atoms.x[3 * i + 2] += dt * atoms.v[3 * i + 2]

        parallelize[update_x](nlocal)
