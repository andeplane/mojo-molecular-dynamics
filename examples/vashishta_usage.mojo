from mojo_md import (
    Atoms, Simulation, PairVashishta, VashishtaParam, VelocityVerlet, init_velocities_mb
)

fn main():
    var lx: Float64 = 9.28; var ly: Float64 = 9.28; var lz: Float64 = 10.22
    var nlocal = 9
    var atoms = Atoms(nlocal, lx, ly, lz)
    var si_mass: Float64 = 28.086; var o_mass: Float64 = 15.999

    var pos_x = List[Float64](); var pos_y = List[Float64](); var pos_z = List[Float64]()
    pos_x.append(0.465*lx); pos_y.append(0.000*ly); pos_z.append(0.000*lz)
    pos_x.append(0.000*lx); pos_y.append(0.465*ly); pos_z.append(0.333*lz)
    pos_x.append(0.535*lx); pos_y.append(0.535*ly); pos_z.append(0.667*lz)
    pos_x.append(0.413*lx); pos_y.append(0.268*ly); pos_z.append(0.119*lz)
    pos_x.append(0.268*lx); pos_y.append(0.413*ly); pos_z.append(0.881*lz)
    pos_x.append(0.732*lx); pos_y.append(0.868*ly); pos_z.append(0.452*lz)
    pos_x.append(0.868*lx); pos_y.append(0.732*ly); pos_z.append(0.548*lz)
    pos_x.append(0.132*lx); pos_y.append(0.587*ly); pos_z.append(0.786*lz)
    pos_x.append(0.587*lx); pos_y.append(0.132*ly); pos_z.append(0.215*lz)

    for i in range(nlocal):
        atoms.x[3*i] = pos_x[i]; atoms.x[3*i+1] = pos_y[i]; atoms.x[3*i+2] = pos_z[i]
        atoms.tag[i] = i
        if i < 3:
            atoms.type_id[i] = 0; atoms.mass[i] = si_mass
        else:
            atoms.type_id[i] = 1; atoms.mass[i] = o_mass

    var pair = PairVashishta(2)
    var p_sisi = VashishtaParam(bigh=0.82023, eta=11.0, zi=1.6, zj=1.6, lambda1=999.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.0, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_sioo = VashishtaParam(bigh=188.0, eta=9.0, zi=1.6, zj=-0.8, lambda1=10.0, bigd=1.245, lambda4=4.43, bigw=22.1179, cut=5.5, bigb=4.7325, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.777)
    var p_osio = VashishtaParam(bigh=188.0, eta=9.0, zi=-0.8, zj=1.6, lambda1=10.0, bigd=1.245, lambda4=4.43, bigw=22.1179, cut=5.5, bigb=19.972, gamma=1.0, r0=2.60, bigc=0.0, costheta=-0.333)
    var p_ooo  = VashishtaParam(bigh=88.0, eta=7.0, zi=-0.8, zj=-0.8, lambda1=10.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.5, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    var p_zero = VashishtaParam(bigh=0.0, eta=9.0, zi=0.0, zj=0.0, lambda1=999.0, bigd=0.0, lambda4=999.0, bigw=0.0, cut=5.0, bigb=0.0, gamma=0.0, r0=0.0, bigc=0.0, costheta=-0.333)
    pair.set_param(0, 0, 0, p_sisi)
    pair.set_param(0, 0, 1, p_zero)
    pair.set_param(0, 1, 0, p_zero)
    pair.set_param(0, 1, 1, p_sioo)
    pair.set_param(1, 0, 0, p_zero)
    pair.set_param(1, 0, 1, p_osio)
    pair.set_param(1, 1, 0, p_zero)
    pair.set_param(1, 1, 1, p_ooo)

    init_velocities_mb(atoms, 0.01)
    var integrator = VelocityVerlet()
    var sim = Simulation[PairVashishta, VelocityVerlet](
        atoms^, pair^, integrator^, dt=0.001, skin=0.3, rebuild_interval=5,
    )
    sim.run(200, print_interval=50)
    print("Vashishta demo complete. Final KE =", sim.atoms.kinetic_energy())
