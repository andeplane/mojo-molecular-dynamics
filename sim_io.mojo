from std.python import Python, PythonObject
from atom import Atoms
from integrator import VelocityVerlet
from pair_lj import PairLJ
from pair_vashishta import PairVashishta, VashishtaParam
from random_utils import init_velocities_mb
from simulation import Simulation


fn _py_float(obj: PythonObject) raises -> Float64:
    return Float64(py=obj)


fn _py_int(obj: PythonObject) raises -> Int:
    return Int(py=obj)


fn _py_str(obj: PythonObject) raises -> String:
    return String(obj)


fn _find_type_id(type_names: List[String], name: String) raises -> Int:
    for k in range(len(type_names)):
        if type_names[k] == name:
            return k
    raise Error("Unknown atom type: " + name)


fn _fcc_lattice(
    mut atoms: Atoms,
    a: Float64, nx: Int, ny: Int, nz: Int,
    type_id: Int, mass: Float64,
):
    """Fill atoms with an FCC crystal: 4-atom basis, nx×ny×nz unit cells."""
    var bx = List[Float64](); var by_ = List[Float64](); var bz = List[Float64]()
    bx.append(0.0); by_.append(0.0); bz.append(0.0)
    bx.append(0.5); by_.append(0.5); bz.append(0.0)
    bx.append(0.5); by_.append(0.0); bz.append(0.5)
    bx.append(0.0); by_.append(0.5); bz.append(0.5)

    var idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in range(4):
                    atoms.x[3 * idx]     = (Float64(ix) + bx[b]) * a
                    atoms.x[3 * idx + 1] = (Float64(iy) + by_[b]) * a
                    atoms.x[3 * idx + 2] = (Float64(iz) + bz[b]) * a
                    atoms.mass[idx]    = mass
                    atoms.type_id[idx] = type_id
                    atoms.tag[idx]     = idx
                    idx += 1


fn _load_atoms_and_types(
    data: PythonObject,
    mut type_names: List[String],
    mut type_masses: List[Float64],
) raises -> Atoms:
    """
    Parse `box`, `atom_types`, and either `atoms` or `lattice` from JSON.
    Populates type_names/type_masses; returns a fully populated Atoms.
    """
    var builtins = Python.import_module("builtins")
    var lx = _py_float(data["box"][0])
    var ly = _py_float(data["box"][1])
    var lz = _py_float(data["box"][2])

    var type_list = data["atom_types"]
    var n_types = _py_int(builtins.len(type_list))
    for ti in range(n_types):
        type_names.append(_py_str(type_list[ti]["name"]))
        type_masses.append(_py_float(type_list[ti]["mass"]))

    var has_atoms = Bool(py=builtins.bool(data.__contains__("atoms")))
    var has_lattice = Bool(py=builtins.bool(data.__contains__("lattice")))

    if has_atoms:
        var atom_list = data["atoms"]
        var n = _py_int(builtins.len(atom_list))
        var atoms = Atoms(n, lx, ly, lz)
        for i in range(n):
            var a = atom_list[i]
            var tid = _find_type_id(type_names, _py_str(a["type"]))
            atoms.x[3 * i]     = _py_float(a["x"])
            atoms.x[3 * i + 1] = _py_float(a["y"])
            atoms.x[3 * i + 2] = _py_float(a["z"])
            atoms.mass[i]      = type_masses[tid]
            atoms.type_id[i]   = tid
            atoms.tag[i]       = _py_int(a["id"])
        return atoms^
    elif has_lattice:
        var lat = data["lattice"]
        if _py_str(lat["type"]) != "fcc":
            raise Error("Only 'fcc' lattice supported")
        var a   = _py_float(lat["a"])
        var nx_ = _py_int(lat["nx"])
        var ny_ = _py_int(lat["ny"])
        var nz_ = _py_int(lat["nz"])
        var at_id = _find_type_id(type_names, _py_str(lat["atom_type"]))
        var n = nx_ * ny_ * nz_ * 4
        var atoms = Atoms(n, lx, ly, lz)
        _fcc_lattice(atoms, a, nx_, ny_, nz_, at_id, type_masses[at_id])
        return atoms^
    else:
        raise Error("JSON must contain either 'atoms' or 'lattice'")


fn _apply_velocities(mut atoms: Atoms, data: PythonObject) raises:
    """Apply optional `velocities` block (Maxwell-Boltzmann) to atoms."""
    var builtins = Python.import_module("builtins")
    if not Bool(py=builtins.bool(data.__contains__("velocities"))):
        return
    var vel = data["velocities"]
    if _py_str(vel["type"]) != "maxwell_boltzmann":
        raise Error("Unknown velocity type")
    var temp = _py_float(vel["temperature"])
    var has_seed = Bool(py=builtins.bool(vel.__contains__("seed")))
    var seed = _py_int(vel["seed"]) if has_seed else 42
    init_velocities_mb(atoms, temp, seed)


fn _read_json(path: String) raises -> PythonObject:
    """Parse a JSON file and return the resulting Python dict."""
    var json_mod = Python.import_module("json")
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var data = json_mod.load(f)
    f.close()
    return data^


fn run_from_file(path: String) raises:
    """
    Parse a JSON config and dispatch on `pair_style` to build and run the
    appropriate Simulation specialisation.

    Schema:
      box, atom_types, lattice|atoms, velocities, pair_style,
      lj_pairs|vashishta_triplets, run
    """
    var data = _read_json(path)
    var pair_style = _py_str(data["pair_style"])
    if pair_style == "lj":
        _run_lj(data)
    elif pair_style == "vashishta":
        _run_vashishta(data)
    else:
        raise Error("Unknown pair_style: " + pair_style)


fn _read_run_params(
    data: PythonObject,
    mut dt: Float64, mut skin: Float64,
    mut nsteps: Int, mut rebuild_every: Int, mut print_every: Int,
) raises:
    var run = data["run"]
    dt            = _py_float(run["timestep"])
    skin          = _py_float(run["skin"])
    nsteps        = _py_int(run["nsteps"])
    rebuild_every = _py_int(run["rebuild_every"])
    print_every   = _py_int(run["print_every"])


fn _run_lj(data: PythonObject) raises:
    var type_names = List[String]()
    var type_masses = List[Float64]()
    var atoms = _load_atoms_and_types(data, type_names, type_masses)
    _apply_velocities(atoms, data)

    var builtins = Python.import_module("builtins")
    var n_types = len(type_names)
    var pair = PairLJ(n_types)

    var lj_list = data["lj_pairs"]
    var np = _py_int(builtins.len(lj_list))
    for pi in range(np):
        var p = lj_list[pi]
        var t0 = _find_type_id(type_names, _py_str(p["types"][0]))
        var t1 = _find_type_id(type_names, _py_str(p["types"][1]))
        pair.set_pair(
            t0, t1,
            _py_float(p["epsilon"]),
            _py_float(p["sigma"]),
            _py_float(p["cutoff"]),
        )

    var dt: Float64 = 0.0; var skin: Float64 = 0.0
    var nsteps: Int = 0; var rebuild_every: Int = 0; var print_every: Int = 0
    _read_run_params(data, dt, skin, nsteps, rebuild_every, print_every)

    var sim = Simulation[PairLJ, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(),
        dt=dt, skin=skin, rebuild_interval=rebuild_every,
    )
    sim.run(nsteps, print_every)


fn _run_vashishta(data: PythonObject) raises:
    var type_names = List[String]()
    var type_masses = List[Float64]()
    var atoms = _load_atoms_and_types(data, type_names, type_masses)
    _apply_velocities(atoms, data)

    var builtins = Python.import_module("builtins")
    var n_types = len(type_names)
    var pair = PairVashishta(n_types)

    var trip_list = data["vashishta_triplets"]
    var nt = _py_int(builtins.len(trip_list))
    for ti in range(nt):
        var t = trip_list[ti]
        var apex_id = _find_type_id(type_names, _py_str(t["apex"]))
        var j_id    = _find_type_id(type_names, _py_str(t["j"]))
        var k_id    = _find_type_id(type_names, _py_str(t["k"]))
        var param = VashishtaParam(
            bigh=_py_float(t["H"]),
            eta=_py_float(t["eta"]),
            zi=_py_float(t["zi"]),
            zj=_py_float(t["zj"]),
            lambda1=_py_float(t["lambda1"]),
            bigd=_py_float(t["D"]),
            lambda4=_py_float(t["lambda4"]),
            bigw=_py_float(t["W"]),
            cut=_py_float(t["cutoff"]),
            bigb=_py_float(t["B"]),
            gamma=_py_float(t["gamma"]),
            r0=_py_float(t["r0"]),
            bigc=_py_float(t["C"]),
            costheta=_py_float(t["cos_theta0"]),
        )
        pair.set_param(apex_id, j_id, k_id, param)

    var dt: Float64 = 0.0; var skin: Float64 = 0.0
    var nsteps: Int = 0; var rebuild_every: Int = 0; var print_every: Int = 0
    _read_run_params(data, dt, skin, nsteps, rebuild_every, print_every)

    var sim = Simulation[PairVashishta, VelocityVerlet](
        atoms^, pair^, VelocityVerlet(),
        dt=dt, skin=skin, rebuild_interval=rebuild_every,
    )
    sim.run(nsteps, print_every)
