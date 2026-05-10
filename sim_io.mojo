from python import Python, PythonObject
from atom import Atoms
from pair_lj import PairLJ
from pair_vashishta import PairVashishta, VashishtaParam
from random_utils import init_velocities_mb


struct SimInput:
    """
    All data parsed from a JSON config file.

    Because Simulation[P, I] is a compile-time generic, we cannot return it
    from a runtime JSON loader. Instead, SimInput holds the pair objects for
    all known styles; the caller checks `pair_style` and constructs the
    appropriate Simulation specialisation.
    """
    var atoms: Atoms
    var pair_style: String       # "lj" | "vashishta"
    var lj_pair: PairLJ          # populated when pair_style == "lj"
    var vashishta_pair: PairVashishta  # populated when pair_style == "vashishta"
    var dt: Float64
    var skin: Float64
    var rebuild_interval: Int
    var nsteps: Int
    var print_interval: Int

    fn __init__(
        out self,
        var atoms: Atoms,
        pair_style: String,
        var lj_pair: PairLJ,
        var vashishta_pair: PairVashishta,
        dt: Float64,
        skin: Float64,
        rebuild_interval: Int,
        nsteps: Int,
        print_interval: Int,
    ):
        self.atoms = atoms^
        self.pair_style = pair_style
        self.lj_pair = lj_pair^
        self.vashishta_pair = vashishta_pair^
        self.dt = dt
        self.skin = skin
        self.rebuild_interval = rebuild_interval
        self.nsteps = nsteps
        self.print_interval = print_interval


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


fn _py_float(obj: PythonObject) raises -> Float64:
    return Float64(obj.__float__())


fn _py_int(obj: PythonObject) raises -> Int:
    return Int(obj.__int__())


fn _py_str(obj: PythonObject) raises -> String:
    return String(obj.__str__())


fn _find_type_id(type_names: List[String], name: String) raises -> Int:
    for k in range(len(type_names)):
        if type_names[k] == name:
            return k
    raise Error("Unknown atom type: " + name)


fn load_input(path: String) raises -> SimInput:
    """
    Parse a JSON simulation config and return a fully populated SimInput.

    JSON schema (see examples/argon.json and examples/sio2.json):
      box          : [lx, ly, lz]
      atom_types   : [{"name": str, "mass": float}, ...]
      lattice      : {type: "fcc", a: float, nx: int, ny: int, nz: int, atom_type: str}
                     OR
      atoms        : [{"id": int, "type": str, "x": f, "y": f, "z": f}, ...]
      velocities   : {"type": "maxwell_boltzmann", "temperature": float, "seed": int}
                     OR omitted (zero velocities)
      pair_style   : "lj" | "vashishta"
      lj_pairs     : [{"types": [str,str], "epsilon": f, "sigma": f, "cutoff": f}]
      vashishta_triplets: [{apex, j, k, H, eta, zi, zj, lambda1, D, lambda4, W,
                            cutoff, B, gamma, r0, C, cos_theta0}, ...]
      run          : {timestep, nsteps, skin, rebuild_every, print_every}
    """
    var json_mod = Python.import_module("json")
    var builtins  = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var data = json_mod.load(f)
    f.close()

    # --- Box ---
    var box_list = data["box"]
    var lx = _py_float(box_list[0])
    var ly = _py_float(box_list[1])
    var lz = _py_float(box_list[2])

    # --- Atom types: name → (index, mass) ---
    var type_list = data["atom_types"]
    var n_types = _py_int(builtins.len(type_list))
    var type_names = List[String]()
    var type_masses = List[Float64]()
    for ti in range(n_types):
        type_names.append(_py_str(type_list[ti]["name"]))
        type_masses.append(_py_float(type_list[ti]["mass"]))

    # --- Atoms: from lattice or explicit list ---
    var atoms: Atoms
    var has_atoms = builtins.bool(data.__contains__("atoms")).__bool__()
    var has_lattice = builtins.bool(data.__contains__("lattice")).__bool__()

    if has_atoms:
        var atom_list = data["atoms"]
        var n = _py_int(builtins.len(atom_list))
        atoms = Atoms(n, lx, ly, lz)
        for i in range(n):
            var a = atom_list[i]
            var tname = _py_str(a["type"])
            var tid = _find_type_id(type_names, tname)
            atoms.x[3 * i]     = _py_float(a["x"])
            atoms.x[3 * i + 1] = _py_float(a["y"])
            atoms.x[3 * i + 2] = _py_float(a["z"])
            atoms.mass[i]      = type_masses[tid]
            atoms.type_id[i]   = tid
            atoms.tag[i]       = _py_int(a["id"])
    elif has_lattice:
        var lat = data["lattice"]
        var lat_type = _py_str(lat["type"])
        if lat_type != "fcc":
            raise Error("Only 'fcc' lattice supported, got: " + lat_type)
        var a   = _py_float(lat["a"])
        var nx_ = _py_int(lat["nx"])
        var ny_ = _py_int(lat["ny"])
        var nz_ = _py_int(lat["nz"])
        var at_name = _py_str(lat["atom_type"])
        var at_id   = _find_type_id(type_names, at_name)
        var n = nx_ * ny_ * nz_ * 4
        atoms = Atoms(n, lx, ly, lz)
        _fcc_lattice(atoms, a, nx_, ny_, nz_, at_id, type_masses[at_id])
    else:
        raise Error("JSON must contain either 'atoms' or 'lattice'")

    # --- Velocities ---
    var has_vel = builtins.bool(data.__contains__("velocities")).__bool__()
    if has_vel:
        var vel = data["velocities"]
        var vel_type = _py_str(vel["type"])
        if vel_type == "maxwell_boltzmann":
            var temp = _py_float(vel["temperature"])
            var has_seed = builtins.bool(vel.__contains__("seed")).__bool__()
            var seed = _py_int(vel["seed"]) if has_seed else 42
            init_velocities_mb(atoms, temp, seed)
        else:
            raise Error("Unknown velocity type: " + vel_type)

    # --- Pair style ---
    var pair_style = _py_str(data["pair_style"])

    var lj_pair   = PairLJ(n_types)
    var vash_pair = PairVashishta(n_types)

    if pair_style == "lj":
        var lj_list = data["lj_pairs"]
        var np = _py_int(builtins.len(lj_list))
        for pi in range(np):
            var p = lj_list[pi]
            var t0 = _find_type_id(type_names, _py_str(p["types"][0]))
            var t1 = _find_type_id(type_names, _py_str(p["types"][1]))
            lj_pair.set_pair(
                t0, t1,
                _py_float(p["epsilon"]),
                _py_float(p["sigma"]),
                _py_float(p["cutoff"]),
            )
    elif pair_style == "vashishta":
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
            vash_pair.set_param(apex_id, j_id, k_id, param)
    else:
        raise Error("Unknown pair_style: " + pair_style)

    # --- Run parameters ---
    var run = data["run"]
    var dt              = _py_float(run["timestep"])
    var nsteps          = _py_int(run["nsteps"])
    var skin            = _py_float(run["skin"])
    var rebuild_interval = _py_int(run["rebuild_every"])
    var print_interval  = _py_int(run["print_every"])

    return SimInput(
        atoms^, pair_style,
        lj_pair^, vash_pair^,
        dt, skin, rebuild_interval, nsteps, print_interval,
    )
