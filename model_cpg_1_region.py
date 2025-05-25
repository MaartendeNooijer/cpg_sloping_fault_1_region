import numpy as np
import os

from cpg_reservoir_1_region import CPG_Reservoir, save_array, read_arrays, check_arrays, make_burden_layers, make_full_cube
from darts.discretizer import load_single_float_keyword
from darts.engines import value_vector

from darts.tools.gen_cpg_grid import gen_cpg_grid

# from darts.models.cicd_model import CICDModel
from cicd_model_1_region import CICDModel

from darts.engines import value_vector, index_vector

def fmt(x):
    return '{:.3}'.format(x)

#####################################################

class Model_CPG(CICDModel):
    def __init__(self):
        super().__init__()

    def init_reservoir(self):
        if self.idata.generate_grid:
            if self.idata.grid_out_dir is None:
                self.idata.gridname = None
                self.idata.propname = None
            else:  # save generated grid to grdecl files
                os.makedirs(self.idata.grid_out_dir, exist_ok=True)
                self.idata.gridname = os.path.join(self.idata.grid_out_dir, 'grid.grdecl')
                self.idata.propname = os.path.join(self.idata.grid_out_dir, 'reservoir.in')
                arrays = gen_cpg_grid(nx=self.idata.geom.nx, ny=self.idata.geom.ny, nz=self.idata.geom.nz,
                                  dx=self.idata.geom.dx, dy=self.idata.geom.dy, dz=self.idata.geom.dz,
                                  start_z=self.idata.geom.start_z,
                                  permx=self.idata.rock.permx, permy=self.idata.rock.permy, permz=self.idata.rock.permz,
                                  poro=self.idata.rock.poro,
                                  gridname=self.idata.gridname, propname=self.idata.propname)
        else:
            # read grid and rock properties
            arrays = read_arrays(self.idata.gridfile, self.idata.propfile)
            check_arrays(arrays)

            poro_array = arrays.get('PORO')
            if poro_array is not None:
                poro_array[poro_array <= 0] = self.idata.geom.min_poro
                arrays['PORO'] = poro_array

        if self.idata.geom.burden_layers > 0:
            # add over- and underburden layers
            make_burden_layers(number_of_burden_layers=self.idata.geom.burden_layers,
                               initial_thickness=self.idata.geom.burden_init_thickness,
                               property_dictionary=arrays,
                               burden_layer_prop_value=self.idata.rock.burden_prop)

        self.reservoir = CustomCPGReservoir(self.timer, arrays, faultfile=self.idata.faultfile, minpv=self.idata.geom.minpv, )

        self.reservoir.discretize()

        # store modified arrrays (with burden layers) for output to grdecl
        self.reservoir.input_arrays = arrays

        volume = np.array(self.reservoir.mesh.volume, copy=False)
        poro = np.array(self.reservoir.mesh.poro, copy=False)
        print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))

        # imitate open-boundaries with a large volume
        bv = self.idata.geom.bound_volume  # volume, will be assigned to each boundary cell [m3]
        self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
        self.reservoir.apply_volume_depth()

        poro_shale_threshold = self.idata.rock.poro_shale_threshold  # short name
        poro = np.array(self.reservoir.mesh.poro)
        self.reservoir.conduction[poro <= poro_shale_threshold] = self.idata.rock.conduction_shale
        self.reservoir.conduction[poro > poro_shale_threshold] = self.idata.rock.conduction_sand
        self.reservoir.hcap[poro <= poro_shale_threshold] = self.idata.rock.hcap_shale
        self.reservoir.hcap[poro > poro_shale_threshold] = self.idata.rock.hcap_sand

        # add hcap and rcond to be saved into mesh.vtu
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
        self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap, l2g, g2l),
                                           'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l)})

        self.set_physics()

        # time stepping and convergence parameters
        sim = self.idata.sim  # short name
        self.set_sim_params(first_ts=sim.first_ts, mult_ts=sim.mult_ts, max_ts=sim.max_ts, runtime=sim.runtime,
                            tol_newton=sim.tol_newton, tol_linear=sim.tol_linear)
        if hasattr(sim, 'linear_type'):
            self.params.linear_type = sim.linear_type

        self.init_tracked_cells()
        self.timer.node["initialization"].stop()
    #
    # def init_reservoir(self):
    #     if self.idata.generate_grid:
    #         if self.idata.grid_out_dir is None:
    #             self.idata.gridname = None
    #             self.idata.propname = None
    #         else:  # save generated grid to grdecl files
    #             os.makedirs(self.idata.grid_out_dir, exist_ok=True)
    #             self.idata.gridname = os.path.join(self.idata.grid_out_dir, 'grid.grdecl')
    #             self.idata.propname = os.path.join(self.idata.grid_out_dir, 'reservoir.in')
    #         arrays = gen_cpg_grid(nx=self.idata.geom.nx, ny=self.idata.geom.ny, nz=self.idata.geom.nz,
    #                               dx=self.idata.geom.dx, dy=self.idata.geom.dy, dz=self.idata.geom.dz,
    #                               start_z=self.idata.geom.start_z,
    #                               permx=self.idata.rock.permx, permy=self.idata.rock.permy, permz=self.idata.rock.permz,
    #                               poro=self.idata.rock.poro,
    #                               gridname=self.idata.gridname, propname=self.idata.propname)
    #     else:
    #         # read grid and rock properties
    #         arrays = read_arrays(self.idata.gridfile, self.idata.propfile)
    #
    #         # from darts.discretizer import index_vector as index_vector_discr, load_single_int_keyword
    #         # satnum_cpp = index_vector_discr()  # self.discr_mesh.coord
    #         # load_single_int_keyword(satnum_cpp, self.idata.propfile, 'SATNUM', -1)
    #         # arrays['SATNUM'] = np.array(satnum_cpp, copy=False)
    #
    #         check_arrays(arrays)
    #         if self.physics_type == 'deadoil':  # set inactive cells with small porosity (isothermal case)
    #             arrays['ACTNUM'][arrays['PORO'] < self.idata.geom.min_poro] = 0
    #         elif self.physics_type == 'geothermal':  # process cells with small poro (thermal case)
    #             for arr in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
    #                 arrays[arr][arrays['PORO'] < self.idata.geom.min_poro] = self.idata.geom.min_poro
    #
    #     if self.idata.geom.burden_layers > 0:
    #         # add over- and underburden layers
    #         make_burden_layers(number_of_burden_layers=self.idata.geom.burden_layers,
    #                            initial_thickness=self.idata.geom.burden_init_thickness,
    #                            property_dictionary=arrays,
    #                            burden_layer_prop_value=self.idata.rock.burden_prop)
    #
    #     self.reservoir = CPG_Reservoir(self.timer, arrays, minpv=self.idata.geom.minpv,
    #                                    faultfile=self.idata.geom.faultfile)
    #     self.reservoir.discretize()
    #
    #     # store modified arrrays (with burden layers) for output to grdecl
    #     self.reservoir.input_arrays = arrays
    #
    #     volume = np.array(self.reservoir.mesh.volume, copy=False)
    #     poro = np.array(self.reservoir.mesh.poro, copy=False)
    #     print("Pore volume = " + str(sum(volume[:self.reservoir.mesh.n_blocks] * poro)))
    #
    #     # imitate open-boundaries with a large volume
    #     bv = self.idata.geom.bound_volume  # volume, will be assigned to each boundary cell [m3]
    #     self.reservoir.set_boundary_volume(xz_minus=bv, xz_plus=bv, yz_minus=bv, yz_plus=bv)
    #     self.reservoir.apply_volume_depth()
    #
    #     poro_shale_threshold = self.idata.rock.poro_shale_threshold  # short name
    #     poro = np.array(self.reservoir.mesh.poro)
    #     self.reservoir.conduction[poro <= poro_shale_threshold] = self.idata.rock.conduction_shale
    #     self.reservoir.conduction[poro > poro_shale_threshold] = self.idata.rock.conduction_sand
    #     self.reservoir.hcap[poro <= poro_shale_threshold] = self.idata.rock.hcap_shale
    #     self.reservoir.hcap[poro > poro_shale_threshold] = self.idata.rock.hcap_sand
    #
    #     # add hcap and rcond to be saved into mesh.vtu
    #     l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
    #     g2l = np.array(self.reservoir.discr_mesh.global_to_local, copy=False)
    #     self.reservoir.global_data.update({'heat_capacity': make_full_cube(self.reservoir.hcap, l2g, g2l),
    #                                        'rock_conduction': make_full_cube(self.reservoir.conduction, l2g, g2l)})
    #
    #     self.set_physics()
    #
    #     # time stepping and convergence parameters
    #     sim = self.idata.sim  # short name
    #     self.set_sim_params(first_ts=sim.first_ts, mult_ts=sim.mult_ts, max_ts=sim.max_ts, runtime=sim.runtime,
    #                         tol_newton=sim.tol_newton, tol_linear=sim.tol_linear)
    #     if hasattr(sim, 'linear_type'):
    #         self.params.linear_type = sim.linear_type
    #
    #     self.timer.node["initialization"].stop()

    def init_tracked_cells(self):

        self.ijk_to_track = []

        self.tracked_cells = []

        for (i, j, k) in self.ijk_to_track:
            global_idx = self.reservoir.discr_mesh.get_global_index(i - 1, j - 1, k - 1)
            local_idx = self.reservoir.discr_mesh.global_to_local[global_idx]
            if local_idx >= 0:
                self.tracked_cells.append((i, j, k, int(local_idx)))  # Make sure it's an int!
                print(f"[Tracking] IJK=({i},{j},{k}) => local ID = {local_idx}")
            else:
                print(f"[Tracking] Skipping inactive cell at IJK=({i},{j},{k})")

        print("[Tracker] Found tracked cells:", self.tracked_cells)
        self.my_tracked_pressures = {
            f'P_I{i}_J{j}_K{k}': [] for (i, j, k, _) in self.tracked_cells
        }
        self.my_average_pressure = []

        self.initial_pressure = None
        self.max_pressure_threshold = None
        self.max_fracture_threshold = None

    def set_wells(self):
        # read perforation data from a file
        if hasattr(self.idata, 'schfile'):
            # apply to the reservoir; add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for perf_tuple in wdata.perforations:
                    perf = perf_tuple[1]
                    # adjust to account for added overburden layers
                    perf_ijk_new = (perf.loc_ijk[0], perf.loc_ijk[1], perf.loc_ijk[2] + self.idata.geom.burden_layers)
                    self.reservoir.add_perforation(wname,
                                                   cell_index=perf_ijk_new,
                                                   well_index=perf.well_index, well_indexD=0,
                                                   multi_segment=True, verbose=True)
        else:
            # add wells and perforations, 1-based indices
            for wname, wdata in self.idata.well_data.wells.items():
                self.reservoir.add_well(wname)
                for k in range(1 + self.idata.geom.burden_layers,  self.reservoir.nz+1-self.idata.geom.burden_layers):
                    self.reservoir.add_perforation(wname,
                                                   cell_index=(wdata.location.I, wdata.location.J, k),
                                                   well_index=None, multi_segment=False, verbose=True)

    def set_initial_pressure_from_file(self, fname : str):
        # set initial pressure
        p_cpp = value_vector()
        load_single_float_keyword(p_cpp, fname, 'PRESSURE', -1)
        p_file = np.array(p_cpp, copy=False)
        p_mesh = np.array(self.reservoir.mesh.pressure, copy=False)
        try:
            actnum = np.array(self.reservoir.actnum, copy=False) # CPG Reservoir
        except:
            actnum = self.reservoir.global_data['actnum']  #Struct reservoir
        p_mesh[:self.reservoir.mesh.n_res_blocks * 2] = p_file[actnum > 0]

    def well_is_inj(self, wname : str):  # determine well control by its name
        return "INJ" in wname

    def do_after_step(self):
        self.physics.engine.report()

        # Dimensions
        nx, ny, nz = map(int, self.reservoir.dims)

        # --- Extract active cell pressure values ---
        l2g = np.array(self.reservoir.discr_mesh.local_to_global, copy=False)
        active_mask = l2g >= 0
        active_l2g = l2g[active_mask]

        pressure = np.array(self.physics.engine.X[::self.physics.n_vars], copy=False)
        pressure_active = pressure[:active_mask.sum()]

        average_pressure = np.mean(pressure_active)
        self.my_tracked_pressures.setdefault("P_MEAN_Reservoir", []).append(average_pressure)

        # --- Initialize pressure threshold on first step ---
        if self.initial_pressure is None:
            self.initial_pressure = pressure_active.copy()
            self.max_pressure_threshold = self.initial_pressure + 0.5  # 0.5 bar threshold

        if self.max_fracture_threshold is None:
            depth = np.array(self.reservoir.mesh.depth, copy=False)
            depth_active = depth[:active_mask.sum()]  # ✅ same shape as pressure_active
            self.max_fracture_threshold = depth_active * 139 / 1000 + 1

        fracture_exceed_mask = pressure_active > self.max_fracture_threshold
        # --- Manually exclude all cells where I=90 and J=95 (0-based: i=89, j=94) ---
        for idx, g in enumerate(active_l2g):
            k = g // (nx * ny)
            j = (g % (nx * ny)) // nx
            i = g % nx
            if i == 89 and j == 94:
                fracture_exceed_mask[idx] = False

        # --- Exceedance ---
        exceed_mask = pressure_active > self.max_pressure_threshold
        exceed_count = int(np.sum(exceed_mask))
        self.my_tracked_pressures.setdefault("PRESSURE_EXCEED_COUNT", []).append(exceed_count)

        # Debug: print exceeding cell indices (I, J, K)
        gidx_exceed = active_l2g[exceed_mask]
        if exceed_count > 0:
            # print(f"[Exceedance] {exceed_count} cells exceed threshold")
            for g in gidx_exceed:
                k = g // (nx * ny)
                j = (g % (nx * ny)) // nx
                i = g % nx
                # print(f"  - I={i + 1}, J={j + 1}, K={k + 1}")  # 1-based indexing

        # --- Pressure plume area (km²) and unique (I,J) count ---
        ij_exceed = {(g % nx, (g // nx) % ny) for g in gidx_exceed}
        unique_ij_count = len(ij_exceed)
        dx = 25
        area_km2 = unique_ij_count * dx * dx / 1e6
        self.my_tracked_pressures.setdefault("PRESSURE_PLUME_AREA_KM2", []).append(area_km2)
        self.my_tracked_pressures.setdefault("PRESSURE_EXCEED_SURFACE_CELLS", []).append(unique_ij_count)
        # print(f"[Exceedance] Unique (I, J) surface cells exceeding threshold: {unique_ij_count}")

        # --- Additional: Fracture pressure exceedance analysis ---

        # Apply fracture threshold only to active cells
        fracture_exceed_count = int(np.sum(fracture_exceed_mask))

        self.my_tracked_pressures.setdefault("FRACTURE_EXCEED_COUNT", []).append(fracture_exceed_count)

        # Surface coverage (I, J) for fracture exceedance
        gidx_fracture_exceed = active_l2g[fracture_exceed_mask]
        ij_fracture_exceed = {(g % nx, (g // nx) % ny) for g in gidx_fracture_exceed}
        fracture_ij_count = len(ij_fracture_exceed)
        fracture_area_km2 = fracture_ij_count * dx * dx / 1e6

        self.my_tracked_pressures.setdefault("FRACTURE_EXCEED_SURFACE_CELLS", []).append(fracture_ij_count)
        self.my_tracked_pressures.setdefault("FRACTURE_PLUME_AREA_KM2", []).append(fracture_area_km2)

        # --- Additional: dP stats for cells exceeding +0.5 bar threshold ---
        if self.initial_pressure is not None:
            dp_exceed = pressure_active[exceed_mask] - self.initial_pressure[exceed_mask]
            if dp_exceed.size > 0:
                avg_dp_exceed = float(np.mean(dp_exceed))
                var_dp_exceed = float(np.var(dp_exceed))
            else:
                avg_dp_exceed = 0.0
                var_dp_exceed = 0.0
        else:
            avg_dp_exceed = 0.0
            var_dp_exceed = 0.0

        self.my_tracked_pressures.setdefault("DP_EXCEED_MEAN", []).append(avg_dp_exceed)
        self.my_tracked_pressures.setdefault("DP_EXCEED_VAR", []).append(var_dp_exceed)

        # --- Fault ΔP analysis ---
        from collections import defaultdict
        fault_deltas = defaultdict(list)
        for (i1, j1, k1), (i2, j2, k2) in self.reservoir.fault_connections_ijk:
            idx1 = i1 + nx * j1 + nx * ny * k1
            idx2 = i2 + nx * j2 + nx * ny * k2
            idx1_local = self.reservoir.discr_mesh.global_to_local[idx1]
            idx2_local = self.reservoir.discr_mesh.global_to_local[idx2]

            delta = abs(pressure[idx1_local] - pressure[idx2_local])
            fault_name = self.reservoir.fault_connection_to_name.get(((i1, j1, k1), (i2, j2, k2)), "UNNAMED")
            fault_deltas[fault_name].append(delta)

        all_deltas = [dp for deltas in fault_deltas.values() for dp in deltas]
        filtered_deltas = []
        if all_deltas:
            max_physical_dp = 10.0  # bar, adjust to model reality
            p1, p99 = np.percentile(all_deltas, [1, 99])
            filtered_deltas = [dp for dp in all_deltas if p1 <= dp <= p99 and dp <= max_physical_dp]
            var_dp = np.var(filtered_deltas) if filtered_deltas else 0.0
        else:
            var_dp = 0.0

        self.my_tracked_pressures.setdefault("DP_FAULT_VAR", []).append(var_dp)
        self.my_tracked_pressures.setdefault("DP_FAULT_MAX", []).append(
            np.max(filtered_deltas) if filtered_deltas else 0.0)

        for fault_name, deltas in fault_deltas.items():
            self.my_tracked_pressures.setdefault(f"DP_FAULT_{fault_name}", []).append(
                np.mean(deltas) if deltas else 0.0)
            self.my_tracked_pressures.setdefault(f"DP_FAULT_{fault_name}_MAX", []).append(
                np.max(deltas) if deltas else 0.0)

        # --- Tracked cell logging ---
        for (i, j, k, local_idx) in self.tracked_cells:
            key_p = f'P_I{i}_J{j}_K{k}'
            key_t = f'T_I{i}_J{j}_K{k}'
            state_idx = local_idx * self.physics.n_vars
            try:
                p = self.physics.engine.X[state_idx]
                T = self.physics.engine.X[state_idx + 2]
                self.my_tracked_pressures.setdefault(key_p, []).append(p)
                self.my_tracked_pressures.setdefault(key_t, []).append(T)
            except Exception as e:
                print(f"[Tracker Error] {key_p}/{key_t}: {e}")

        self.print_well_rate()

    def output_properties(self, output_properties, timestep):
        # overload to add additional arrays (geomechanical proxy results) to vtk output
        output_properties_2 = ['pressure'] + output_properties
        tsteps, props = super().output_properties(output_properties=output_properties_2, timestep=timestep)
        if hasattr(self, 'out'):
            props.update(self.out)
        return tsteps, props



class CustomCPGReservoir(CPG_Reservoir):
    def apply_fault_mult(self, faultfile, cell_m, cell_p, mpfa_tran, ids):
        print("[FaultMult] Building fast lookup map...")
        conn_map = {(cell_m[idx], cell_p[idx]): idx for idx in range(len(ids))}

        reading_editnnc = False
        applied = 0
        skipped = 0
        current_fault_label = "UNNAMED"

        # Store fault connections
        self.fault_connections_ijk = []
        self.fault_connections_flat = []
        self.fault_connections_mult = []
        self.fault_connection_to_name = {}

        nx, ny, nz = map(int, self.dims)
        print("[FaultMult] Reading fault file...")

        with open(faultfile) as f:
            for buff in f:
                line = buff.strip()
                if not line:
                    continue

                # Detect fault name from comment lines
                if line.startswith('--'):
                    if "FAULT" in line.upper():
                        current_fault_label = line.strip("- ").upper()
                    continue

                # Toggle based on keywords
                if line.startswith(('MULTIPLY', 'ENDFLUXNUM', 'REGIONS', 'EQLDIMS', 'BOX', 'ENDBOX', 'COPY')):
                    reading_editnnc = False

                if line.startswith('EDITNNC'):
                    reading_editnnc = True
                    continue

                # Parse EDITNNC connections
                if reading_editnnc:
                    if line.endswith('/'):
                        line = line[:-1].strip()

                    parts = line.split()
                    if len(parts) < 7:
                        continue

                    try:
                        i1, j1, k1 = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
                        i2, j2, k2 = int(parts[3]) - 1, int(parts[4]) - 1, int(parts[5]) - 1
                        mult = float(parts[6])
                    except ValueError:
                        skipped += 1
                        continue

                    # Skip negative indices
                    if any(x < 0 for x in [i1, j1, k1, i2, j2, k2]):
                        skipped += 1
                        continue

                    # Skip indices outside grid
                    if any([
                        i1 >= nx, j1 >= ny, k1 >= nz,
                        i2 >= nx, j2 >= ny, k2 >= nz
                    ]):
                        skipped += 1
                        continue

                    # Local connection lookup
                    m_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i1, j1, k1)]
                    p_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i2, j2, k2)]

                    key = (m_idx, p_idx)
                    if key in conn_map:
                        idx = conn_map[key]
                        mpfa_tran[2 * ids[idx]] *= mult
                        applied += 1

                        # Save fault info
                        self.fault_connections_ijk.append(((i1, j1, k1), (i2, j2, k2)))
                        idx1 = i1 + nx * j1 + nx * ny * k1
                        idx2 = i2 + nx * j2 + nx * ny * k2
                        self.fault_connections_flat.append((idx1, idx2))
                        self.fault_connections_mult.append(mult)
                        self.fault_connection_to_name[((i1, j1, k1), (i2, j2, k2))] = current_fault_label
                    else:
                        skipped += 1

        print(f"[FaultMult] Done. Applied: {applied}, Skipped: {skipped}")

