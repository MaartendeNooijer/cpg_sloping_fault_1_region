import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os, sys

from darts.engines import redirect_darts_output
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output, abort_redirection

from model_co2_1_region import ModelCCS

import os
from darts.tools.plot_darts import *
from darts.tools.logging import redirect_all_output

from darts.engines import redirect_darts_output
import pandas as pd
import re
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from darts_model_1_region import DartsModel
import darts.physics.super.operator_evaluator
from darts.physics.super.property_container import PropertyContainer
from darts.tools.hdf5_tools import load_hdf5_to_dict


def extract_elapsed_time(log_path, output_path):
    """
    Extracts the last occurrence of ELAPSED time from a log file and writes it to a text file.
    """
    elapsed_time = None
    try:
        with open(log_path, "r") as file:
            lines = file.readlines()

        for line in reversed(lines):
            if "ELAPSED" in line:
                parts = line.split("ELAPSED")
                if len(parts) > 1:
                    time_part = parts[1].strip().strip("()")
                    if time_part.count(":") == 2:
                        elapsed_time = time_part
                        break

        if elapsed_time:
            with open(output_path, "w") as out:
                out.write(elapsed_time)
            print(f"Elapsed time '{elapsed_time}' written to {output_path}")
        else:
            print("No elapsed time found in the log file.")

    except Exception as e:
        print(f"Error: {e}")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # Ensure real-time output

    def flush(self):
        for s in self.streams:
            s.flush()

def run_simulation_main(self, out_dir="."):
    time_simulated = 0.0
    self.do_after_step()
    for ith_step, dt in enumerate(self.idata.sim.time_steps):
        self.set_well_controls(time=time_simulated)
        success = run_main(self, days=dt, out_dir=out_dir)
        if not success:
            print(f"[FAIL] Simulation step {ith_step} with dt={dt} failed or aborted.")
            return False
        self.do_after_step()
        time_simulated += dt
    return True

def run_main(self, days: float = None, restart_dt: float = 0., save_well_data: bool = True, save_solution_data: bool = True,
        log_3d_body_path: bool = False, verbose: bool = True, out_dir: str = "."):
    """
    Method to run simulation for specified time with optional wall-time progress monitoring.
    """
    import time
    import os
    import numpy as np
    from math import fabs

    days = days if days is not None else self.runtime
    t = self.physics.engine.t
    stop_time = t + days
    ts = 0

    # Wall-time monitoring setup
    time_start = time.time()
    progress_thresholds = [
        (2.5 * 3600, 3650),  # After 2.5 hours, expect ≥10 years
        (5 * 3600, 7300)     # After 5 hours, expect ≥20 years
    ]

    # Determine first timestep
    if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
        dt = self.params.first_ts
    elif restart_dt > 0.:
        dt = restart_dt
    else:
        dt = min(self.prev_dt * self.params.mult_ts, days, self.params.max_ts)
    self.prev_dt = dt

    if log_3d_body_path:
        self.physics.body_path_start(output_folder=self.output_folder)

    while t < stop_time:
        # Time monitoring
        wall_time = time.time() - time_start
        print(f"[TimeCheck] Step {ts}, Walltime = {wall_time:.1f}s, Simulated = {t:.1f} days")

        for max_wall_sec, min_sim_days in progress_thresholds:
            if wall_time > max_wall_sec and t < min_sim_days:
                reason = (f"[STOP] Simulation too slow: {wall_time:.1f}s elapsed but only {t:.1f} days simulated.\n"
                          f"Expected ≥ {min_sim_days} days after {max_wall_sec / 60:.1f} min.")
                print(reason)
                with open(os.path.join(out_dir, "no_convergence_info.txt"), "w") as f:
                    f.write(reason)
                    f.write(f"\nStopped at timestep {ts}, dt={dt}\n")
                return -1

        # Main timestep logic
        converged = self.run_timestep(dt, t, verbose)

        if converged:
            t += dt
            self.physics.engine.t = t
            ts += 1
            if verbose:
                print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI = %d"
                      % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

            dt = min(dt * self.params.mult_ts, self.params.max_ts)

            if np.fabs(t + dt - stop_time) < self.params.min_ts:
                dt = stop_time - t
            elif t + dt > stop_time:
                dt = stop_time - t
            else:
                self.prev_dt = dt

            if log_3d_body_path:
                self.physics.body_path_add_bodys(output_folder=self.output_folder, time=t)

            if save_well_data:
                self.save_data_to_h5(kind='well')

        else:
            dt /= self.params.mult_ts
            if verbose:
                print("Cut timestep to %2.10f" % dt)
            if dt < self.params.min_ts:
                print('Stop simulation. Reason: reached min. timestep', self.params.min_ts, 'dt =', dt)
                return False

    self.physics.engine.t = stop_time

    if save_solution_data:
        self.save_data_to_h5(kind='solution')

    if verbose:
        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
              % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                 self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                 self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))

    return True

def run(physics_type : str, case: str, out_dir: str, export_vtk=True, redirect_log=False, platform='cpu'):
    '''
    :param physics_type: "geothermal" or "dead_oil"
    :param case: input grid name
    :param out_dir: directory name for outpult files
    :param export_vtk:
    :return:
    '''
    print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform)

    if platform == 'gpu':
        from darts.engines import set_gpu_device
        device = int(os.getenv("DEVICE", 0))
        set_gpu_device(device)  # Ensure the GPU Device Index
        print('Test started', 'physics_type:', physics_type, 'case:', case, 'platform=', platform, 'GPU device=',
              device)
    import os
    os.makedirs(out_dir, exist_ok=True)

    redirect_log = True

    if redirect_log:
        log_path = os.path.join(out_dir, "run_n.log")
        redirect_darts_output(log_path)

    m = ModelCCS()

    m.physics_type = physics_type

    # Save init output (set_input_data + init_reservoir) to a dedicated log file
    init_log_path = os.path.join(out_dir, "init_reservoir_output.log")
    with open(init_log_path, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            m.set_input_data(case=case)
            m.init_reservoir()
            m.init(output_folder=out_dir, platform=platform)  # NEW

    m.save_data_to_h5(kind = 'solution')
    m.set_well_controls()

    standard_print_path = os.path.join(out_dir, "run_n_standard_print.log")
    with open(standard_print_path, "w") as std_log_file:
        tee = Tee(sys.stdout, std_log_file)
        with redirect_stdout(tee):
            completed = run_simulation_main(m, out_dir=out_dir)
    if not completed:
        print("[Watch Out] Simulation stopped early. Skipping output writing.")
        return None, None, None, None, None

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def output_to_satV(self, output_directory, satV=[], threshold=0.05):
        cell_area_m2 = 25 * 25  # m²
        props_names = ['satV']
        nx, ny, nz = map(int, self.reservoir.dims)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # COORD array → reshape pillars
        COORD = self.reservoir.arrays['COORD']
        coord = np.array(COORD).reshape(-1, 6)

        # Manual well location (1-based), convert to 0-based
        i_well, j_well = 14, 14 #90 - 1, 95 - 1  # 14, 14 #90 - 1, 95 - 1 #WATCH OUT, CHANGE!
        x_well, y_well = get_cell_center(i_well, j_well, coord, nx)

        # Top-layer indices
        top_layer_indices = np.arange(nx * ny)
        i_coords = top_layer_indices % nx
        j_coords = top_layer_indices // nx

        # Get (x, y) for each top-layer cell
        x_centers, y_centers = [], []
        for i, j in zip(i_coords, j_coords):
            x, y = get_cell_center(i, j, coord, nx)
            x_centers.append(x)
            y_centers.append(y)

        x_centers = np.array(x_centers)
        y_centers = np.array(y_centers)

        def classify_direction(dx, dy):
            angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
            return directions[int(((angle + 22.5) % 360) // 45)]

        # Assign directions to top-layer cells
        directions = [
            classify_direction((x - x_well), -(y - y_well))
            for x, y in zip(x_centers, y_centers)
        ]

        directional_plume = []

        for ith_step in range(1, len(self.idata.sim.time_steps) + 1):
            _, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)
            satV_top = property_array['satV'][0][top_layer_indices]

            dir_to_max_dist = {d: 0.0 for d in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}

            for idx, s in enumerate(satV_top):
                if s < threshold:
                    continue
                dx = x_centers[idx] - x_well
                dy = y_centers[idx] - y_well
                dist = np.sqrt(dx ** 2 + dy ** 2)
                direction = directions[idx]
                if dist > dir_to_max_dist[direction]:
                    dir_to_max_dist[direction] = dist

            # Compute plume surface area
            plume_cell_count = np.sum(satV_top >= threshold)
            plume_surface_m2 = plume_cell_count * cell_area_m2
            dir_to_max_dist['PlumeSurface_m2'] = plume_surface_m2

            directional_plume.append(dir_to_max_dist)

        # Save outputs
        save_directional_plume_to_csv(directional_plume, output_directory)
        save_final_rose_diagram(directional_plume, output_directory)

        return directional_plume

    def get_cell_center(i, j, coord, nx):
        p0 = i + (nx + 1) * j
        p1 = i + 1 + (nx + 1) * j
        p2 = i + (nx + 1) * (j + 1)
        p3 = i + 1 + (nx + 1) * (j + 1)
        x = np.mean([coord[p, 0] for p in [p0, p1, p2, p3]])
        y = np.mean([coord[p, 1] for p in [p0, p1, p2, p3]])
        return x, y

    def save_directional_plume_to_csv(directional_plume, output_directory, filename='plume_migration.csv'):
        df = pd.DataFrame(directional_plume)
        df.index.name = 'Timestep'
        df['MaxDistance'] = df.max(axis=1)
        df['MaxDirection'] = df.idxmax(axis=1)
        full_path = os.path.join(output_directory, filename)
        df.to_csv(full_path)
        print(f"[Output] Saved CSV to: {full_path}")

    def save_final_rose_diagram(plume_data, output_directory, filename='plume_final_rose.png'):
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        angles = np.deg2rad(np.arange(0, 360, 45))
        distances = [plume_data[-1][d] for d in directions]
        distances.append(distances[0])
        angles = np.append(angles, angles[0])

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, distances, marker='o')
        ax.fill(angles, distances, alpha=0.3)
        ax.set_title('Final Plume Spread')
        ax.set_theta_zero_location('N')  # 0° at North (top)
        ax.set_theta_direction(1)  # counterclockwise rotation

        full_path = os.path.join(output_directory, filename)
        plt.savefig(full_path)
        plt.close()
        print(f"[Output] Saved rose diagram to: {full_path}")

    directional_plume = output_to_satV(m, output_directory=out_dir, satV=[], threshold=0.05)

    def add_columns_time_data(time_data):
        time_data['Time (years)'] = time_data['time'] / 365.25
        molar_mass_co2 = 44.01  # kg/kmol
        molar_density = 19.7239  # kmol/m3 at 161 bar, 300K
        for k in time_data.keys():
            if 'temperature' in k:
                time_data[k.replace('K', 'degrees')] = time_data[k] - 273.15
                time_data.drop(columns=k, inplace=True)
            if physics_type == 'ccs' and 'INJ : V rate (m3/day)' in k:
                time_data[k.replace('INJ : V rate (m3/day)', 'INJ : V rate (Ton/Day)')] = time_data[k] * molar_mass_co2 / 1000  # kmol/day * kg/kmol / 1000 = ton/day
            if physics_type == 'ccs' and 'V  volume (m3)' in k:
                time_data[k.replace('INJ : V  volume (m3)', 'V volume (Mt/year)')] = time_data[k] * molar_mass_co2 / 1000 / 1e6  # m3/year * kmol/m3 * kg/kmol / 1000 = ton/year / 1e6 = Mt/year

    time_data = pd.DataFrame.from_dict(m.physics.engine.time_data)
    add_columns_time_data(time_data)
    time_data.to_pickle(os.path.join(out_dir, 'time_data.pkl'))

    time_data_report = pd.DataFrame.from_dict(m.physics.engine.time_data_report)
    add_columns_time_data(time_data_report)
    time_data_report.to_pickle(os.path.join(out_dir, 'time_data_report.pkl'))

    for key, values in m.my_tracked_pressures.items():
        if len(values) == len(time_data_report):
            time_data_report[key] = values
        elif len(values) == len(time_data_report) + 1:
            # First value corresponds to t=0
            time_data_report[f"{key}_0"] = values[0]  # Store as a separate column
            time_data_report[key] = values[1:]  # Assign the rest as usual
        else:
            print(f"[Tracker Warning] Skipping {key}: unexpected length {len(values)} vs {len(time_data_report)}")

    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data.xlsx'))
    time_data.to_excel(writer, sheet_name='time_data')
    writer.close()

    # filter time_data_report and write to xlsx
    # list the column names that should be removed
    press_gridcells = time_data_report.filter(like='reservoir').columns.tolist()
    chem_cols = time_data_report.filter(like='Kmol').columns.tolist()
    # remove columns from data
    time_data_report.drop(columns=press_gridcells + chem_cols, inplace=True)
    # add time in years
    time_data_report['Time (years)'] = time_data_report['time'] / 365.25
    writer = pd.ExcelWriter(os.path.join(out_dir, 'time_data_report.xlsx'))
    time_data_report.to_excel(writer, sheet_name='time_data_report')
    writer.close()

    m.reservoir.centers_to_vtk(out_dir)

    m.print_timers()

    timing_info = m.timer.print("", "")

    print(timing_info)

    save_combined_timing_info(
        log_path=log_path,
        timing_info_str=timing_info,
        output_path=os.path.join(out_dir, "elapsed_time.txt")
    )

    if export_vtk:
        # read h5 file and write vtk
        m.reservoir.create_vtk_wells(output_directory=out_dir)
        steps_to_export = [0, 2, len(m.idata.sim.time_steps)]
        for ith_step in steps_to_export:
            m.output_to_vtk(ith_step=ith_step)

    # failed, sim_time = check_performance_local(m=m, case=case, physics_type=physics_type)
    #
    # print('Failed' if failed else 'Ok')

    return time_data, time_data_report, m.idata.well_data.wells.keys(), m.well_is_inj

##########################################################################################################

def plot_results(wells, well_is_inj, time_data_list, time_data_report_list,
                 label_list, physics_type, out_dir, case):  # ✅ Add case here

    plt.rc('font', size=12)

    def plot_total_inj_gas_rate_darts_volume(
            darts_df,
            style='-',
            color='#00A6D6',
            ax=None,
            alpha=1,
            out_dir=None,
            case_name="unknown"
    ):
        from scipy.integrate import trapezoid
        import pandas as pd
        import numpy as np
        import os

        acc_df = darts_df.copy()

        # ✅ Step 1: Detect when injection starts (INJ1 rate > 1)
        #inj_cols = [col for col in acc_df.columns if "INJ" in col and "V rate (ton/day)" in col]
        inj_cols = [col for col in acc_df.columns if "INJ : V rate (Ton/Day)" in col]
        first_injection_idx = None
        for col in inj_cols:
            first_above_one = acc_df[acc_df[col] > 1.0]
            if not first_above_one.empty:
                idx = first_above_one.index.min()
                if first_injection_idx is None or idx < first_injection_idx:
                    first_injection_idx = idx

        # ✅ Step 2: Slice the DataFrame starting from first real injection
        acc_df = acc_df.loc[first_injection_idx:].copy()

        # ✅ Step 3: Skip early time noise again if there's a new reset to zero
        if acc_df['time'].iloc[0] < 0.5:
            acc_df = acc_df[acc_df["time"] > 1.0]

        # ✅ Step 4: Sum all injection rates
        acc_df['total'] = 0.0
        for col in inj_cols:
            acc_df['total'] += acc_df[col]

        # ✅ Step 5: Plot (convert time to years for display)
        ax = ax or plt.gca()
        ax.plot(acc_df['time'] / 365.25, acc_df['total'], style, color=color, alpha=alpha, label="total")
        ax.set_ylabel("Inj Gas Rate [Ton/Day]")
        ax.set_xlabel("Years")

        # ✅ Step 6: Integration
        total_mass_tons = trapezoid(acc_df['total'], acc_df['time'])
        total_mass_kg = total_mass_tons * 1000
        total_mass_mt = total_mass_kg / 1e9
        density = 880
        injected_volume_m3 = total_mass_kg / density

        annotation = f"Injected: {injected_volume_m3:,.0f} m³ CO₂"
        summary_line = f"CASE: {case_name or 'unknown'}\n"
        summary_line += f"  Total Injected Mass: {total_mass_mt:.3f} Mt\n"
        summary_line += f"  Total Injected Volume: {injected_volume_m3:,.0f} m³\n"

        # ✅ Step 8: Annotate on plot
        ax.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        # ✅ Step 9: Save summary
        if out_dir is not None:
            summary_file = os.path.join(out_dir, 'injection_summary.txt')
            with open(summary_file, 'a', encoding='utf-8') as f:
                f.write(summary_line + "\n")

        # ✅ Step 10: Adjust Y axis
        ymin, ymax = ax.get_ylim()
        if ymax < 0:
            ax.set_ylim(ymin * 1.1, 0)
        if ymin > 0:
            ax.set_ylim(0, ymax * 1.1)

        return ax, {
            "total_mass_tons": total_mass_tons,
            "total_mass_mt": total_mass_mt,
            "total_volume_m3": injected_volume_m3,
            "summary_text": summary_line
        }

    for well_name in wells:

        ax = None
        for time_data, label in zip(time_data_list, label_list):
            # ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax, total_poro_volume=total_poro_volume)
            ax, _ = plot_total_inj_gas_rate_darts_volume(
                time_data,
                ax=ax,
                out_dir=out_dir,
                case_name=case
            )
            # ax = plot_total_inj_gas_rate_darts_volume(time_data, ax=ax)  # , label=label) #NEW was ax = plot_total_inj_gas_rate_darts(time_data, ax=ax)
        ax.set(xlabel="Years", ylabel="Inj Gas Rate [Ton/Day]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'total_inj_gas_rate_' + well_name + '_' + case + '.png'))
        plt.close()

    for well_name in wells:
        ax = None
        for time_data_report, label in zip(time_data_report_list, label_list):
            ax = plot_bhp_darts(well_name, time_data_report, ax=ax)  # , label=label)
        ax.set(xlabel="Days", ylabel="BHP [bar]")
        plt.savefig(os.path.join(out_dir, 'well_' + well_name + '_bhp_' + case + '.png'))
        plt.tight_layout()
        plt.close()


def parse_elapsed_time_txt(path):
    timing_data = {}
    try:
        with open(path, "r") as f:
            for line in f:
                match = re.match(r"\s*([\w\s<>]+)\s+([\d\.]+)\s+sec", line)
                if match:
                    label = match.group(1).strip().lower().replace(" ", "_")
                    timing_data[label] = float(match.group(2))

        # Handle broken initialization values
        if "initialization" in timing_data and timing_data["initialization"] > 86400:  # > 1 day
            total = timing_data.get("total_elapsed", 0)
            sim = timing_data.get("simulation", 0)
            newton = timing_data.get("newton_update", 0)
            vtk = timing_data.get("vtk_output", 0)
            recomputed_init = total - sim - newton - vtk
            timing_data["initialization"] = max(recomputed_init, 0)
            print(f"⚠️ Recomputed initialization time: {timing_data['initialization']:.2f} sec")

    except Exception as e:
        print(f"  ❌ Failed to read {path}: {e}")
    return timing_data


def plot_convergence_metrics_from_log(log_file_path, output_dir, case_name):
    """
    Parses and plots convergence metrics from a DARTS log file,
    using computed DT instead of logged DT.
    """
    patterns = {
        "T": r"T\s*=\s*([\d\.eE+-]+)",
        "DT": r"DT\s*=\s*([\d\.eE+-]+)",
        "NI": r"NI\s*=\s*(\d+)",
        "LI": r"LI\s*=\s*(\d+)",
        "RES": r"RES\s*=\s*([\d\.eE+-]+)",
    }

    components = [
        "initialization",
        "simulation",
        "jacobian_assembly",
        "linear_solver_solve"
    ]

    data = {key: [] for key in patterns}
    computed_dt = []
    elapsed_time = None
    timing_breakdown = None

    try:
        with open(log_file_path, "r") as f:
            prev_t = None
            for line in f:
                if "T =" in line and "CFL=" in line:
                    for key, pattern in patterns.items():
                        match = re.search(pattern, line)
                        if match:
                            val = float(match.group(1))
                            data[key].append(val)

                    # Compute DT manually from T
                    t_now = data["T"][-1]
                    if prev_t is not None:
                        computed_dt.append(t_now - prev_t)
                    else:
                        computed_dt.append(data["DT"][-1] if data["DT"] else 0.0)
                    prev_t = t_now

                if "ELAPSED" in line:
                    match = re.search(r"ELAPSED\s+(\d+):(\d+):(\d+)", line)
                    if match:
                        h, m, s = map(int, match.groups())
                        elapsed_time = h * 3600 + m * 60 + s

        # Try to load timing breakdown from elapsed_time.txt
        elapsed_txt_path = os.path.join(os.path.dirname(log_file_path), "elapsed_time.txt")
        if os.path.exists(elapsed_txt_path):
            timing_breakdown = parse_elapsed_time_txt(elapsed_txt_path)

        if not data["T"]:
            print("⚠️ No convergence data found in log file.")
            return

        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        plt.plot(data["T"], data["NI"], label="Newton Iterations")
        plt.xlabel("Time")
        plt.ylabel("NI")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(data["T"], data["LI"], label="Linear Iterations", color="orange")
        plt.xlabel("Time")
        plt.ylabel("LI")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(data["T"], data["RES"], label="Residual", color="green")
        plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("RES")
        plt.grid(True)

        plt.subplot(3, 2, 4)
        li_ni_ratio = [li / ni if ni != 0 else 0 for li, ni in zip(data["LI"], data["NI"])]
        plt.plot(data["T"], li_ni_ratio, label="LI / NI", color="red")
        plt.xlabel("Time")
        plt.ylabel("LI / NI")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(data["T"], computed_dt, label="Computed DT", color="purple")
        plt.xlabel("Time")
        plt.ylabel("DT")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        if timing_breakdown:
            x = [0]
            bottom = 0
            for comp in components:
                val = timing_breakdown.get(comp, 0)
                plt.bar(x, [val], bottom=bottom, label=comp.replace("_", " ").title())
                bottom += val
            plt.xticks([0], ["Elapsed Time"])
            plt.ylabel("Time (sec)")
            plt.title("Component-wise Elapsed Time")
            plt.legend()
        elif elapsed_time:
            elapsed_minutes = elapsed_time / 60
            plt.bar(["Elapsed Time"], [elapsed_minutes * 60], color="steelblue")
            plt.ylabel("Seconds")
            plt.title("Final Elapsed Time")
            plt.text(0, elapsed_minutes * 60, f"{elapsed_minutes:.1f} min", ha='center', va='bottom')
        else:
            plt.text(0.5, 0.5, "No ELAPSED found", ha='center', va='center')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(f"Convergence Metrics – {case_name}", fontsize=18)

        fig_path = os.path.join(output_dir, f"convergence_metrics_{case_name}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"✅ Convergence plot saved to {fig_path}")

    except Exception as e:
        print(f"❌ Error parsing log file: {e}")


def save_combined_timing_info(log_path, timing_info_str, output_path):
    """
    Combines wall-clock elapsed time from run_n.log and detailed DARTS timing info
    into a single file at output_path.
    """
    elapsed_time_str = None
    try:
        with open(log_path, "r") as file:
            lines = file.readlines()
        for line in reversed(lines):
            if "ELAPSED" in line:
                parts = line.split("ELAPSED")
                if len(parts) > 1:
                    time_part = parts[1].strip().strip("()")
                    if time_part.count(":") == 2:
                        elapsed_time_str = time_part
                        break
    except Exception as e:
        print(f"❌ Failed to extract wall-clock time: {e}")

    try:
        with open(output_path, "w") as f:
            if elapsed_time_str:
                f.write(f"Wall-clock elapsed time (from log): {elapsed_time_str}\n\n")
            else:
                f.write("Wall-clock elapsed time not found in log.\n\n")

            f.write("Detailed timing breakdown (from simulation):\n")
            f.write(timing_info_str)

        print(f"✅ Combined timing info written to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to write timing info file: {e}")


def extract_and_visualize_combined_rings(model, mesh, output_dir, rings=[], property_name="pressure"):
    import csv
    """
    Extracts values in diamond-shaped rings (Manhattan distance) around each injector,
    and stores all data in a single CSV. Then visualizes it all in one combined plot.
    """
    if property_name == "saturation":
        property_name = "satV"

    discr = model.reservoir.discr_mesh
    get_global = discr.get_global_index
    nx, ny = discr.nx, discr.ny
    centroids = np.array([c.values for c in model.reservoir.centroids_all_cells])

    unique_ij = {(i, j) for (i, j, _) in model.well_cells}
    unique_injectors = [(i, j, 1) for (i, j) in unique_ij]
    stopped_injectors = set()
    all_entries = []

    for r in rings:
        for well_id, (i0, j0, k0) in enumerate(unique_injectors):
            if well_id in stopped_injectors:
                continue

            ring_entries = []
            has_nonzero = False

            # 🔁 Manhattan ring logic — just like original
            for di in range(-r, r + 1):
                dj = r - abs(di)
                dj_signs = [-1, 1] if dj != 0 else [1]  # prevent duplicates
                for dj_sign in dj_signs:
                    i = i0 + di
                    j = j0 + dj_sign * dj
                    k = k0

                    if not (1 <= i <= nx and 1 <= j <= ny):
                        continue

                    try:
                        g_idx = get_global(i - 1, j - 1, k - 1)
                        value = mesh.cell_data[property_name][g_idx]
                        x, y, z = centroids[g_idx]
                        ring_entries.append((well_id, i, j, k, x, y, z, r, value))
                        if value > 0:
                            has_nonzero = True
                    except Exception:
                        continue

            if not has_nonzero:
                print(f"🛑 Injector {well_id}: all values 0 in r={r}. Skipping further rings.")
                stopped_injectors.add(well_id)

            all_entries.extend(ring_entries)

    # Save single combined CSV
    if all_entries:
        out_csv = os.path.join(output_dir, f"diamond_rings_combined_ijk_{property_name}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["well_id", "i", "j", "k", "x", "y", "z", "ring", property_name])
            writer.writerows(all_entries)
        print(f"✅ Combined CSV saved: {out_csv}")

        # Plot
        df = pd.read_csv(out_csv)
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df["x"], df["y"], c=df[property_name], cmap="viridis", s=80)
        plt.colorbar(sc, label=property_name)

        for well_id, group in df.groupby("well_id"):
            inj_x = group[group["i"] == group["i"].median()]["x"].iloc[0]
            inj_y = group[group["j"] == group["j"].median()]["y"].iloc[0]
            plt.scatter(inj_x, inj_y, color="red", s=120, marker="*", label=f"Injector {well_id}")

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.grid(True)
        plotted_rings = sorted(df["ring"].unique())
        plt.title(f"{property_name} in Diamond Rings {plotted_rings} (All Injectors)")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"diamond_rings_all_{property_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Combined plot saved: {plot_path}")
    else:
        print("❌ No data to save or visualize.")


##########################################################################################################
# for CI/CD
def check_performance_local(m, case, physics_type):
    import platform

    os.makedirs('ref', exist_ok=True)

    pkl_suffix = ''
    if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
        pkl_suffix = '_gpu'
    elif os.getenv('ODLS') != None and os.getenv('ODLS') == '-a':
        pkl_suffix = '_iter'
    else:
        pkl_suffix = '_odls'
    print('pkl_suffix=', pkl_suffix)

    file_name = os.path.join('ref', 'perf_' + platform.system().lower()[:3] + pkl_suffix +
                             '_' + case + '_' + physics_type + '.pkl')
    overwrite = 0
    if os.getenv('UPLOAD_PKL') == '1':
        overwrite = 1

    is_plk_exist = os.path.isfile(file_name)

    failed = m.check_performance(perf_file=file_name, overwrite=overwrite, pkl_suffix=pkl_suffix)

    if not is_plk_exist or overwrite == '1':
        m.save_performance_data(file_name=file_name, pkl_suffix=pkl_suffix)
        return False, 0.0

    if is_plk_exist:
        return (failed > 0), -1.0 #data[-1]['simulation time']
    else:
        return False, -1.0

def run_test(args: list = [], platform='cpu'):
    if len(args) > 1:
        case = args[0]
        physics_type = args[1]

        out_dir = 'results_' + physics_type + '_' + case
        ret = run(case=case, physics_type=physics_type, out_dir=out_dir, platform=platform)
        return ret[0], ret[1] #failed_flag, sim_time
    else:
        print('Not enough arguments provided')
        return True, 0.0
##########################################################################################################


if __name__ == '__main__':
    platform = 'gpu' if os.getenv('TEST_GPU') == '1' else 'cpu'

    physics_list = ['ccs']

    # #if platform == 'gpu': #Deactivated for now to make model more userfriendly for inspection
    if platform == 'gpu':
        cases_array = np.load("cases_array.npy", allow_pickle=True)
        case_index = os.getenv("CASE_INDEX")
        case_name = os.getenv("CASE_NAME")

        if case_index is not None:
            import ast

            case_index = ast.literal_eval(case_index)
            if isinstance(case_index, int):
                cases_list = [(case_index, cases_array[case_index])]
            elif isinstance(case_index, list):
                cases_list = [(i, cases_array[i]) for i in case_index]
        elif case_name is not None:
            cases_list = [(i, c) for i, c in enumerate(cases_array) if case_name in c]
        else:
            n_splits = int(os.getenv("SPLITS", 3))
            split_id = int(os.getenv("ID", 0))
            chunks = np.array_split(list(enumerate(cases_array)), n_splits)
            cases_list = chunks[split_id]
    else:
        # CPU fallback
        cases_list = []
        cases_list += ['grid_CCS_maarten']  # very simple heterogeneous case, converges
        # cases_list =+ ['fault=FM1_cut=CO1_grid=G1_top=TS2_mod=OBJ_mult=1'] #This one does converge
        # cases_list =+ ['fault=FM1_cut=CO1_grid=G1_top=TS2_mod=OBJ_mult=2'] #This one does not converge

        cases_list = [(i, case) for i, case in enumerate(cases_list)]

    # ✅ Subprocess mode: Filter ONLY by CASE_IDX
    only_case_idx = os.getenv("CASE_IDX")
    if only_case_idx:
        only_case_idx = int(only_case_idx)
        cases_list = [(i, c) for i, c in cases_list if i == only_case_idx]
        print(f"[INFO] Subprocess mode: filtering to run only CASE_IDX = {only_case_idx}")

    cases_list = []
    cases_list += ['grid_CCS_maarten']
    #cases_list += ['fault=FM1_cut=CO1_grid=G1_top=TS1_mod=OBJ_mult=1']  # This one does converge
    #cases_list += ['fault=FM1_cut=CO1_grid=G1_top=TS2_mod=PIX_mult=2'] #This one does not converge
    cases_list = [(i, case) for i, case in enumerate(cases_list)]

    well_controls = []
    well_controls += ['wbhp']
    #well_controls += ['wrate']

    for physics_type in physics_list:
        for case_idx, case_geom in cases_list:
            for wctrl in well_controls:
                if physics_type == 'deadoil' and wctrl == 'wrate':
                    continue

                tag = f"{int(case_idx):03d}"

                case = case_geom + '_' + wctrl
                folder_name = 'results_' + physics_type + '_' + case + '_' + tag + "1_REGION_5_BAR_MS_TRUE"
                out_dir = os.path.join("results", folder_name)

                time_data, time_data_report, wells, well_is_inj = run(
                    physics_type=physics_type,
                    case=case,
                    out_dir=out_dir,
                    redirect_log=False,
                    export_vtk=True,
                    platform=platform)

                # one can read well results from pkl file to add/change well plots without re-running the model
                pkl1_dir = '..'
                pkl_fname = 'time_data.pkl'
                pkl_report_fname = 'time_data_report.pkl'
                time_data_list = [time_data]
                time_data_report_list = [time_data_report]
                label_list = [None]

                # compare the current results with another run
                # pkl1_dir = r'../../../open-darts_dev/models/cpg_sloping_fault/results_' + physics_type + '_' + case_geom
                # time_data_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_fname))
                # time_data_report_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_report_fname))
                # time_data_list = [time_data_1, time_data]
                # time_data_report_list = [time_data_report_1, time_data_report]
                # label_list = ['1', 'current']

                if time_data is not None:
                    plot_results(wells=wells, well_is_inj=well_is_inj,
                                 time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                                 label_list=label_list,
                                 physics_type=physics_type, out_dir=out_dir,
                                 case=case)  # ✅ Add this

                # plot_results(wells=wells, well_is_inj=well_is_inj,
                #              time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                #              label_list=label_list,
                #              physics_type=physics_type, out_dir=out_dir,
                #              total_poro_volume=total_poro_volume)

                # plot_results(wells=wells, well_is_inj=well_is_inj,
                #              time_data_list=time_data_list, time_data_report_list=time_data_report_list,
                #              label_list=label_list,
                #              physics_type=physics_type, out_dir=out_dir)




# if __name__ == '__main__':
#     platform = 'cpu'
#     if os.getenv('TEST_GPU') != None and os.getenv('TEST_GPU') == '1':
#             platform = 'gpu'
#
#     physics_list = []
#     #physics_list += ['geothermal']
#     #physics_list += ['deadoil']
#     physics_list += ['ccs']
#
#     cases_list = []
#     #cases_list += ['generate_5x3x4']
#     #cases_list += ['generate_51x51x1']
#     #cases_list += ['generate_51x51x1_faultmult']
#     #cases_list += ['generate_100x100x100']
#     #cases_list += ['case_40x40x10']
#     #cases_list += ['brugge']
#     cases_list += ['grid_CCS_maarten']
#
#     well_controls = []
#     #well_controls += ['wrate']
#     #well_controls += ['wbhp']
#     #well_controls += ['wperiodic']
#     well_controls += ['co2_wbhp']
#     #well_controls += ['co2_wrate']
#
#     for physics_type in physics_list:
#         for case_geom in cases_list:
#             for wctrl in well_controls:
#                 if physics_type == 'deadoil' and wctrl == 'wrate':
#                     continue
#                 tag = '_' + 'thermal_ms'
#                 case = case_geom + '_' + wctrl
#                 out_dir = 'results_' + physics_type + '_' + case
#                 failed, sim_time, time_data, time_data_report, wells, well_is_inj = run(physics_type=physics_type,
#                                                                                         case=case, out_dir=out_dir,
#                                                                                         redirect_log=False,
#                                                                                         platform=platform)
#
#                 # one can read well results from pkl file to add/change well plots without re-running the model
#                 pkl1_dir = '.'
#                 pkl_fname = 'time_data.pkl'
#                 pkl_report_fname = 'time_data_report.pkl'
#                 time_data_list = [time_data]
#                 time_data_report_list = [time_data_report]
#                 label_list = [None]
#
#                 # compare the current results with another run
#                 #pkl1_dir = r'../../../open-darts_dev/models/cpg_sloping_fault/results_' + physics_type + '_' + case_geom
#                 #time_data_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_fname))
#                 #time_data_report_1 = pd.read_pickle(os.path.join(pkl1_dir, pkl_report_fname))
#                 #time_data_list = [time_data_1, time_data]
#                 #time_data_report_list = [time_data_report_1, time_data_report]
#                 #label_list = ['1', 'current']
#
#                 plot_results(wells=wells, well_is_inj=well_is_inj,
#                              time_data_list=time_data_list, time_data_report_list=time_data_report_list, label_list=label_list,
#                              physics_type=physics_type, out_dir=out_dir)
