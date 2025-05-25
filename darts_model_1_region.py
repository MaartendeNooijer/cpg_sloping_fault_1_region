from math import fabs
import pickle
import xarray as xr
import h5py
import os
import numpy as np
from scipy.interpolate import interp1d

from darts.reservoirs.reservoir_base import ReservoirBase
from darts.physics.base.physics_base import PhysicsBase

from darts.engines import timer_node, sim_params, value_vector, index_vector, op_vector, ms_well_vector
from darts.engines import print_build_info as engines_pbi
from darts.discretizer import print_build_info as discretizer_pbi
from darts.print_build_info import print_build_info as package_pbi


class DartsModel:
    """
    This is a base class for creating a model in DARTS.
    A model is composed of a :class:`Reservoir` object and a :class:`Physics` object.
    Initialization and communication between these two objects takes place through the Model object

    :ivar reservoir: Reservoir object
    :type reservoir: :class:`ReservoirBase`
    :ivar physics: Physics object
    :type physics: :class:`PhysicsBase`
    """
    reservoir: ReservoirBase
    physics: PhysicsBase

    def __init__(self):
        """
        Initialize DartsModel class.

        :ivar timer: Timer object
        :type timer: :class:`darts.engines.timer_node`
        :ivar params: Object to set simulation parameters
        :type params: :class:`darts.engines.sim_params`
        """
        # print out build information
        engines_pbi()
        discretizer_pbi()
        package_pbi()
        self.timer = timer_node()  # Create time_node object for time record
        self.timer.start()  # Start time record
        self.timer.node["simulation"] = timer_node()  # Create timer.node called "simulation" to record simulation time
        self.timer.node["newton update"] = timer_node()
        self.timer.node["vtk_output"] = timer_node()
        self.timer.node["initialization"] = timer_node()  # Create timer.node called "initialization" to record initialization time
        self.timer.node["initialization"].start()  # Start recording "initialization" time
        self.output_folder = 'output'
        self.sol_filename = "solution.h5"
        self.well_filename = 'well_data.h5'

        self.params = sim_params()  # Create sim_params object to set simulation parameters

        self.timer.node["initialization"].stop()  # Stop recording "initialization" time

    def init(self, discr_type: str = 'tpfa', platform: str = 'cpu', restart: bool = False,
             verbose: bool = False, output_folder: str = None, itor_mode: str = 'adaptive',
             itor_type: str = 'multilinear', is_barycentric: bool = False):
        """
        Function to initialize the model, which includes:
        - initialize well (perforation) position
        - initialize well rate parameters
        - initialize reservoir initial conditions
        - initialize well control settings
        - define list of operator interpolators for accumulation-flux regions and wells
        - initialize engine

        :param discr_type: 'tpfa' for using Python implementation of TPFA, 'mpfa' activates C++ implementation of MPFA
        :type discr_type: str
        :param platform: 'cpu' for CPU, 'gpu' for using GPU for matrix assembly/solvers/interpolators
        :type platform: str
        :param restart: Boolean to check if existing file should be overwritten or appended
        :type restart: bool
        :param verbose: Switch for verbose
        :type verbose: bool
        :param output_folder: folder for h5 output files
        :type output_folder: str
        :param itor_mode: specifies either 'static' or 'adaptive' interpolator
        :type itor_mode: str
        :param itor_type: specifies either 'linear' or 'multilinear' interpolator
        :type itor_type: str
        :param is_barycentric: Flag which turn on barycentric interpolation on Delaunay simplices
        :type is_barycentric: bool
        """
        # Initialize reservoir and Mesh object
        assert self.reservoir is not None, "Reservoir object has not been defined"
        self.reservoir.init_reservoir(verbose)
        self.set_wells()

        # Initialize physics and Engine object
        assert self.physics is not None, "Physics object has not been defined"
        self.physics.init_physics(discr_type=discr_type, platform=platform, verbose=verbose,
                                  itor_mode=itor_mode, itor_type=itor_type, is_barycentric=is_barycentric)
        if platform == 'gpu':
            self.params.linear_type = sim_params.gpu_gmres_cpr_amgx_ilu

        # Initialize well objects
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.init_well_rates()

        if output_folder is not None:
            self.output_folder = output_folder

        self.set_op_list()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.set_well_controls()
        self.reset()

        # self.restart = restart

        # save solution vector
        if restart is False:
            self.save_data_to_h5(kind = 'solution')

    def reset(self):
        """
        Function to initialize the engine by calling 'engine.init()' method.
        """
        self.physics.engine.init(self.reservoir.mesh, ms_well_vector(self.reservoir.wells), op_vector(self.op_list),
                                 self.params, self.timer.node["simulation"])

    def configure_h5_output(self, filename: str, cell_ids, description, add_static_data: bool = False):
        """
        Configuration of *.h5 output

        :param filename: *.h5 filename
        :param cell_ids: np.array of cell indexes for output
        :param description: description for *.h5
        :param add_static_data: flag to add static output
        """
        with h5py.File(filename, 'w') as f:
            ## static data group
            if add_static_data:
                static_group = f.create_group('static')
                block_m = np.array(self.reservoir.mesh.block_m, copy=False)
                block_p = np.array(self.reservoir.mesh.block_p, copy=False)
                static_group.create_dataset('block_m', data=block_m)
                static_group.create_dataset('block_p', data=block_p)

            ## dynamic data group
            dynamic_group = f.create_group('dynamic')
            dynamic_group.create_dataset('time', shape=(0,), maxshape=(None,))

            # add solution
            if self.reservoir.mesh.n_blocks > 0 and self.physics.n_vars > 0:
                nb = cell_ids.size
                cell_ids_dataset = dynamic_group.create_dataset('cell_id', shape=(nb,), dtype=np.int32)
                cell_ids_dataset[:] = cell_ids
                dynamic_group.create_dataset('X', shape=(0, nb, self.physics.n_vars),
                                             maxshape=(None, nb, self.physics.n_vars), dtype = np.float64)

            # add variable names
            datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
            var_names = dynamic_group.create_dataset('variable_names', (self.physics.n_vars,), dtype=datatype)
            var_names[:] = self.physics.vars

            # write brief description
            f.attrs['description'] = description

    def configure_output(self, kind: str):
        """
        Configuration of output
        
        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        :param restart: Boolean to check if existing file should be overwritten or appended
        :type restart: bool
        """

        # Ensure the directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # solution ouput
        if kind == 'solution':
            sol_output_path = os.path.join(self.output_folder, self.sol_filename)
            if os.path.exists(sol_output_path): #and not restart:
                os.remove(sol_output_path)
            self.configure_h5_output(filename=sol_output_path, cell_ids=np.arange(self.reservoir.mesh.n_blocks),
                                add_static_data=False, description='Reservoir data')

        # Find relevant connections for well data
        if kind == 'well':
            block_m = np.array(self.reservoir.mesh.block_m, copy=False)
            block_p = np.array(self.reservoir.mesh.block_p, copy=False)
            well_conn_ids = np.argwhere(block_p >= self.reservoir.mesh.n_res_blocks)[:, 0]
            self.id_well_data = np.unique(block_m[well_conn_ids])

            # well output
            well_output_path = os.path.join(self.output_folder, self.well_filename)
            if os.path.exists(well_output_path):
                os.remove(well_output_path)
            self.configure_h5_output(filename=well_output_path, cell_ids=self.id_well_data,
                                add_static_data=True, description='Well data')

        if hasattr(self, 'output_configured'):
            self.output_configured.append(kind)
        else:
            self.output_configured = [kind]

    def load_restart_data(self, filename: str = os.path.join('restart', 'solution.h5'), timestep = -1):
        """
        Function to load data from previous simulation and uses them for following simulation.
        
        :param output_folder: restart_data filename
        :type output_folder: str
        """
        time, cell_id, X, var_names = self.read_specific_data(filename, timestep)

        print('Restarting from %s at time = %f days' % (filename, time[0]))

        self.physics.engine.t = time[0]
        self.physics.engine.X = value_vector(X.flatten())
        self.physics.engine.Xn = value_vector(X.flatten())

        self.save_data_to_h5(kind='solution')

    def set_wells(self, verbose: bool = False):
        """
        Function to define wells. The default method of DartsModel.set_wells() calls Reservoir.set_wells().

        :param verbose: Switch for verbose
        :type verbose: bool
        """
        self.reservoir.set_wells(verbose)
        return

    def set_initial_conditions(self, initial_values: dict = None, gradient: dict = None):
        """
        Function to set initial conditions. Passes initial conditions to :class:`Mesh` object.

        :param initial_values: Map of scalars/arrays of initial values for each primary variable, keys are the variables
        :type initial_values: dict
        :param gradient: Map of scalars of gradients for initial values
        :type gradient: dict
        """
        initial_values = initial_values if initial_values is not None else self.initial_values
        gradient = gradient if gradient is not None else (self.gradient if hasattr(self, 'gradient') else None)

        self.reservoir.mesh.composition.resize(self.reservoir.mesh.n_blocks * (self.physics.nc - 1))

        for i, variable in enumerate(self.physics.vars):
            # Check if variable exists in initial values dictionary
            if variable not in initial_values.keys():
                raise RuntimeError("Primary variable {} was not assigned initial values.".format(variable))

            if variable == 'pressure':
                values = np.array(self.reservoir.mesh.pressure, copy=False)
            elif variable == 'temperature':
                values = np.array(self.reservoir.mesh.temperature, copy=False)
            elif variable == 'enthalpy':
                values = np.array(self.reservoir.mesh.enthalpy, copy=False)
            else:
                values = np.array(self.reservoir.mesh.composition, copy=False)

            # values = np.array(self.reservoir.mesh.values[i], copy=False)
            initial_value = initial_values[variable]

            if variable not in ['pressure', 'temperature', 'enthalpy']:
                c = i - 1
                values[c::(self.physics.nc - 1)] = initial_value
            elif isinstance(initial_values[variable], (list, np.ndarray)):
                # If initial value is an array, assign array
                values[:] = initial_value
            elif gradient is not None and variable in gradient.keys():
                # If gradient has been defined, calculate distribution over depth and assign to array
                values[:self.reservoir.mesh.n_res_blocks] = initial_value + \
                    np.asarray(self.reservoir.mesh.depth)[:self.reservoir.mesh.n_res_blocks] * gradient[variable]
            else:
                # Else, assign constant value to each cell in array
                values.fill(initial_value)

        return

    def set_initial_conditions_from_depth_table(self, depth, initial_distribution: dict):
        """
        Function to set initial conditions from given distribution of properties over depth.

        :param depth: depth
        :param initial_distribution: initial distributions of unknowns over depth,
                                    must have keys equal to self.physics.vars
        :type initial_distribution: dict
        """

        # all depths
        depths = np.asarray(self.reservoir.mesh.depth)

        # adjust the size of composition array in c++
        self.reservoir.mesh.composition.resize(self.reservoir.mesh.n_blocks * (self.physics.nc - 1))

        z_counter = 0
        nz_vars = self.physics.nc - 1
        for variable in self.physics.vars:
            if variable not in initial_distribution.keys():
                raise RuntimeError("Primary variable {} was not assigned initial values.".format(variable))

            values_foo = interp1d(depth, initial_distribution[variable], kind='linear', fill_value='extrapolate')

            if variable == 'pressure':
                np.asarray(self.reservoir.mesh.pressure)[:] = values_foo(depths)
            elif variable == 'temperature':
                np.asarray(self.reservoir.mesh.temperature)[:] = values_foo(depths)
            elif variable == 'enthalpy':
                np.asarray(self.reservoir.mesh.enthalpy)[:] = values_foo(depths)
            else:           # compositions
                np.asarray(self.reservoir.mesh.composition)[z_counter::nz_vars] = values_foo(depths)
                z_counter += 1

    def set_boundary_conditions(self):
        """
        Function to set boundary conditions. Passes boundary conditions to :class:`Physics` object and wells.

        This function is empty in DartsModel, needs to be overloaded in child Model.
        """
        pass

    def set_well_controls(self):
        """
        Function to set well controls. Passes well controls to :class:`Physics` object and wells.

        This function is empty in DartsModel, needs to be overloaded in child Model.
        """
        pass

    def set_op_list(self):
        """
        Function to define list of operator interpolators for accumulation-flux regions and wells.

        Operator list is in order [acc_flux_itor[0], ..., acc_flux_itor[n-1], acc_flux_w_itor]
        """
        self.op_list = [self.physics.acc_flux_itor[region] for region in self.physics.regions] + [self.physics.acc_flux_w_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks:] = len(self.op_list) - 1


    def set_sim_params(self, first_ts: float = None, mult_ts: float = None, max_ts: float = None, runtime: float = 1000,
                       tol_newton: float = None, tol_linear: float = None, it_newton: int = None, it_linear: int = None,
                       newton_type=None, newton_params=None):
        """
        Function to set simulation parameters.

        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        :param tol_newton: Tolerance for Newton iterations
        :type tol_newton: float
        :param tol_linear: Tolerance for linear iterations
        :type tol_linear: float
        :param it_newton: Maximum number of Newton iterations
        :type it_newton: int
        :param it_linear: Maximum number of linear iterations
        :type it_linear: int
        :param newton_type:
        :param newton_params:
        """
        self.params.first_ts = first_ts if first_ts is not None else self.params.first_ts
        self.params.mult_ts = mult_ts if mult_ts is not None else self.params.mult_ts
        self.params.max_ts = max_ts if max_ts is not None else self.params.max_ts
        self.runtime = runtime

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = tol_newton if tol_newton is not None else self.params.tolerance_newton
        self.params.tolerance_linear = tol_linear if tol_linear is not None else self.params.tolerance_linear
        self.params.max_i_newton = it_newton if it_newton is not None else self.params.max_i_newton
        self.params.max_i_linear = it_linear if it_linear is not None else self.params.max_i_linear

        self.params.newton_type = newton_type if newton_type is not None else self.params.newton_type
        self.params.newton_params = newton_params if newton_params is not None else self.params.newton_params

    def run_simple(self, physics, params, days):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        days = days if days is not None else self.runtime
        self.physics = physics
        self.params = params
        verbose = False

        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0.:
            dt = restart_dt
        else:
            dt = min(self.prev_dt * self.params.mult_ts, self.params.max_ts)
        self.prev_dt = dt

        ts = 0

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                ts += 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt = min(dt * self.params.mult_ts, self.params.max_ts)

                # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
                # to not allow the next time step be smaller than min_ts
                if np.fabs(t + dt - stop_time) < self.params.min_ts:
                    dt = stop_time - t


                if t + dt > stop_time:
                    dt = stop_time - t
                else:
                    self.prev_dt = dt

            else:
                dt /= self.params.mult_ts
                if verbose:
                    print("Cut timestep to %2.10f" % dt)
                if dt < self.params.min_ts:
                    break
                    
        # update current engine time
        self.physics.engine.t = stop_time

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))

    def run(self, days: float = None, restart_dt: float = 0., save_well_data : bool = True, save_solution_data : bool = True, 
            log_3d_body_path: bool = False, verbose: bool = True):
        """
        Method to run simulation for specified time. Optional argument to specify dt to restart simulation with.

        :param days: Time increment [days]
        :type days: float
        :param restart_dt: Restart value for timestep size [days, optional]
        :type restart_dt: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        :param save_well_data: if True save states of well blocks at every time step to 'well_data.h5', default is True
        :type save_well_data: bool
        :param save_solution_data: if True save states of all reservoir blocks at the end of run to 'solution.h5', default is True
        :type save_solution_data: bool
        :param log_3d_body_path: hypercube output
        :type verbose: bool
        """
        days = days if days is not None else self.runtime
        # get current engine time
        t = self.physics.engine.t
        stop_time = t + days

        # same logic as in engine.run
        if fabs(t) < 1e-15 or not hasattr(self, 'prev_dt'):
            dt = self.params.first_ts
        elif restart_dt > 0.:
            dt = restart_dt
        else:

            dt = min(self.prev_dt*self.params.mult_ts, days, self.params.max_ts)
        self.prev_dt = dt

        ts = 0

        if log_3d_body_path:
            self.physics.body_path_start(output_folder=self.output_folder)

        while t < stop_time:
            converged = self.run_timestep(dt, t, verbose)

            if converged:
                t += dt
                self.physics.engine.t = t
                ts += 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt = min(dt * self.params.mult_ts, self.params.max_ts)

                # if the current dt almost covers the rest time amount needed to reach the stop_time, add the rest
                # to not allow the next time step be smaller than min_ts
                if np.fabs(t + dt - stop_time) < self.params.min_ts:
                    dt = stop_time - t

                if t + dt > stop_time:
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
                    print('Stop simulation. Reason: reached min. timestep', self.params.min_ts, 'dt=', dt)
                    return -1

        # update current engine time
        self.physics.engine.t = stop_time

        # save solution vector
        if save_solution_data:
            self.save_data_to_h5(kind='solution')

        if verbose:
            print("TS = %d(%d), NI = %d(%d), LI = %d(%d)"
                  % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                     self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                     self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
        return 0

    def run_timestep(self, dt: float, t: float, verbose: bool = True):
        """
        Method to solve Newton loop for specified timestep

        :param dt: Timestep size [days]
        :type dt: float
        :param t: Current time [days]
        :type t: float
        :param verbose: Switch for verbose, default is True
        :type verbose: bool
        """
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            # self.physics.engine.run_single_newton_iteration(dt)
            self.physics.engine.assemble_linear_system(dt)  # assemble Jacobian and residual of reservoir and well blocks
            self.apply_rhs_flux(dt, t)  # apply RHS flux
            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()  # calc norm of residual

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < self.params.stationary_point_tolerance:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.physics.engine.well_residual_last_dt < self.params.well_tolerance_coefficient * self.params.tolerance_newton) or
                    self.physics.engine.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            r_code = self.physics.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)

        self.timer.node['simulation'].stop()
        return converged

    def do_after_step(self):
        '''
        can be overrided by an user to be executed in the 'run_simulation()'
        '''
        pass

    def run_simulation(self):
        time = 0.0
        for ith_step, dt in enumerate(self.idata.sim.time_steps):
            self.set_well_controls(time=time)
            ret = self.run(dt)
            if ret != 0:
                print('run() failed for the step=', ith_step, 'dt=', dt)
                return 1
            self.do_after_step()
            time += dt
        return 0

    def set_rhs_flux(self, t: float = None) -> np.ndarray:
        """
        Function to specify modifications to RHS vector. User can implement his own boundary conditions here.

        This function is empty in DartsModel, needs to be overloaded in child Model.

        :param t: current time [days]
        :type t: float
        :return: Vector of modification to RHS vector
        :rtype: np.ndarray
        """
        pass

    def apply_rhs_flux(self, dt: float, t: float):
        """
        Function to apply modifications to RHS vector.

        If self.set_rhs_flux() is defined in Model, this function will add its values to rhs

        :param dt: timestep [days]
        :type dt: float
        :param t: current time [days]
        :type t: float
        """
        if type(self).set_rhs_flux is DartsModel.set_rhs_flux:
            # If the function has not been overloaded, pass
            return
        rhs = np.array(self.physics.engine.RHS, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks * self.physics.n_vars
        rhs[:n_res] += self.set_rhs_flux(t) * dt
        return

    def save_data_to_h5(self, kind):
        """
        Function to write output solution or well output to *.h5 file
        
        :param kind: 'well' for well output or 'solution' to write the whole solution vector
        :type kind: str
        
        """

        if not hasattr(self, 'output_configured') or kind not in self.output_configured:
            self.configure_output(kind=kind)

        if kind == 'well':
            path = os.path.join(self.output_folder, self.well_filename)
        elif kind == 'solution':
            path = os.path.join(self.output_folder, self.sol_filename)
        else:
            print("Please use either kind='well' or kind='solution' in save_data_to_h5")
            return
        self.save_specific_data(path)

    def save_specific_data(self, filename):
        """
        Function to write output to *.h5 file
        
        :param filename: path to *.h5 filename to append data to
        :type filename: str
        """
        X = np.array(self.physics.engine.X, copy=False)

        # Open the HDF5 file in append mode
        with h5py.File(filename, "a") as f:
            # Append to time dataset under the dynamic group
            time_dataset = f["dynamic/time"]
            time_dataset.resize((time_dataset.shape[0] + 1,))
            time_dataset[-1] = self.physics.engine.t

            cell_id = f["dynamic/cell_id"][:]

            x_dataset = f["dynamic/X"]
            x_dataset.resize((x_dataset.shape[0] + 1, x_dataset.shape[1], x_dataset.shape[2]))
            x_dataset[x_dataset.shape[0] - 1, :, :] = X.reshape((self.reservoir.mesh.n_blocks, self.physics.n_vars))[cell_id]

    def read_specific_data(self, filename: str, timestep: int = None):
        """
        Function to read *.h5 files contents.
        
        :param filename: path to *.h5 filename to append data to
        :param timestep:
        :return time: time of the saved data in days
        :rtype: np.ndarray
        :return cell_id: cell id of each of the saved grid blocks
        :rtype: np.ndarray
        :return X: variables names
        :rtype: np.ndarray
        """

        # Open the HDF5 file
        with h5py.File(filename, 'r') as file:
            if timestep is None:
                datapoints = file['dynamic/X'].shape[0] * file['dynamic/X'].shape[1] * file['dynamic/X'].shape[2]
                print('WARNING: %s contains %d data points...' % (filename, datapoints))

                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][:]
                X = file['dynamic/X'][:]
            else:
                cell_id = file['dynamic/cell_id'][:]
                var_names = file['dynamic/variable_names'][:]
                time = file['dynamic/time'][timestep].reshape(1)
                X = file['dynamic/X'][timestep].reshape(1, len(cell_id), len(var_names))


        for i, name in enumerate(var_names):
            var_names[i] = name.decode()

        return time, cell_id, X, var_names

    def output_properties(self, output_properties: list = None, timestep: int = None) -> tuple:  # NEW
        """
        Function to read *.h5 data and evaluate properties per grid block, per timestep
        :param output_properties: List of properties to evaluate for output
        :return property_array : dictionary containing the states and evaluated properties
        :return timesteps: np.ndarray containing the timesteps at which the properties were evaluated
        :rtype: tuple
        """
        # Read binary file
        path = os.path.join(self.output_folder, self.sol_filename)
        if timestep is None:
            timesteps, cell_id, X, var_names = self.read_specific_data(path)
        else:
            timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)

        # Initialize property_array
        n_vars = len(var_names)
        n_ops = self.physics.n_ops
        nb = self.reservoir.mesh.n_res_blocks

        output_properties = output_properties if output_properties is not None else []
        primary_props = [prop for prop in output_properties if prop in var_names]
        primary_prop_idxs = {prop: list(var_names).index(prop) for prop in primary_props}
        secondary_props = [prop for prop in output_properties if prop not in var_names]
        secondary_prop_idxs = {
            prop: list(self.physics.property_containers[next(iter(self.physics.property_containers))].
                       output_props.keys()).index(prop) for prop in secondary_props}
        property_array = {prop: np.zeros((len(timesteps), nb)) for prop in primary_props + secondary_props}

        # Loop over timesteps
        for k, timestep in enumerate(timesteps):
            # Extract primary properties from X vector
            for var_name, var_idx in primary_prop_idxs.items():
                property_array[var_name][k] = X[k, :nb, var_idx]

            # Interpolate secondary properties
            if secondary_props:
                state = value_vector(np.stack([X[k, :nb, j] for j in range(n_vars)]).T.flatten())
                values = value_vector(np.zeros(n_ops * nb))
                values_numpy = np.array(values, copy=False)
                dvalues = value_vector(np.zeros(n_ops * nb * n_vars))
                i = 0
                for region, prop_itor in self.physics.property_itor.items():
                    region_indices = self.physics.engine.region_cell_idx[i]
                    region_np_idx = np.array(region_indices, dtype=np.int32)

                    # Temporary buffer for current region
                    temp_values = value_vector(np.zeros(n_ops * nb))
                    prop_itor.evaluate_with_derivatives(state, region_indices, temp_values, dvalues)
                    temp_numpy = np.array(temp_values, copy=False)

                    # Insert only this region's results into the shared buffer
                    for op in range(n_ops):
                        values_numpy[region_np_idx * n_ops + op] = temp_numpy[region_np_idx * n_ops + op]

                    i += 1

                for prop_name, prop_idx in secondary_prop_idxs.items():
                    property_array[prop_name][k] = values_numpy[prop_idx::n_ops]

        # add additional user's arrays: if self.out is defined
        if hasattr(self, 'out'):
            for k in self.out.keys():
                if self.out[k] is not None:
                    self.out[k] = self.out[k].reshape((1, self.out[k].size))
            property_array.update(self.out)

        return timesteps, property_array


    #
    # def output_properties(self, output_properties: list = None, timestep: int = None) -> tuple:  # NEW
    #     """
    #     Function to read *.h5 data and evaluate properties per grid block, per timestep
    #     :param output_properties: List of properties to evaluate for output
    #     :return property_array : dictionary containing the states and evaluated properties
    #     :return timesteps: np.ndarray containing the timesteps at which the properties were evaluated
    #     :rtype: tuple
    #     """
    #     # Read binary file
    #     path = os.path.join(self.output_folder, self.sol_filename)
    #     if timestep is None:
    #         timesteps, cell_id, X, var_names = self.read_specific_data(path)
    #     else:
    #         timesteps, cell_id, X, var_names = self.read_specific_data(path, timestep)
    #
    #     # Initialize property_array
    #     n_vars = len(var_names)
    #     n_ops = self.physics.n_ops
    #     nb = self.reservoir.mesh.n_res_blocks
    #
    #     output_properties = output_properties if output_properties is not None else []
    #     primary_props = [prop for prop in output_properties if prop in var_names]
    #     primary_prop_idxs = {prop: list(var_names).index(prop) for prop in primary_props}
    #     secondary_props = [prop for prop in output_properties if prop not in var_names]
    #     secondary_prop_idxs = {
    #         prop: list(self.physics.property_containers[next(iter(self.physics.property_containers))].
    #                    output_props.keys()).index(prop) for prop in secondary_props}
    #     property_array = {prop: np.zeros((len(timesteps), nb)) for prop in primary_props + secondary_props}
    #
    #     # Loop over timesteps
    #     for k, timestep in enumerate(timesteps):
    #         # Extract primary properties from X vector
    #         for var_name, var_idx in primary_prop_idxs.items():
    #             property_array[var_name][k] = X[k, :nb, var_idx]
    #
    #         # Interpolate secondary properties
    #         if secondary_props:
    #             state = value_vector(np.stack([X[k, :nb, j] for j in range(n_vars)]).T.flatten())
    #             values = value_vector(np.zeros(n_ops * nb))
    #             values_numpy = np.array(values, copy=False)
    #             dvalues = value_vector(np.zeros(n_ops * nb * n_vars))
    #             i = 0
    #             for region, prop_itor in self.physics.property_itor.items():
    #                 region_indices = self.physics.engine.region_cell_idx[i]
    #                 region_np_idx = np.array(region_indices, dtype=np.int32)
    #
    #                 # Temporary buffer for current region
    #                 temp_values = value_vector(np.zeros(n_ops * nb))
    #                 prop_itor.evaluate_with_derivatives(state, region_indices, temp_values, dvalues)
    #                 temp_numpy = np.array(temp_values, copy=False)
    #
    #                 # Insert only this region's results into the shared buffer
    #                 for op in range(n_ops):
    #                     values_numpy[region_np_idx * n_ops + op] = temp_numpy[region_np_idx * n_ops + op]
    #
    #                 i += 1
    #
    #             for prop_name, prop_idx in secondary_prop_idxs.items():
    #                 property_array[prop_name][k] = values_numpy[prop_idx::n_ops]
    #
    #     # add additional user's arrays: if self.out is defined
    #     if hasattr(self, 'out'):
    #         for k in self.out.keys():
    #             if self.out[k] is not None:
    #                 self.out[k] = self.out[k].reshape((1, self.out[k].size))
    #         property_array.update(self.out)
    #
    #     return timesteps, property_array
    #

    def output_to_plt(self, output_properties: list = None, ith_step: int = None, lims: dict = None, fig=None,
                      figsize: tuple = None, axs_shape: tuple = None, aspect_ratio: str = 'equal', logx: bool = False,
                      plot_zeros: bool = True, cmap: str = 'jet', colorbar_loc: str = 'right',
                      output_directory: str = None, file_format: str = "pdf"):
        """
        Function to plot results with matplotlib.

        :param output_properties: List of properties to plot, default is None which will pass all
        :type output_properties: list
        :param ith_step: i'th reporting step
        :type ith_step: int
        :param lims: Ranges of colorbars, default is empty
        :type lims: dict
        :param fig: Optional figure object to append plots, default is None
        :param figsize: Tuple of (width, height) for figure
        :param axs_shape: Tuple of (rows, columns) for figure
        :param aspect_ratio: Aspect ratio of plots ('equal', 'auto', or float), default is 'equal'
        :param logx: Bool to plot x-axis in logscale, default is False
        :param plot_zeros: Bool to plot zero values, default is True
        :param cmap: plt.Colourmap, default is 'jet'
        :param colorbar_loc: Location of colorbar ('right' or 'bottom'), default is 'right'
        :param output_directory: Directory to save file
        :param file_format: File format, 'pdf' is default
        """
        # Set default output directory
        if output_directory is None:
            output_directory = self.output_folder

        # Find properties to output
        timesteps, property_array = self.output_properties(output_properties=output_properties, timestep=ith_step)

        # Pass to Reservoir.plot() method
        fig = self.reservoir.output_to_plt(data=property_array, output_props=output_properties, lims=lims, fig=fig,
                                           figsize=figsize, axs_shape=axs_shape, aspect_ratio=aspect_ratio, logx=logx,
                                           plot_zeros=plot_zeros, cmap=cmap, colorbar_loc=colorbar_loc)

        import matplotlib.pyplot as plt
        plt.savefig(output_directory + '/step' + str(ith_step) + '.' + file_format)

        return fig

    def output_to_xarray(self, output_properties: list = None, timestep: int = None):
        """
        Function to return array of properties.
        Primary variables (vars) are obtained from engine, secondary variables (props) are interpolated by property_itor.

        :returns: property_array
        :rtype: np.ndarray
        """
        # Interpolate properties
        if timestep is None:
            timesteps, data = self.output_properties(output_properties)
        else:
            timesteps, data = self.output_properties(output_properties, timestep)
        props = list(data.keys())

        # Initialize coords and data_vars for Xarray Dataset
        array_shape = (len(timesteps), self.reservoir.nx, self.reservoir.ny, self.reservoir.nz)
        for prop, array in data.items():
            data[prop] = array.reshape(array_shape)

        # Initialize coords and data_vars for Xarray Dataset
        dx, dy, dz = self.reservoir.global_data['dx'], self.reservoir.global_data['dy'], self.reservoir.global_data['dz']
        x = np.cumsum(dx[:, 0, 0]) - dx[0, 0, 0]*0.5
        y = np.cumsum(dy[0, :, 0]) - dy[0, 0, 0]*0.5
        z = np.cumsum(dz[0, 0, :]) - dz[0, 0, 0]*0.5
        coords = {'time': timesteps, 'x': x, 'y': y, 'z': z}
        data_vars = {prop: (list(coords.keys()), data[prop]) for prop in props}
        dataset = xr.Dataset(data_vars=data_vars, coords=coords)

        dataset.to_netcdf(os.path.join(self.output_folder, 'solution_xarray.nc'))

        return dataset

    def output_to_vtk(self, ith_step: int = None, output_directory: str = None, output_properties: list = None):
        """
        Function to export results at timestamp t into `.vtk` format.

        :param ith_step: i'th reporting step
        :type ith_step: int
        :param output_directory: Name to save .vtk file
        :type output_directory: str
        :param output_properties: List of properties to include in .vtk file, default is None which will pass all
        :type output_properties: list
        """
        self.timer.node["vtk_output"].start()
        # Set default output directory
        if output_directory is None:
            output_directory = self.output_folder

        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = output_properties if output_properties is not None else list(ev_props)
        
        #Adding temperature and pressure to output VTK 
        if self.physics.thermal == True:
            props_names = props_names + ['pressure','temperature']
        else:
            props_names = props_names + ['pressure'] #, 'temperature']

        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)

        # Pass to Reservoir.output_to_vtk() method
        self.reservoir.output_to_vtk(ith_step, timesteps, output_directory, list(property_array.keys()), property_array)
        self.timer.node["vtk_output"].stop()

    def print_timers(self):
        """
        Function to print the time information, including total time elapsed,
        time consumption at different stages of the simulation, etc..
        """
        print(self.timer.print("", ""))

    def print_stat(self):
        """
        Function to print the statistics information, including total timesteps, Newton iteration, linear iteration, etc..
        """
        self.physics.engine.print_stat()

    def init_well_rates(self):
        """
        Function that prepare data for the calculation of well rates (in Python)
        """
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        self.well_perf_conn_ids = {}
        self.well_head_conn_id = {}
        for well in self.reservoir.wells:
            res_cell_ids = [perf[1] for perf in well.perforations]

            # find ids of those connections which 1. block_p is in res_cell_ids, 2. block_m is well cell
            conn_ids = np.nonzero(np.logical_and(np.isin(block_p, res_cell_ids), \
                                                 block_m >= self.reservoir.mesh.n_res_blocks))
            self.well_perf_conn_ids[well.name] = conn_ids[0]
            assert (self.well_perf_conn_ids[well.name].size == len(well.perforations) and \
                    (block_m[self.well_perf_conn_ids[well.name]] > self.reservoir.mesh.n_res_blocks).all())
            # find id of well_head -> well_body connection in the connection list
            well_head_conn_id = np.where(np.logical_and(block_m == well.well_head_idx, block_p == well.well_body_idx))[0]
            assert(len(well_head_conn_id) == 1)
            self.well_head_conn_id[well.name] = well_head_conn_id[0]

    def reconstruct_velocities(self):
        # velocity discretization
        values, offset = self.reservoir.discretizer.discretize_velocities(cell_m=np.asarray(self.reservoir.mesh.block_m),
                                                                            cell_p=np.asarray(self.reservoir.mesh.block_p),
                                                                            geom_coef=np.asarray(self.reservoir.mesh.tranD),
                                                                            n_res_blocks=self.reservoir.mesh.n_res_blocks)
        self.reservoir.mesh.velocity_appr.resize(len(values))
        self.reservoir.mesh.velocity_offset.resize(len(offset))

        velocity_appr = np.asarray(self.reservoir.mesh.velocity_appr)
        velocity_appr[:] = values
        velocity_offset = np.asarray(self.reservoir.mesh.velocity_offset)
        velocity_offset[:] = offset

        # specify molar weights to get rid of molar density multiplier in flux terms
        nc = self.physics.nc
        self.physics.engine.molar_weights.resize(nc * len(self.physics.regions))
        molar_weights = np.asarray(self.physics.engine.molar_weights)
        for i, region in enumerate(self.physics.regions):
            molar_weights[i * nc:(i + 1) * nc] = self.physics.property_containers[region].Mw

        # resize storage for velocities inside engine
        self.physics.engine.darcy_velocities.resize(self.reservoir.mesh.n_res_blocks * self.physics.nph * 3)
        
        # allocate & transfer data to device
        if self.platform == 'gpu':
            from darts.engines import copy_data_to_device, allocate_device_data
            # velocity_appr
            velocity_appr_d = self.physics.engine.get_velocity_appr_d()
            allocate_device_data(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            copy_data_to_device(self.reservoir.mesh.velocity_appr, velocity_appr_d)
            # velocity_offset_d
            velocity_offset_d = self.physics.engine.get_velocity_offset_d()
            allocate_device_data(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            copy_data_to_device(self.reservoir.mesh.velocity_offset, velocity_offset_d)
            # darcy_velocities_d
            darcy_velocities_d = self.physics.engine.get_darcy_velocities_d()
            allocate_device_data(self.physics.engine.darcy_velocities, darcy_velocities_d)
            # molar_weights_d
            molar_weights_d = self.physics.engine.get_molar_weights_d()
            allocate_device_data(self.physics.engine.molar_weights, molar_weights_d)
            copy_data_to_device(self.physics.engine.molar_weights, molar_weights_d)
            # op_num_d
            op_num_d = self.physics.engine.get_op_num_d()
            allocate_device_data(self.reservoir.mesh.op_num, op_num_d)
            copy_data_to_device(self.reservoir.mesh.op_num, op_num_d)

    # destructor to force to destroy all created C objects and free memory
    def __del__(self):
        for name in list(vars(self).keys()):
            delattr(self, name)
