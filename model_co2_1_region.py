import numpy as np

from darts.input.input_data import InputData
# from darts.models.darts_model import DartsModel
from darts_model_1_region import DartsModel
from set_case import set_input_data
from darts.engines import value_vector

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData
from model_cpg_1_region import Model_CPG, fmt

class AlternativeContainer(PropertyContainer):
    def run_flash(self, pressure, temperature, zc):
        # Normalize fluid compositions
        zc_norm = zc if not self.ns else zc[:self.nc_fl] / (1. - np.sum(zc[self.nc_fl:]))

        # Evaluates flash, then uses getter for nu and x - for compatibility with DARTS-flash
        error_output = self.flash_ev.evaluate(pressure, temperature, zc_norm)
        flash_results = self.flash_ev.get_flash_results()
        self.nu = np.array(flash_results.nu)
        try:
            self.x = np.array(flash_results.X).reshape(self.np_fl, self.nc_fl)
        except ValueError as e:
            print(e.args[0], pressure, temperature, zc)
            error_output += 1

        # Set present phase idxs
        ph = np.array([j for j in range(self.np_fl) if self.nu[j] > 0])

        if ph.size == 1:
            self.x[ph[0]] = zc_norm

        return ph

class ModelCCS(Model_CPG):
    def __init__(self):
        self.zero = 1e-10
        super().__init__()



    def set_physics(self):
        """Physical properties"""
        # Fluid components, ions and solid
        # components = ["H2O", "CO2"]
        # phases = ["Aq", "V"]

        self.components = ["CO2", "H2O"]
        phases = ["V", "Aq"]

        nc = len(self.components)
        comp_data = CompData(self.components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        # aq = Jager2003(comp_data)
        aq = AQEoS(comp_data, AQEoS.Ziabakhsh2012)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["PR", "AQ"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        temperature = None

        self.physics = Compositional(self.components, phases, self.timer, n_points=self.idata.obl.n_points,
                                     min_p=self.idata.obl.min_p, max_p=self.idata.obl.max_p,
                                     min_z=self.idata.obl.min_z, max_z=self.idata.obl.max_z,
                                     min_t=self.idata.obl.min_t, max_t=self.idata.obl.max_t,
                                     thermal=self.idata.thermal, cache=self.idata.obl.cache)

        self.physics.n_axes_points[0] = 1001

        """ properties correlations """
        property_container = AlternativeContainer(phases_name=phases, components_name=self.components,
                                                  Mw=comp_data.Mw[:2],
                                                  temperature=temperature, min_z=self.zero / 10)

        property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])

        property_container.density_ev = dict([('V', EoSDensity(pr, comp_data.Mw[:2])),
                                              ('Aq', Garcia2001(self.components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(self.components))])

        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas")),
                                               ('Aq', PhaseRelPerm("oil"))])

        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(pr)),
                                               ('Aq', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('V', ConstFunc(8.4)),
                                                   ('Aq', ConstFunc(170.)), ])

        # property_container.capillary_pressure_ev = CapillaryPressure(corey_params)  # interesting

        self.physics.add_property_region(property_container)

        property_container.output_props = {"satV": lambda: property_container.sat[0],
                                           "satA": lambda: property_container.sat[1],
                                           "xCO2": lambda: property_container.x[1, 0],
                                           "yCO2": lambda: property_container.x[0, 0],
                                           "xCO2": lambda: property_container.x[1, 1],
                                           "yCO2": lambda: property_container.x[0, 1], }

        return

    def get_arrays(self, ith_step):
        '''
        :return: dictionary of current unknown arrays (p, T)
        '''
        # Find index of properties to output
        ev_props = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        # If output_properties is None, all variables and properties from property_operators will be passed
        props_names = list(ev_props)
        timesteps, property_array = self.output_properties(output_properties=props_names, timestep=ith_step)
        return property_array

    def set_input_data(self, case=''):
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type='gradient')
        set_input_data(self.idata, case)

        self.idata.geom.burden_layers = 0

        # well controls
        wdata = self.idata.well_data
        wells = wdata.wells  # short name
        # set default injection composition
        #wdata.inj = value_vector([self.zero])  # injection composition - water

        mt = 1e9  # kg
        kg_per_kmol = 44.01  # kg/kmol
        inj_rate = mt / kg_per_kmol / 365.25  # kmol/day

        if 'wbhp' in case:
            for w in wells:
                wdata.add_inj_bhp_control(name=w, bhp=180, comp_index=0, temperature=300)  # kmol/day | bars | K
        elif 'wrate' in case:
            for w in wells:
                wdata.add_inj_rate_control(name=w, rate=inj_rate, comp_index=0, bhp_constraint=180, temperature=300)  # kmol/day | bars | K

        self.idata.obl.n_points = 1000
        self.idata.obl.zero = 1e-11
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 400.
        self.idata.obl.min_t = 273.15
        self.idata.obl.max_t = 373.15
        self.idata.obl.min_z = self.idata.obl.zero
        self.idata.obl.max_z = 1 - self.idata.obl.zero
        self.idata.obl.cache = False
        self.idata.thermal = True

    def set_initial_conditions(self):
        self.temperature_initial_ = 273.15 + 76.85  # K
        self.initial_values = {"pressure": 100.,
                            "H2O": 1 - self.zero, #0.99995,
                            "CO2": self.zero,
                            "temperature": self.temperature_initial_
                            }

        super().set_initial_conditions()

        mesh = self.reservoir.mesh
        depth = np.array(mesh.depth, copy=True)
        # set initial pressure
        pressure_grad = 97.75 #bar/km
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = depth[:pressure.size] / 1000 * pressure_grad + 1
        # set initial temperature
        temperature_grad = 30
        temperature = np.array(mesh.temperature, copy=False)
        temperature[:] = depth[:pressure.size] / 1000 * temperature_grad + 273.15 + 20
        #temperature[:] = 350

    def set_well_controls(self, time: float = 0., verbose=True):
        '''
        :param time: simulation time, [days]
        :return:
        '''
        inj_stream_base = [1 - self.zero * 100]
        eps_time = 1e-15

        wdata = self.idata.well_data
        wells = wdata.wells  # short name

        i, j, k = 15, 15, 1  # 105, 59, 1 #90, 95, 1 #do this manually - not nice but easy
        nx, ny, nz = int(self.reservoir.dims[0]), int(self.reservoir.dims[1]), int(self.reservoir.dims[2])

        res_block_local = (i - 1) + nx * (j - 1) + nx * ny * (k - 1)
        well_head_depth = self.reservoir.depth[res_block_local]
        pressure_gradient = 97.75  # bar / km
        well_head_depth_inj_pressure = 1 + well_head_depth * pressure_gradient / 1000 + 5  # dP of 5 bar
        print('well_head_depth_pressure = ', well_head_depth_inj_pressure)
        #well_head_depth_inj_pressure = 175
        mt = 1e9  # kg
        kg_per_kmol = 44.01  # kg/kmol
        inj_rate = mt / kg_per_kmol / 365.25  # kmol/day

        for w in self.reservoir.wells:
            # find next well control in controls list for different timesteps
            wctrl = None
            for wctrl_t in self.idata.well_data.wells[w.name].controls:
                if np.fabs(wctrl_t[0] - time) < eps_time:  # check time
                    wctrl = wctrl_t[1]
                    break
            if wctrl is None:
                continue
            if wctrl.type == 'inj':  # INJ well
                inj_stream = inj_stream_base
                if self.physics.thermal:
                    inj_stream += [wctrl.inj_bht]
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_inj(wctrl.rate, inj_stream, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_inj(well_head_depth_inj_pressure, inj_stream)
                    #w.constraint = self.physics.new_bhp_inj(wctrl.bhp_constraint, inj_stream)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_inj(well_head_depth_inj_pressure, inj_stream)
                    #w.control = self.physics.new_bhp_inj(wctrl.bhp, inj_stream)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            elif wctrl.type == 'prod':  # PROD well
                if wctrl.mode == 'rate': # rate control
                    w.control = self.physics.new_rate_prod(wctrl.rate, wctrl.comp_index)
                    w.constraint = self.physics.new_bhp_prod(wctrl.bhp_constraint)
                elif wctrl.mode == 'bhp': # BHP control
                    w.control = self.physics.new_bhp_prod(wctrl.bhp)
                else:
                    print('Unknown well ctrl.mode', wctrl.mode)
                    exit(1)
            else:
                print('Unknown well ctrl.type', wctrl.type)
                exit(1)
            if verbose:
                print('set_well_controls: time=', time, 'well=', w.name, w.control, w.constraint)

        # check
        for w in self.reservoir.wells:
            assert w.control is not None, 'well control is not initialized for the well ' + w.name
            if verbose and w.constraint is None and 'rate' in str(type(w.control)):
                print('A constraint for the well ' + w.name + ' is not initialized!')


    def print_well_rate(self):
        return