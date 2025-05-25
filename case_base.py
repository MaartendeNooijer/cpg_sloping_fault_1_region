import numpy as np
import os

from darts.input.input_data import InputData
from darts.input.input_data import WellData
from darts.models.darts_model import sim_params

class CustomWellData(WellData):
    def read_and_add_perforations(self, sch_fname, verbose: bool = False):
        """
        read COMPDAT from SCH file in Eclipse format, add wells and perforations
        note: uses only I,J,K1,K2 and optionally WellIndex parameters from the COMPDAT keyword
        :param: sch_fname - path to file
        """
        if sch_fname is None:
            return
        print('reading wells (COMPDAT) from', sch_fname)
        well_diam = 0.152  # m.  #TODO read from the keyword parameters
        well_radius = well_diam / 2.

        keep_reading = True
        prev_well_name = ''
        with open(sch_fname) as f:
            while keep_reading:
                buff = f.readline()
                if 'COMPDAT' in buff:
                    while True:  # be careful here
                        buff = f.readline()
                        if len(buff) != 0:
                            CompDat = buff.split()
                            wname = CompDat[0].strip('"').strip("'")  # remove quotas (" and ')
                            if len(CompDat) != 0 and '/' != wname:  # skip the empty line and '/' line
                                # define perforation
                                i1 = int(CompDat[1])
                                j1 = int(CompDat[2])
                                k1 = int(CompDat[3])
                                k2 = int(CompDat[4])

                                well_index = None
                                if len(CompDat) > 7:
                                    if CompDat[7] != '*':
                                        well_index = float(CompDat[7])

                                for k in range(k1, k2 + 1):
                                    #TODO support time>0
                                    self.add_perforation(name=wname, time=0.0, loc_ijk=(i1, j1, k),
                                                         status='open', well_radius=well_radius,
                                                         well_index=well_index, well_indexD=None,
                                                         multi_segment= True)
                                                        # multi_segment= True) #NEW, used to be FALSE. Not sure if it helps
                            if len(CompDat) != 0 and '/' == CompDat[0]:
                                keep_reading = False
                                break
        print('WELLS read from SCH file:', len(self.wells))

class CustomInputData(InputData):
    def __init__(self, type_hydr, type_mech, init_type):
        super().__init__(type_hydr, type_mech, init_type)
        self.well_data = CustomWellData()  # 👈 override just this

class InputDataGeom():  # to group geometry input data
    def __init__(self):
        pass

def get_case_files(case: str):
    prefix = os.path.join('meshes', case[:case.rfind('_')])
    grid_file = os.path.join(prefix, 'grid.grdecl')
    prop_file = os.path.join(prefix, 'reservoir.in')
    sch_file = os.path.join(prefix, 'sch.inc')
    fault_file = os.path.join(prefix, 'faults.inc')  #NEW # Add fault file
    assert os.path.exists(fault_file), 'Cannot open ' + fault_file #NEW
    assert os.path.exists(grid_file), 'cannot open' + grid_file
    assert os.path.exists(prop_file), 'cannot open' + prop_file
    assert os.path.exists(sch_file), 'cannot open' + sch_file
    return grid_file, prop_file, sch_file, fault_file #NEW

#def input_data_base(idata: InputData, case: str):
def input_data_base(idata: CustomInputData, case: str):
    dt = 50 #365 #365.25 #365.25  # one report timestep length, [days]
    idata.sim.n_time_steps = 15 #3 #8 #5 #8 #20 #20 #12#4 #30 #was 20
    idata.sim.time_steps = np.zeros(idata.sim.n_time_steps) + dt

    # time stepping and convergence parameters
    idata.sim.first_ts = 1e-5 #0.01  #was 0.01 #Alex suggested 1e-5
    idata.sim.mult_ts = 1.3
    idata.sim.max_ts = 10 #
    idata.sim.runtime = 300
    idata.sim.tol_newton = 1e-3 #1e-2 #1e-2 #was 1e-2 #Alex suggested 1e-4
    idata.sim.tol_linear = 1e-5 #1e-4 #1e-5 #was 1e-4 #Alex suggested 1e-5

    # use direct linear solver:
    #idata.sim.linear_type = sim_params.linear_solver_t.cpu_superlu

    idata.generate_grid = 'generate' in case
    idata.geom = InputDataGeom()
    geom = idata.geom  # a short name
    well_data = idata.well_data  # a short name

    # grid processing parameters
    geom.minpv = 1e-5  # minimal pore volume threshold to set cells inactive, m^3

    # properties processing parameters
    # for the isothermal physics - porosity cutoff value
    # for thermal physics - poro and perm with lower values will be replaced by geom.min_poro:
    #     poro - to keep those cells active even though they have poro=0
    #     perm - to avoid convergence issues
    geom.min_poro = 1e-5

    # boundary conditions
    geom.bound_volume = 1e10 # lateral boundary volume, m^3

    if idata.generate_grid:
        idata.rock.poro = 0.2
        idata.rock.permx = 100  # mD
        idata.rock.permy = 100  # mD
        idata.rock.permz = 10   # mD
    else:  # read from files
        # setup filenames
        gridfile, propfile, schfile, faultfile = get_case_files(case)
        idata.gridfile = gridfile
        idata.propfile = propfile if os.path.exists(propfile) else gridfile
        idata.schfile = schfile
        idata.faultfile = faultfile #NEW
        # read from a file to idata.well_data.wells[well_name].perforations
        idata.well_data.read_and_add_perforations(idata.schfile)
    idata.grid_out_dir = None  # output path for the generated grid and prop files

    # rock compressibility
    idata.rock.compressibility = 1e-5  # [1/bars]
    idata.rock.compressibility_ref_p = 1 # [bars]
    idata.rock.compressibility_ref_T = 273.15  # [K]

    #########################################################################
    # only for the thermal case (Geothermal physics):
    geom.burden_layers = 4  # the number of additional (generated on-the-fly) overburden/underburden layers
    geom.burden_init_thickness = 10  # first over/under burden layer thickness, [m.]
    idata.rock.burden_prop = 1e-5  # perm and poro value for burden layers

    idata.rock.conduction_shale = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
    idata.rock.conduction_sand = 3 * 86.4 # Sandstone conductivity kJ/m/day/K

    idata.rock.hcap_shale = 2300 # Shale heat capacity kJ/m3/K
    idata.rock.hcap_sand = 2450 # Sandstone heat capacity kJ/m3/K

    # the cells with lower poro will be treated as shale when setting the rock thermal properties
    idata.rock.poro_shale_threshold = 1e-3
    ############################################################################


# import numpy as np
# import os
#
# from darts.input.input_data import InputData
# from darts.models.darts_model import sim_params
#
# class InputDataGeom():  # to group geometry input data
#     def __init__(self):
#         pass
#
# def get_case_files(case: str):
#     prefix = os.path.join('meshes', case[:case.rfind('_')])
#     grid_file = os.path.join(prefix, 'grid.grdecl')
#     prop_file = os.path.join(prefix, 'reservoir.in')
#     sch_file = os.path.join(prefix, 'sch.inc')
#     fault_file = os.path.join(prefix, 'faults.inc')  # NEW # Add fault file
#     assert os.path.exists(grid_file), 'cannot open' + grid_file
#     assert os.path.exists(prop_file), 'cannot open' + prop_file
#     assert os.path.exists(sch_file), 'cannot open' + sch_file
#     return grid_file, prop_file, sch_file, fault_file
#
# def input_data_base(idata: InputData, case: str):
#     dt = 365.25  # one report timestep length, [days]
#     n_time_steps = 1
#     idata.sim.time_steps = np.zeros(n_time_steps) + dt
#
#     # time stepping and convergence parameters
#     idata.sim.first_ts = 1e-5
#     idata.sim.mult_ts = 1.3
#     idata.sim.max_ts = 10
#     idata.sim.runtime = None
#     idata.sim.tol_newton = 1e-3
#     idata.sim.tol_linear = 1e-5
#     # use direct linear solver:
#     #idata.sim.linear_type = sim_params.linear_solver_t.cpu_superlu
#
#     idata.generate_grid = 'generate' in case
#     idata.geom = InputDataGeom()
#     geom = idata.geom  # a short name
#     well_data = idata.well_data  # a short name
#
#     # grid processing parameters
#     geom.minpv = 1e-5  # minimal pore volume threshold to set cells inactive, m^3
#
#     # properties processing parameters
#     # for the isothermal physics - porosity cutoff value
#     # for thermal physics - poro and perm with lower values will be replaced by geom.min_poro:
#     #     poro - to keep those cells active even though they have poro=0
#     #     perm - to avoid convergence issues
#     geom.min_poro = 1e-5
#
#     # boundary conditions
#     geom.bound_volume = 1e10 # lateral boundary volume, m^3
#
#     geom.faultfile = None  # a text file with fault locations and multipliers
#
#     if idata.generate_grid:
#         idata.rock.poro = 0.2
#         idata.rock.permx = 100  # mD
#         idata.rock.permy = 100  # mD
#         idata.rock.permz = 10   # mD
#     else:  # read from files
#         # setup filenames
#         gridfile, propfile, schfile, fault_file = get_case_files(case)
#         idata.gridfile = gridfile
#         idata.propfile = propfile if os.path.exists(propfile) else gridfile
#         idata.schfile = schfile
#         idata.faultfile = fault_file
#         # read from a file to idata.well_data.wells[well_name].perforations
#         idata.well_data.read_and_add_perforations(idata.schfile)
#     idata.grid_out_dir = None  # output path for the generated grid and prop files
#
#     # rock compressibility
#     idata.rock.compressibility = 1e-5  # [1/bars]
#     idata.rock.compressibility_ref_p = 1 # [bars]
#     idata.rock.compressibility_ref_T = 273.15  # [K]
#
#     #########################################################################
#     # only for the thermal case (Geothermal physics):
#     geom.burden_layers = 4  # the number of additional (generated on-the-fly) overburden/underburden layers
#     geom.burden_init_thickness = 10  # first over/under burden layer thickness, [m.]
#     idata.rock.burden_prop = 1e-5  # perm and poro value for burden layers
#
#     idata.rock.conduction_shale = 2.2 * 86.4 # Shale conductivity kJ/m/day/K
#     idata.rock.conduction_sand = 3 * 86.4 # Sandstone conductivity kJ/m/day/K
#     idata.rock.hcap_shale = 2300 # Shale heat capacity kJ/m3/K
#     idata.rock.hcap_sand = 2450 # Sandstone heat capacity kJ/m3/K
#
#     # the cells with lower poro will be treated as shale when setting the rock thermal properties
#     idata.rock.poro_shale_threshold = 1e-3
#     ############################################################################
