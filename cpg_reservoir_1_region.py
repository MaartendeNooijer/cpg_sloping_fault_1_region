from darts.engines import conn_mesh, ms_well, ms_well_vector, timer_node
from darts.discretizer import load_single_float_keyword, load_single_int_keyword
from darts.discretizer import value_vector as value_vector_discr
from darts.discretizer import index_vector as index_vector_discr
import numpy as np
from typing import Union, List, Dict

from opmcpg._cpggrid import UnstructuredGrid, process_cpg_grid
from opmcpg._cpggrid import value_vector as value_vector_cpggrid
from opmcpg._cpggrid import index_vector as index_vector_cpggrid

from darts.discretizer import Mesh, Elem, Discretizer, BoundaryCondition, elem_loc, elem_type
from darts.discretizer import index_vector, value_vector, matrix33, vector_matrix33, vector_vector3
from darts.discretizer import load_single_float_keyword
from darts.reservoirs.mesh.struct_discretizer import StructDiscretizer
from darts.reservoirs.reservoir_base import ReservoirBase

import datetime, time
import darts
from pyevtk import hl, vtk
from pyevtk.hl import pointsToVTK
import warnings

try:
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtk import vtkCellArray, vtkHexahedron, vtkPoints
except ImportError:
    warnings.warn("No vtk module loaded.")

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.insert(0, os.path.join(parentdir2, 'python'))


class CPG_Reservoir(ReservoirBase):
    def __init__(self, timer: timer_node, arrays=None, faultfile: str = None, minpv: float = 0., cache: bool = False):
        """
        Class constructor for CPG_Reservoir class (corner-point geometry)
        :param arrays: dictionary of numpy arrays with grid (COORD, ZCORN, ACTNUM) and props (PORO, PERMX, PERMY, PERMZ)
        :param faultfile: file name with fault locations (IJK) and fault transmicssibility multipliers
        :param minpv: cells with poro volume smaller than minpv will be set inactive
        :param cache:
        """
        super().__init__(timer, cache)

        self.arrays = arrays
        self.faultfile = faultfile
        self.minpv = minpv  # minimal pore volume threshold to make cells inactive, m3

        self.snap_counter = 0

        self.vtk_z = 0
        self.vtk_y = 0
        self.vtk_x = 0
        self.vtk_filenames_and_times = {}
        self.vtkobj = 0
        self.vtk_grid_type = 1

    def set_arrays(self, arrays):
        """
        :param arrays: dictionary of input data for the grid and grid properties
        """
        self.dims = arrays['SPECGRID']  # dimensions, array of 3 integer elements: nx, ny ,nz
        self.coord = arrays['COORD']  # grid pillars, array of (nx+1)*(ny+1)*6 elements
        self.zcorn = arrays['ZCORN']  # grid nodes depths, array of nx*ny*nz*8 elements
        self.actnum = arrays['ACTNUM']  # integer array of nx*ny*nz elements, 0 - inactive cell, 1 - active cell
        self.poro = arrays['PORO']  # porosity array, nx*ny*nz elements
        # permeability arrays, nx*ny*nz elements
        self.permx = arrays['PERMX']
        self.permy = arrays['PERMY']
        self.permz = arrays['PERMZ']
        
        self.discr_mesh.poro = value_vector_discr(self.poro)
        self.discr_mesh.coord = value_vector_discr(self.coord)
        self.discr_mesh.zcorn = value_vector_discr(self.zcorn)
        self.discr_mesh.actnum = index_vector_discr(self.actnum)

        self.permx_cpp = value_vector_discr(self.permx)
        self.permy_cpp = value_vector_discr(self.permy)
        self.permz_cpp = value_vector_discr(self.permz)


    def discretize(self, verbose: bool = False) -> conn_mesh:
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        # discretizer's mesh object - for computing transmissibility and create connectivity graph
        self.discr_mesh = Mesh()

        self.set_arrays(self.arrays)

        self.discretize_cpg()
        # self.discretizer.write_mpfa_results('conn.dat')

        self.global_data = {'volume': self.volume_all_cells[:self.discr_mesh.n_cells],
                            'global_to_local': self.discr_mesh.global_to_local,
                            'poro': self.poro, 'permx': self.permx, 'permy': self.permy, 'permz': self.permz,
                            'depth': self.depth_all_cells[:self.discr_mesh.n_cells], 'actnum': self.actnum
                            }

        mpfa_tran = np.array(self.discretizer.flux_vals, copy=False)
        mpfa_tranD = np.array(self.discretizer.flux_vals_thermal, copy=False)
        ids = np.array(self.discretizer.get_one_way_tpfa_transmissibilities())
        cell_m = np.array(self.discretizer.cell_m)[ids]
        cell_p = np.array(self.discretizer.cell_p)[ids]

        # self.discretizer.write_tran_cube('tran_cpg.grdecl', 'nnc_cpg.txt')
        if self.faultfile is not None:
            self.apply_fault_mult(self.faultfile, cell_m, cell_p, mpfa_tran, ids)
            # self.discretizer.write_tran_cube('tran_faultmult.grdecl', 'nnc_faultmult.txt')

        tran = mpfa_tran[::2][ids]
        tranD = mpfa_tranD[::2][ids]

        print('tran  mean=', tran.mean(),  'tran  max=', tran.max())
        print('tranD mean=', tranD.mean(), 'tranD max=', tranD.max())

        
        self.mesh.init(darts.engines.index_vector(cell_m), darts.engines.index_vector(cell_p),
                       darts.engines.value_vector(tran), darts.engines.value_vector(tranD))

        # debug
        # d = {'cell_m': cell_m, 'cell_p': cell_p, 'tran': tran, 'tranD': tranD}
        # import pandas as pd
        # df_cpg = pd.DataFrame(data=d)
        # df_struct = pd.read_excel('tran_struct.xlsx')
        # with pd.ExcelWriter('tran.xlsx') as writer:
        #    df_struct.to_excel(writer, sheet_name='struct')
        #    df_cpg.to_excel(writer, sheet_name='cpg')

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        self.mesh.depth = darts.engines.value_vector(self.discr_mesh.depths)
        self.mesh.volume = darts.engines.value_vector(self.discr_mesh.volumes)
        self.bc = np.array(self.mesh.bc, copy=False)

        # rock thermal properties
        self.hcap = np.array(self.mesh.heat_capacity, copy=False)
        self.conduction = np.array(self.mesh.rock_cond, copy=False)
        print('hello')
        return self.mesh

    def discretize_cpg(self):
        '''
        reads grid and reservoir properties, initialize mesh, creates discretizer object and computes
        transmissibilities using two point flux approximation
        :param gridfile: text file with DIMENS, COORD, ZCORN data (grdecl)
        :param propfile: text file with PORO, PERM data (grdecl)
        :return: None
        '''

        # empty dict just to pass to func
        displaced_tags = dict()
        displaced_tags[elem_loc.MATRIX] = set()
        displaced_tags[elem_loc.FRACTURE] = set()
        displaced_tags[elem_loc.BOUNDARY] = set()
        displaced_tags[elem_loc.FRACTURE_BOUNDARY] = set()

        result_fname = 'results.grdecl'

        dims_cpp = index_vector_cpggrid(self.dims)
        coord_cpp = value_vector_cpggrid(self.coord)
        zcorn_cpp = value_vector_cpggrid(self.zcorn)
        actnum_cpp = index_vector_cpggrid(self.actnum)
        unstr_grid = process_cpg_grid(dims_cpp, coord_cpp, zcorn_cpp, actnum_cpp, self.minpv, result_fname)

        self.nx = self.discr_mesh.nx = self.dims[0]
        self.ny = self.discr_mesh.ny = self.dims[1]
        self.nz = self.discr_mesh.nz = self.dims[2]
        self.nb = self.mesh.n_res_blocks
        self.discr_mesh.n_cells = unstr_grid.number_of_cells
        # cells + boundary_faces, approximate
        self.discr_mesh.num_of_elements = self.discr_mesh.n_cells + \
                                          2 * (self.nx * self.ny +
                                               self.ny * self.nz +
                                               self.nx * self.nz)

        number_of_nodes = unstr_grid.number_of_nodes
        number_of_cells = unstr_grid.number_of_cells
        number_of_faces = unstr_grid.number_of_faces
        node_coordinates = value_vector(np.array(unstr_grid.node_coordinates, copy=False))
        face_nodes = index_vector(np.array(unstr_grid.face_nodes, copy=False))
        face_nodepos = index_vector(np.array(unstr_grid.face_nodepos, copy=False))
        face_cells = index_vector(np.array(unstr_grid.face_cells, copy=False))
        cell_faces = index_vector(np.array(unstr_grid.cell_faces, copy=False))
        cell_facetag = index_vector(np.array(unstr_grid.cell_facetag, copy=False))
        global_cell = index_vector(np.array(unstr_grid.global_cell, copy=False))
        cell_facepos = index_vector(np.array(unstr_grid.cell_facepos, copy=False))
        cell_volumes = value_vector(np.array(unstr_grid.cell_volumes, copy=False))
        cell_centroids = value_vector(np.array(unstr_grid.cell_centroids, copy=False))
        face_normals = value_vector(np.array(unstr_grid.face_normals, copy=False))
        face_areas = value_vector(np.array(unstr_grid.face_areas, copy=False))
        face_centroids = value_vector(np.array(unstr_grid.face_centroids, copy=False))

        face_order = index_vector()

        res = self.discr_mesh.cpg_elems_nodes(
            number_of_nodes, number_of_cells, number_of_faces,
            node_coordinates, face_nodes, face_nodepos,
            face_cells, cell_faces, cell_facepos,
            cell_volumes, face_order)

        bnd_faces_num = res[0]
        # self.discr_mesh.print_elems_nodes()

        # store min max coordinates
        nodes_3d = np.array(unstr_grid.node_coordinates, copy=False)
        nodes_3d = nodes_3d.reshape(number_of_nodes, 3)
        self.x_min, self.x_max = nodes_3d[:, 0].min(), nodes_3d[:, 0].max()
        self.y_min, self.y_max = nodes_3d[:, 1].min(), nodes_3d[:, 1].max()
        self.z_min, self.z_max = nodes_3d[:, 2].min(), nodes_3d[:, 2].max()

        self.discr_mesh.construct_local_global(global_cell)

        self.discr_mesh.cpg_cell_props(number_of_nodes, number_of_cells, number_of_faces,
                                       cell_volumes, cell_centroids, global_cell,
                                       face_areas, face_centroids,
                                       bnd_faces_num, face_order)

        self.discr_mesh.cpg_connections(number_of_cells, number_of_faces,
                                        node_coordinates, face_nodes, face_nodepos,
                                        face_cells, cell_faces, cell_facepos,
                                        face_centroids, face_areas, face_normals,
                                        cell_facetag, displaced_tags)

        self.discr_mesh.generate_adjacency_matrix()

        self.discretizer = Discretizer()
        self.cpp_bc = self.set_boundary_conditions(displaced_tags)
        self.discretizer.set_mesh(self.discr_mesh)

        self.volume_all_cells = np.array(self.discr_mesh.volumes, copy=False)
        self.depth_all_cells = np.array(self.discr_mesh.depths, copy=False)
        self.centroids_all_cells = np.array(self.discr_mesh.centroids)
        self.actnum = np.array(self.discr_mesh.actnum, copy=False)

        self.discretizer.set_permeability(self.permx_cpp, self.permy_cpp, self.permz_cpp)

        n_all = self.nx * self.ny * self.nz
        print("Number of all cells    = ", n_all)
        print("Number of active cells = ", self.discr_mesh.n_cells)

        # poro could be modified here
        # self.poro[poro < 1e-2] = 1e-2
        self.discretizer.set_porosity(self.discr_mesh.poro)
        self.mesh.poro = darts.engines.value_vector(self.discretizer.poro)
        self.poro = np.array(self.discr_mesh.poro, copy=False)

        # calculate transmissibilities
        self.discretizer.calc_tpfa_transmissibilities(displaced_tags)
        return

    def calc_well_index(self, i, j, k, well_radius=0.0762, segment_direction='z_axis', skin=0):
        """
        Class method which construct the well index for each well segment/perforation
        :param i: "human" counting of x-location coordinate of perforation
        :param j: "human" counting of y-location coordinate of perforation
        :param k: "human" counting of z-location coordinate of perforation
        :param well_radius: radius of the well-bore
        :param segment_direction: direction in which the segment perforates the reservoir block
        :param skin: skin factor for pressure loss around well-bore due to formation damage
        :return well_index: well-index of particular perforation
        """
        assert (i > 0), "Perforation block coordinate should be positive"
        assert (j > 0), "Perforation block coordinate should be positive"
        assert (k > 0), "Perforation block coordinate should be positive"
        assert (i <= self.nx), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (j <= self.ny), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        assert (k <= self.nz), "Perforation block coordinate should not exceed corresponding reservoir dimension"
        i -= 1
        j -= 1
        k -= 1

        # compute reservoir block index
        res_block = self.discr_mesh.get_global_index(i, j, k)

        # local index
        local_block = self.discr_mesh.global_to_local[res_block]

        # check if target grid block is active
        if local_block > -1:
            dx, dy, dz = self.discr_mesh.calc_cell_sizes(i, j, k)

            eps = 1e-6  # to avoid divizion by zero
            kx = self.permx[res_block] + eps
            ky = self.permy[res_block] + eps
            kz = self.permz[res_block] + eps

            well_index = 0
            if segment_direction == 'z_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kx) * dx ** 2 + np.sqrt(kx / ky) * dy ** 2) / \
                               ((ky / kx) ** (1 / 4) + (kx / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * ky) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dx ** 2 + dy ** 2) / 2.
                well_indexD = 2 * np.pi * dz / (np.log(conduction_rad / well_radius) + skin)
                if kx == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'x_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(ky / kz) * dz ** 2 + np.sqrt(kz / ky) * dy ** 2) / \
                               ((ky / kz) ** (1 / 4) + (kz / ky) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kz * ky) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dz ** 2 + dy ** 2) / 2.
                well_indexD = 2 * np.pi * dx / (np.log(conduction_rad / well_radius) + skin)
                if kz == 0 or ky == 0: well_index = 0.0
            elif segment_direction == 'y_axis':
                peaceman_rad = 0.28 * np.sqrt(np.sqrt(kz / kx) * dx ** 2 + np.sqrt(kx / kz) * dz ** 2) / \
                               ((kz / kx) ** (1 / 4) + (kx / kz) ** (1 / 4))
                well_index = 2 * np.pi * dz * np.sqrt(kx * kz) / (np.log(peaceman_rad / well_radius) + skin)
                conduction_rad = 0.28 * np.sqrt(dz ** 2 + dx ** 2) / 2.
                well_indexD = 2 * np.pi * dy / (np.log(conduction_rad / well_radius) + skin)
                if kx == 0 or kz == 0: well_index = 0.0

            well_index = well_index * StructDiscretizer.darcy_constant
        else:
            well_index = 0
            well_indexD = 0

        return local_block, well_index, well_indexD

    def set_boundary_volume(self, xy_minus=-1, xy_plus=-1, yz_minus=-1, yz_plus=-1, xz_minus=-1, xz_plus=-1):
        mesh_volume = np.array(self.volume_all_cells, copy=False)
        local_to_global = np.array(self.discr_mesh.local_to_global, copy=False)
        global_to_local = np.array(self.discr_mesh.global_to_local, copy=False)

        # get 3d shape
        volume = make_full_cube(mesh_volume[:self.discr_mesh.n_cells], local_to_global, global_to_local)
        volume = volume.reshape(self.nx, self.ny, self.nz, order='F')

        actnum3d = self.actnum.reshape(self.nx, self.ny, self.nz, order='F')

        # apply changes
        if xy_minus > -1:
            for i in range(self.nx):
                for j in range(self.ny):
                    k = 0
                    while k < self.nz and actnum3d[i, j, k] == 0:  # search first active cell
                        k += 1
                    if k < self.nz:
                        volume[i, j, k] = xy_minus
        if xy_plus > -1:
            for i in range(self.nx):
                for j in range(self.ny):
                    k = self.nz - 1
                    while k >= 0 and actnum3d[i, j, k] == 0:
                        k -= 1
                    if k >= 0:
                        volume[i, j, k] = xy_plus
        if yz_minus > -1:
            for k in range(self.nz):
                for j in range(self.ny):
                    i = 0
                    while i < self.nx and actnum3d[i, j, k] == 0:
                        i += 1
                    if i < self.nx:
                        volume[i, j, k] = yz_minus
        if yz_plus > -1:
            for k in range(self.nz):
                for j in range(self.ny):
                    i = self.nx - 1
                    while i >= 0 and actnum3d[i, j, k] == 0:
                        i -= 1
                    if i >= 0:
                        volume[i, j, k] = yz_plus
        if xz_minus > -1:
            for k in range(self.nz):
                for i in range(self.nx):
                    j = 0
                    while j < self.ny and actnum3d[i, j, k] == 0:
                        j += 1
                    if j < self.ny:
                        volume[i, j, k] = xz_minus
        if xz_plus > -1:
            for k in range(self.nz):
                for i in range(self.nx):
                    j = self.ny - 1
                    while j >= 0 and actnum3d[i, j, k] == 0:
                        j -= 1
                    if j >= 0:
                        volume[i, j, k] = xz_plus
        volume_1d = np.reshape(volume, self.discr_mesh.nx * self.discr_mesh.ny * self.discr_mesh.nz,
                               order='F')  # back to 1D
        # apply actnum and assign to mesh.volume
        mesh_volume[:self.discr_mesh.n_cells] = volume_1d[self.discr_mesh.local_to_global]

    def set_boundary_conditions(self, physical_tags):
        return
        bc = BoundaryCondition()

        boundary_range = self.discr_mesh.region_ranges[elem_loc.BOUNDARY]
        a = np.zeros(boundary_range[1] - boundary_range[0])
        b = np.zeros(boundary_range[1] - boundary_range[0])
        r = np.zeros(boundary_range[1] - boundary_range[0])

        # no-flow (impermeable) bc
        a[:] = 0.
        b[:] = 1.
        r[:] = 0.

        bc.a = value_vector(a)
        bc.b = value_vector(b)
        bc.r = value_vector(r)

        return bc

    def add_perforation(self, well_name: str, cell_index: Union[int, tuple], well_radius: float = 0.1524,
                        well_index: float = None, well_indexD: float = None, segment_direction: str = 'z_axis',
                        skin: float = 0., multi_segment: bool = False, verbose: bool = False):
        """
        Function to add perforations to wells.
        """
        well = self.get_well(well_name)

        # calculate well index and get local index of reservoir block
        # ijk indices are is 1-based (starts from 1)
        i, j, k = cell_index
        res_block_local, wi, wiD = self.calc_well_index(i, j, k, well_radius=well_radius,
                                                        segment_direction=segment_direction, skin=skin)

        if well_index is None:
            well_index = wi

        if well_indexD is None:
            well_indexD = wiD

        assert well_index >= 0
        assert well_indexD >= 0

        # set well segment index (well block) equal to index of perforation layer
        if multi_segment:
            well_block = len(well.perforations)
        else:
            well_block = 0

        # add completion only if target block is active
        if res_block_local > -1:
            if len(well.perforations) == 0:  # if adding the first perforation
                well.well_head_depth = self.depth_all_cells[res_block_local]
                well.well_body_depth = well.well_head_depth
                dx, dy, dz = self.discr_mesh.calc_cell_sizes(i - 1, j - 1, k - 1)
                well.segment_depth_increment = dz
                well.segment_volume *= well.segment_depth_increment
            else:  # update well depth
                well.well_head_depth = min(well.well_head_depth, self.depth_all_cells[res_block_local])
                well.well_body_depth = well.well_head_depth
            for p in well.perforations:
                if p[0] == well_block and p[1] == res_block_local:
                    print('Neglected duplicate perforation for well %s to block [%d, %d, %d]' % (well.name, i, j, k))
                    return
            well.perforations = well.perforations + [(well_block, res_block_local, well_index, well_indexD)]
            if verbose:
                c = self.centroids_all_cells[res_block_local].values
                print('Added perforation for well %s to block %d IJK=[%d, %d, %d] XYZ=(%f, %f, %f) with WI=%f WID=%f' % (
                    well.name, res_block_local, i, j, k, c[0], c[1], c[2], well_index, well_indexD))
        else:
            if verbose:
                print('Neglected perforation for well %s to block [%d, %d, %d] (inactive block)' % (well.name, i, j, k))
        return

    def write_mpfa_conn_to_file(self, path='mpfa_conn.dat'):
        stencil = np.array(self.discretizer.flux_stencil, copy=False)
        trans = np.array(self.discretizer.flux_vals, copy=False)

        f = open(path, 'w')
        f.write(str(len(self.discretizer.cell_m)) + '\n')

        for conn_id in range(len(self.discretizer.cell_m)):
            cells = stencil[self.discretizer.flux_offset[conn_id]:self.discretizer.flux_offset[conn_id + 1]]
            coefs = trans[self.discretizer.flux_offset[conn_id]:self.discretizer.flux_offset[conn_id + 1]]
            # row = str(self.discretizer.cell_m[conn_id]) + '\t' + str(self.discretizer.cell_p[conn_id])
            row = str(self.discretizer.cell_m[conn_id]) + '\t' + str(self.discretizer.cell_p[conn_id]) + '\t\t'
            # row_cells = ''#str(cells)
            # row_vals = ''#str(coefs)
            for i in range(cells.size):
                if np.abs(coefs[i]) > 1.E-10:
                    row += str(cells[i]) + '\t' + str('{:.2e}'.format(coefs[i])) + '\t'
                    # row_cells += str(cells[i]) + '\t'
                    # row_vals += str('{:.2e}'.format(coefs[i])) + '\t'
            f.write(row + '\n')  # + row_cells + '\n' + row_vals + '\n')
        f.close()

    def init_vtk(self, output_directory: str, export_grid_data: bool = True):
        """
        Method to initialize objects required for output of structured reservoir into `.vtk` format.
        This method can also export the mesh properties, e.g. porosity, permeability, etc.

        :param output_directory: Path for output
        :type output_directory: str
        :param export_grid_data: Switch for mesh properties output, default is True
        :type export_grid_data: bool
        """
        from pyevtk.hl import gridToVTK

        self.vtk_initialized = True
        self.vtk_z = 0
        self.vtk_y = 0
        self.vtk_x = 0
        self.vtk_filenames_and_times = {}
        self.vtkobj = 0

        self.generate_cpg_vtk_grid()
        self.nodes_tot = self.nx * self.ny * self.nz
        self.local_to_global = np.array(self.discr_mesh.local_to_global, copy=False)

        if export_grid_data:
            cell_data = {}
            mesh_geom_dtype = np.float32
            for key, data in self.global_data.items():
                if np.isscalar(data):
                    if type(data) is int:
                        cell_data[key] = data * np.ones(self.nodes_tot, dtype=int)
                    elif type(data) is float:
                        cell_data[key] = data * np.ones(self.nodes_tot, dtype=mesh_geom_dtype)
                else:
                    cell_data[key] = np.array(data)
            mesh_filename = output_directory + '/mesh'

            if self.vtk_grid_type == 0:
                vtk_file_name = gridToVTK(mesh_filename, self.vtk_x, self.vtk_y, self.vtk_z, cellData=cell_data)
            else:
                g_to_l = np.array(self.discr_mesh.global_to_local, copy=False)
                for key, value in cell_data.items():
                    if cell_data[key].size == g_to_l.size:
                        a = cell_data[key][g_to_l >= 0]
                    else:
                        a = cell_data[key]
                    self.vtkobj.AppendScalarData(key, a)

                vtk_file_name = self.vtkobj.Write2VTU(mesh_filename)
                if len(self.vtk_filenames_and_times) == 0:
                    for key, data in self.global_data.items():
                        self.vtkobj.VTK_Grids.GetCellData().RemoveArray(key)
                    self.vtkobj.VTK_Grids.GetCellData().RemoveArray('cellNormals')
        return

    def output_to_vtk(self, ith_step: int, time_steps: float, output_directory: str, prop_names: list, data: dict):
        from pyevtk.hl import gridToVTK
        from pyevtk.vtk import VtkGroup
        # only for the first export call
        os.makedirs(output_directory, exist_ok=True)
        if not self.vtk_initialized:
            self.init_vtk(output_directory)

        for ts, t in enumerate(time_steps):

            if len(time_steps) == 1:
                vtk_file_name = output_directory + '/solution_ts{}'.format(ith_step)
            else:
                vtk_file_name = output_directory + '/solution_ts{}'.format(ts)

            cell_data = {}
            for prop in prop_names:
                local_data = data[prop][ts]
                global_array = np.ones(self.nodes_tot, dtype=local_data.dtype) * np.nan
                dummy_zeros = np.zeros(
                    self.discr_mesh.n_cells - self.mesh.n_res_blocks)  # workaround for the issue in case of cells without active neighbours
                v = np.append(local_data[:self.mesh.n_res_blocks], dummy_zeros)
                global_array[self.discr_mesh.local_to_global] = v[:]
                cell_data[prop] = global_array

            if self.vtk_grid_type == 0:
                vtk_file_name = gridToVTK(vtk_file_name, self.vtk_x, self.vtk_y, self.vtk_z, cellData=cell_data)
            else:
                for key, value in cell_data.items():

                    g_to_l = np.array(self.discr_mesh.global_to_local, copy=False)
                    if cell_data[key].size == g_to_l.size:
                        a = cell_data[key][g_to_l >= 0]
                    else:
                        a = cell_data[key]
                    self.vtkobj.AppendScalarData(key, a)

                vtk_file_name = self.vtkobj.Write2VTU(vtk_file_name)
                if len(self.vtk_filenames_and_times) == 0:
                    for key, data in self.global_data.items():
                        self.vtkobj.VTK_Grids.GetCellData().RemoveArray(key)
                    self.vtkobj.VTK_Grids.GetCellData().RemoveArray('cellNormals')

            # in order to have correct timesteps in Paraview, write down group file
            # since the library in use (pyevtk) requires the group file to call .save() method in the end,
            # and does not support reading, track all written files and times and re-write the complete
            # group file every time

            self.vtk_filenames_and_times[vtk_file_name] = t
            vtk_group = VtkGroup('solution')
            for fname, t in self.vtk_filenames_and_times.items():
                vtk_group.addFile(fname, t)
            vtk_group.save()

    def generate_cpg_vtk_grid(self):

        from darts.tools import GRDECL2VTK

        self.vtkobj = GRDECL2VTK.GeologyModel()
        self.vtkobj.GRDECL_Data.COORD = self.discr_mesh.coord
        self.vtkobj.GRDECL_Data.ZCORN = self.discr_mesh.zcorn
        self.vtkobj.GRDECL_Data.NX = self.nx
        self.vtkobj.GRDECL_Data.NY = self.ny
        self.vtkobj.GRDECL_Data.NZ = self.nz
        self.vtkobj.GRDECL_Data.N = self.nx * self.ny * self.nz
        self.vtkobj.GRDECL_Data.GRID_type = 'CornerPoint'

        start = time.perf_counter()
        # c++ implementation using discretizer.pyd
        print('[Geometry] Converting GRDECL to Paraview Hexahedron mesh data...')
        nodes_cpp = self.discr_mesh.get_nodes_array()
        nodes_1d = np.array(nodes_cpp, copy=True)
        points = nodes_1d.reshape((nodes_1d.size // 3, 3))
        points[:,2] *= -1  # invert z-coordinate

        cells_1d = np.arange(self.discr_mesh.n_cells * 8)
        cells = cells_1d.reshape((cells_1d.size // 8, 8))
        cells = [("hexahedron", cells)]

        offset = np.arange(self.discr_mesh.n_cells + 1) * 8
        offset_vtk = numpy_to_vtk(np.asarray(offset, dtype=np.int64), deep=True)

        cells_vtk = numpy_to_vtk(np.asarray(cells_1d, dtype=np.int64), deep=True)

        cellArray = vtkCellArray()
        cellArray.SetNumberOfCells(cells_1d.size)
        cellArray.SetData(offset_vtk, cells_vtk)

        Cell = vtkHexahedron()
        self.vtkobj.VTK_Grids.SetCells(Cell.GetCellType(), cellArray)

        vtk_points = vtkPoints()
        vtk_points.SetNumberOfPoints(points.size)
        points_vtk = numpy_to_vtk(np.asarray(points, dtype=np.float32), deep=True)
        vtk_points.SetData(points_vtk)
        self.vtkobj.VTK_Grids.SetPoints(vtk_points)

        print("     NumOfPoints", self.vtkobj.VTK_Grids.GetNumberOfPoints())
        print("     NumOfCells", self.vtkobj.VTK_Grids.GetNumberOfCells())

        # 3. Load grid properties data if applicable
        for keyword, data in self.vtkobj.GRDECL_Data.SpatialDatas.items():
            self.vtkobj.AppendScalarData(keyword, data)

        end = time.perf_counter()
        print('Done! init vtk time:', end - start, 'sec.')

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

    #
    # def apply_fault_mult(self, faultfile, cell_m, cell_p, mpfa_tran, ids):
    #     # Faults
    #
    #     keep_reading = True
    #     prev_fault_name = ''
    #
    #     with open(faultfile) as f:
    #         while True:
    #             buff = f.readline()
    #             strline = buff.split()
    #             if len(strline) == 0 or '/' == strline[0]:
    #                 break
    #             fault_name = strline[0]
    #             # multiply tran
    #             i1 = int(strline[1])
    #             j1 = int(strline[2])
    #             k1 = int(strline[3])
    #             i2 = int(strline[4])
    #             j2 = int(strline[5])
    #             k2 = int(strline[6]) #was k2 = int(strline[6])
    #             fault_tran_mult = float(strline[7])
    #
    #             if i1 > self.discr_mesh.nx or j1 > self.discr_mesh.ny or k1 > self.discr_mesh.nz:
    #                 print('Error:', i1, j1, k1, 'out of grid', buff)
    #                 continue  # skip
    #             if i2 > self.discr_mesh.nx or j2 > self.discr_mesh.ny or k2 > self.discr_mesh.nz:
    #                 print('Error:', i2, j2, k2, 'out of grid', buff)
    #                 continue  # skip
    #
    #             m_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i1 - 1, j1 - 1, k1 - 1)]
    #             p_idx = self.discr_mesh.global_to_local[self.discr_mesh.get_global_index(i2 - 1, j2 - 1, k2 - 1)]
    #
    #             p = set(np.where(cell_p == p_idx)[0])  # find cell idx in cell_p
    #             m = set(np.where(cell_m == m_idx)[0])
    #             res = m & p  # find connection (cell should be in both
    #             if len(res) > 0:
    #                 idx = res.pop()
    #                 mpfa_tran[2 * ids[idx]] *= fault_tran_mult
    #
    #             print('fault tran mult', fault_tran_mult)

    def apply_volume_depth(self):
        self.depth = np.array(self.mesh.depth, copy=False)
        self.volume = np.array(self.mesh.volume, copy=False)

        # self.depth_all_cells[self.depth_all_cells < 1e-6] = 1e-6
        # self.volume_all_cells[self.volume_all_cells < 1e-6] = 1e-6

        self.depth[:] = self.depth_all_cells
        self.volume[:] = self.volume_all_cells

    def create_vtk_wells(self, output_directory: str):
        '''
        creates a file wells.vtk with a tube per well based on its first perforation
        :param output_directory: 
        :return: 
        '''
        import vtk
        well_vtk_filename = os.path.join(output_directory, 'wells.vtk')
        # Append multiple cylinders into one polydata
        appendFilter = vtk.vtkAppendPolyData()

        def create_tube(center, prolongation=1000):
            # Create points for the polyline
            points = vtk.vtkPoints()
            points.InsertNextPoint(center[0], center[1], -center[2] + prolongation)  # Point 1
            points.InsertNextPoint(center[0], center[1], -center[2])  # Point 2

            # Create a polyline that connects the points
            lines = vtk.vtkCellArray()
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)  # Number of points
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            lines.InsertNextCell(line)

            # Create a polydata to hold the points and the polyline
            polyData = vtk.vtkPolyData()
            polyData.SetPoints(points)
            polyData.SetLines(lines)

            # Apply vtkTubeFilter to create a tube around the polyline
            tubeFilter = vtk.vtkTubeFilter()
            tubeFilter.SetInputData(polyData)
            tubeFilter.SetRadius(35)  # Tube radius
            tubeFilter.SetNumberOfSides(50)  # Smoothness of the tube
            tubeFilter.Update()

            return tubeFilter.GetOutput()

        for w in self.wells:
            prolongation = 1000
            for p in w.perforations:
                well_block, res_block_local, well_index, well_indexD = p
                c = self.centroids_all_cells[res_block_local].values
                cyl = create_tube(c, prolongation=prolongation)
                appendFilter.AddInputData(cyl)
                prolongation = 0
                break  # use only the first perf

        # Update the append filter to combine the polydata
        appendFilter.Update()

        # Write the cylinders to a VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(well_vtk_filename)
        writer.SetInputConnection(appendFilter.GetOutputPort())
        writer.Write()

    def get_ijk_from_xyz(self, x, y, z):
        '''
        :return: tuple of I,J,K indices (1-based) of the closest cell to the point with coordinates x,y,z
        '''
        def find_cell_index(centers_flattened, coord) -> int:
            min_dis = None
            idx = None
            for j, centroid in enumerate(centers_flattened):
               dis = np.linalg.norm(np.array(coord) - centroid.values)
               if (min_dis is not None and dis < min_dis) or min_dis is None:
                   min_dis = dis
                   idx = j
            return idx
        def get_ijk(idx, nx, ny, nz):
            k = idx // (nx * ny)
            j = (idx - k * (nx * ny)) // nx
            i = idx % nx
            return (i + 1, j + 1, k + 1)

        centers = self.centroids_all_cells[:self.discr_mesh.n_cells]
        idx = find_cell_index(centers, np.array([x, y, z]))
        ijk = get_ijk(idx, self.nx, self.ny, self.nz)
        return ijk

    def centers_to_vtk(self, out_dir):
        # output center points to VTK
        fname = os.path.join(out_dir, 'centers')
        c_cpg = self.centroids_all_cells[:self.discr_mesh.n_cells]
        c = np.zeros((self.discr_mesh.n_cells, 3))
        for i in range(self.discr_mesh.n_cells):
            cv = c_cpg[i].values
            c[i, 0], c[i, 1], c[i, 2] = cv[0], cv[1], cv[2]  # x, y, z
        x, y, z = c[:, 0].flatten(), c[:, 1].flatten(), -c[:, 2].flatten()
        if c is not None:
            pointsToVTK(fname, x, y, z)

    def save_grdecl(self, arrays_save, fname):
        '''
        saves cubes into a text file (grdecl format), nx*ny*nz values, I is the fastest index
        arrays - dictionary of numpy arrays, dimension of n active cells
        fname - file name to output
        '''

        actnum = self.global_data['actnum']
        fname_suf = fname + '.grdecl'

        local_to_global = np.array(self.discr_mesh.local_to_global, copy=False)
        global_to_local = np.array(self.discr_mesh.global_to_local, copy=False)

        save_array(actnum, fname_suf, 'ACTNUM', local_to_global, global_to_local, 'w')
        for arr_name in arrays_save.keys():
            make_full = True
            if arr_name in ['SPECGRID', 'COORD', 'ZCORN']:
                make_full = False
            save_array(arrays_save[arr_name], fname_suf, arr_name, local_to_global, global_to_local, 'a', make_full)

    def update_perm(self, permx, permy, permz):
        '''
        recompute the transmissiblity without re-initializing the reservoir since there are no changes in the geometry
        '''
        # make 1D, also convert the type to be able to convert to value_vector_discr
        # store to self to save in vtk
        assert self.reservoir.permx.size == permx.flatten().size, f'Grid and perm shapes are not consistent: {self.reservoir.permx.size}, {permx.size}'
        self.reservoir.permx = np.array(permx, dtype=np.float64).flatten()
        self.reservoir.permy = np.array(permy, dtype=np.float64).flatten()
        self.reservoir.permz = np.array(permz, dtype=np.float64).flatten()
        permx = value_vector_discr(self.reservoir.permx)
        permy = value_vector_discr(self.reservoir.permy)
        permz = value_vector_discr(self.reservoir.permz)
        self.reservoir.discretizer.set_permeability(permx, permy, permz)
        # calculate transmissibilities
        displaced_tags = dict()
        displaced_tags[elem_loc.MATRIX] = set()
        displaced_tags[elem_loc.FRACTURE] = set()
        displaced_tags[elem_loc.BOUNDARY] = set()
        displaced_tags[elem_loc.FRACTURE_BOUNDARY] = set()
        self.reservoir.discretizer.calc_tpfa_transmissibilities(displaced_tags)


#####################################################################

def save_array(arr: np.array, fname: str, keyword: str, local_to_global: np.array, global_to_local: np.array, mode='w',
               make_full=True, inactive_value='min'):
    '''
    writes numpy array of n_active_cell size to text file in GRDECL format with n_cells_total
    :param arr: numpy array to write
    :param fname: filename
    :param keyword: keyword for array
    :param actnum: actnum array
    :param mode: 'w' to rewrite the file or 'a' to append
    :param make_full: set this to True if passing arr only in active cells, and to False if it as already nx*ny*nz
    :param inactive_value: if 'min' the value in inactive cells will be set to arr.min(), otherwise to the specified val
    :return: None
    '''
    if make_full:
        arr_full = make_full_cube(arr, local_to_global, global_to_local, inactive_value)
    else:
        arr_full = arr
    with open(fname, mode) as f:
        f.write(keyword + '\n')
        s = ''
        for i in range(arr_full.size):
            s += str(arr_full[i]) + ' '
            if (i + 1) % 6 == 0:  # write only 6 values per row
                f.write(s + '\n')
                s = ''
        f.write(s + '\n')
        f.write('/\n')
        print('Array saved to file', fname, ' (keyword ' + keyword + ')')


def make_full_cube(cube: np.array, local_to_global: np.array, global_to_local: np.array, inactive_value='min'):
    '''
    returns 1d-array of size nx*ny*nz, filled with zeros where actnum is zero
    :param cube: 1d-array of size n_active_cells
    :param actnum: 1d-array of size nx*ny*nz
    :return:
    '''
    if global_to_local.size == cube.size:
        return cube
    if inactive_value == 'min':
        inactive_value_ = cube.min()
    else:
        inactive_value_ = inactive_value
    cube_full = np.zeros(global_to_local.size) + inactive_value_
    cube_full[local_to_global] = cube
    return cube_full


def read_arrays(gridfile: str, propfile: str):
    '''
    :param gridfile: file that contains CPG grid
    :param propfile: file that contains properties defined on grid
    :return: dictionary of arrays
    '''
    # fill the dictionary to return
    arrays = {}

    dims_cpp = index_vector_discr()
    load_single_int_keyword(dims_cpp, gridfile, "SPECGRID", 3)
    arrays['SPECGRID'] = np.array(dims_cpp, copy=False)

    permx_cpp, permy_cpp, permz_cpp = value_vector_discr(), value_vector_discr(), value_vector_discr()
    load_single_float_keyword(permx_cpp, propfile, 'PERMX', -1)
    load_single_float_keyword(permy_cpp, propfile, 'PERMY', -1)
    arrays['PERMX'] = np.array(permx_cpp, copy=False)
    arrays['PERMY'] = np.array(permy_cpp, copy=False)
    for perm_str in ['PERMEABILITYXY', 'PERMEABILITY']:
        if arrays['PERMX'].size == 0 or arrays['PERMY'].size == 0:
            load_single_float_keyword(permx_cpp, propfile, perm_str, -1)
            permy_cpp = permx_cpp
            arrays['PERMX'] = np.array(permx_cpp, copy=False)
            arrays['PERMY'] = np.array(permy_cpp, copy=False)
    if arrays['PERMY'].size == 0:
        arrays['PERMY'] = arrays['PERMX']
        print('No PERMY found in input files. PERMY=PERMX will be used')
    load_single_float_keyword(permz_cpp, propfile, 'PERMZ', -1)
    arrays['PERMZ'] = np.array(permz_cpp, copy=False)
    if arrays['PERMZ'].size == 0:
        arrays['PERMZ'] = arrays['PERMX'] * 0.1
        print('No PERMZ found in input files. PERMZ=PERMX/10 will be used')
    poro_cpp = value_vector_discr()  # self.discr_mesh.poro
    load_single_float_keyword(poro_cpp, propfile, 'PORO', -1)
    arrays['PORO'] = np.array(poro_cpp, copy=False)

    coord_cpp = value_vector_discr()  # self.discr_mesh.coord
    load_single_float_keyword(coord_cpp, gridfile, 'COORD', -1)
    arrays['COORD'] = np.array(coord_cpp, copy=False)

    zcorn_cpp = value_vector_discr()  # self.discr_mesh.zcorn
    load_single_float_keyword(zcorn_cpp, gridfile, 'ZCORN', -1)
    arrays['ZCORN'] = np.array(zcorn_cpp, copy=False)

    actnum_cpp = index_vector_discr()  # self.discr_mesh.actnum
    arrays['ACTNUM'] = np.array([])
    for fname in [gridfile, propfile]:
        if arrays['ACTNUM'].size == 0:
            load_single_int_keyword(actnum_cpp, fname, 'ACTNUM', -1)
            arrays['ACTNUM'] = np.array(actnum_cpp, copy=False)
    if arrays['ACTNUM'].size == 0:
        arrays['ACTNUM'] = np.ones(arrays['SPECGRID'].prod(), dtype=np.int32)
        print('No ACTNUM found in input files. ACTNUM=1 will be used')

    return arrays

def check_arrays(arrays):
    # check dims of loaded arrays
    nx, ny, nz = arrays['SPECGRID']
    n_cells_all =  nx * ny * nz
    coord_dims = (nx + 1) * (ny + 1) * 6
    zcorn_dims = n_cells_all * 8
    for a_name in arrays.keys():
        if a_name == 'SPECGRID':
            assert arrays[a_name].shape[0] == 3, 'Error: arrray ' + a_name + ' dimensions are not correct!' + str(arrays[a_name].shape)
        elif a_name == 'COORD':
            assert arrays[a_name].shape == coord_dims, 'Error: arrray ' + a_name + ' dimensions are not correct!' + str(arrays[a_name].shape)
        elif a_name == 'ZCORN':
            assert arrays[a_name].shape == zcorn_dims, 'Error: arrray ' + a_name + ' dimensions are not correct!' + str(arrays[a_name].shape)
        else:
            assert arrays[a_name].shape == n_cells_all, 'Error: arrray ' + a_name + ' dimensions are not correct!' + str(arrays[a_name].shape)

def make_burden_layers(number_of_burden_layers: int, initial_thickness: float, property_dictionary,
                       burden_layer_prop_value=1e-5):
    """create overburden and underburden layers if the number of burden layers is not zero

    :param number_of_burden_layers: the number of burden layers, applies to overburden and underburden layers
    :type number_of_burden_layers: int
    :param layer_thickness = 10  # thickness of one layer
    :param property_dictionary: the dictionary which has different reservoir properties array
    :type property_dictionary: dict
    :param burden_layer_prop_value: the very low property values of the burden layers
    :type burden_layer_prop_value: float
    :return: a new dictionary
    :rtype: dict
    """
    if number_of_burden_layers == 0:
        return
    thickness = initial_thickness

    nx = property_dictionary['SPECGRID'][0]
    ny = property_dictionary['SPECGRID'][1]
    for i in range(0, number_of_burden_layers):
        # for each burden layer, zcorn has 4 * nx * ny number of values
        property_dictionary['ZCORN'] = np.concatenate(
            [property_dictionary['ZCORN'][:4 * nx * ny] - thickness, property_dictionary['ZCORN'][:4 * nx * ny],
             property_dictionary['ZCORN'],
             property_dictionary['ZCORN'][-4 * nx * ny:],
             property_dictionary['ZCORN'][-4 * nx * ny:] + thickness])
        # for each burden layer, poro, perm have nx * ny number of values
        for property_name in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
            property_dictionary[property_name] = np.concatenate(
                [np.ones(nx * ny) * burden_layer_prop_value, property_dictionary[property_name],
                 np.ones(nx * ny) * burden_layer_prop_value])
        # for each burden layer, actnum has nx * ny number of values
        # which are the same the values from the top reservoir layer
        property_dictionary['ACTNUM'] = np.concatenate(
            [property_dictionary['ACTNUM'][:nx * ny], property_dictionary['ACTNUM'],
             property_dictionary['ACTNUM'][-nx * ny:]])

        thickness *= 2  # increase thickness for each new layer

    # update the grid dimension in z direction for both overburden and underburden layers
    property_dictionary['SPECGRID'][-1] += 2 * number_of_burden_layers
    return property_dictionary

