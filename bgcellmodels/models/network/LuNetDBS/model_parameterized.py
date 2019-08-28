# -*- coding: utf-8 -*-
"""
Basal Ganglia network model consisting of morphologically detailed
cell models for the major cell types.

Parameterized model construction based on configuration file / dictionary.

@author     Lucas Koelman

@date       25/04/2019


USAGE
-----

Run distributed using MPI:

>>> mpirun -n 6 python <command>


Run single-threaded using IPython:

>>> ipython
>>> %run <command>


Example commands:

>>> model_parameterized.py -id testrun --dur 100 --scale 0.5 --seed 888 \
--dd --lfp --nodbs \
--outdir ~/storage \
--transientperiod 0.0 --writeinterval 1000 --reportinterval 25.0 \
--simconfig template_axon-norelay.json \
--cellconfig dummy-cells_axons-full.json \
--axonfile axon_coordinates_full.pkl \
--configdir ~/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs \
--morphdir ~/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies


>>> mpirun -n 6 python model_parameterized.py -id calibrate1 --dur 1000 --scale 1.0 --seed 888 --nodbs --nolfp --dd -dt 0.025 --outdir ~/storage --transientperiod 0.0 --writeinterval 1000 --reportinterval 25.0 --configdir ~/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs --simconfig test_simconfig.json --cellconfig test_cellconfig_5.json --axonfile axon_coordinates_cutoff.pkl --morphdir ~/workspace/bgcellmodels/bgcellmodels/models/STN/Miocinovic2006/morphologies


NOTES
-----

- It may look like some imports are not used but they may be called dynamically
  using eval() based on the config file.

"""
from __future__ import print_function
import time
import os
import cPickle as pickle
from datetime import datetime

import numpy as np

# MPI support
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size() # number of processes
mpi_rank = comm.Get_rank() # rank of current process
WITH_MPI = mpi_size > 1

# PyNN library
import pyNN.neuron as sim
from pyNN import space
from pyNN.random import RandomDistribution
from pyNN.parameters import ArrayParameter
from pyNN.utility import init_logging # connection_plot is bugged
import neo.io

# Custom PyNN extensions
# from bgcellmodels.extensions.pynn import simulator as pynn_sim_ext
from bgcellmodels.extensions.pynn import synapses as custom_synapses
from bgcellmodels.extensions.pynn.utility import connection_plot
from bgcellmodels.extensions.pynn.populations import Population
from bgcellmodels.extensions.pynn.axon_models import AxonRelayType
from bgcellmodels.extensions.pynn import spiketrains as spikegen

# NEURON models and mechanisms
from bgcellmodels.emfield import stimulation, xtra_utils
from bgcellmodels.mechanisms import synapses, noise # loads MOD files
from bgcellmodels.cellpopdata import connectivity # for use in config files

from bgcellmodels.models.STN.Miocinovic2006 import miocinovic_pynn_model as miocinovic
from bgcellmodels.models.GPe.Gunay2008 import gunay_pynn_model as gunay
from bgcellmodels.models.axon.foust2011 import AxonFoust2011

from bgcellmodels.common.configutil import eval_params
from bgcellmodels.common.stdutil import getdictvals
from bgcellmodels.common import logutils, fileutils
from bgcellmodels.morphology import morph_io

# Global variables
h = sim.h
ConnectivityPattern = connectivity.ConnectivityPattern
make_connection_list = connectivity.make_connection_list
make_divergent_pattern = connectivity.make_divergent_pattern

# Logging & debugging
logger = logutils.logging.getLogger('simulation')

logutils.setLogLevel('WARNING', [
    'Neo',
    'bpop_ext',
    'bluepyopt.ephys.parameters',
    'bluepyopt.ephys.mechanisms',
    'bluepyopt.ephys.morphologies',
    'AxonBuilder',
    'PyNN.simulator'])

logutils.setLogLevel('DEBUG', ['simulation'])

h("XTRA_VERBOSITY = 0")

# Translation vectors between Waxholm data DWI and atlas-v1 space
dwi2anat_vec_mm = np.array([-20.01319, -10.01633, -10.01622]) # from manual
blend2anat_vec_um = np.array([-15.9514e3, -8.43968e3, -7.71387e3]) # in Blender file
fem2blend_vec_um = np.array([19935.4375, 9846.599609375, 12519.98046875]) # Karthik -> Blend
gpi_center_um = np.array([[18533.4921875, 5821.53759765625, 7248.1533203125]])


def make_stn_lateral_connlist(pop_size, num_adjacent, fraction, rng):
    """
    Make connection list for STN lateral connectivity.

    @param  pop_size : int
            Number of cells in STN population.

    @param  fraction : float
            Fraction of STN neurons that project laterally to neighbors.

    @param  num_adjacent : int
            Number of neighbors on each side to project to.

    @param  rng : numpy.Random
            Random number generator.

    @return conn_list : list(list[int, int])
            Connection list with [source, target] pairs.
    """
    if fraction == 0 or num_adjacent == 0:
        return []
    source_ids = rng.choice(range(pop_size), int(fraction*pop_size), replace=False)
    targets_relative = range(1, num_adjacent+1) + range(-1, -num_adjacent-1, -1)
    return make_divergent_pattern(source_ids, targets_relative, pop_size)



def write_population_data(pop, output, suffix, gather=True, clear=True):
    """
    Write recorded data for Population to file.

    @param  output : str
            Output path including asterisk as placeholder: "/path/to/*.ext"

    @note   gathers data from MPI nodes so should be executed on all ranks.
    """
    if output is None:
        return
    outdir, extension = output.split('*')
    # Get Neo IO writer for file format associated with extension
    if extension.endswith('h5'):
        IOClass = neo.io.NixIO
    elif extension.endswith('mat'):
        IOClass = neo.io.NeoMatlabIO
    elif extension.endswith('npz'):
        IOClass = neo.io.PyNNNumpyIO
    else:
        IOClass = str # let PyNN guess from extension
    outfile =  "{dir}{label}{suffix}{ext}".format(dir=outdir,
                    label=pop.label, suffix=suffix, ext=extension)
    io = IOClass(outfile)
    pop.write_data(io, variables='all', gather=gather, clear=clear,
                       annotations={'script_name': __file__})


def write_compartment_coordinates(pop, out_dir, scale=1.0, translation=None,
                                  activating_function=False):
    """
    Write compartment coordinates of all cells in population to file.
    """
    # Cell GID and all sections
    gid2seclist = [
        (int(gid), gid._cell.get_all_sections()) for gid in pop if pop.is_local(gid)
    ]
    cell_ids, cell_seclists = zip(*gid2seclist)

    # Vertices and edges for each cell
    verts, edges = morph_io.morphologies_to_edges(cell_seclists,
        flatten_cells=False, scale=scale, translation=translation)

    # Activating function values for each cell
    cells = [gid._cell for gid in pop if pop.is_local(gid)]
    cells_activations_dists = [
        xtra_utils.get_rattay_activating_function(cell.icell,
            'basal', 'somatic', 'axonal') for cell in cells
    ]
    cell_acts, cell_dists = zip(*cells_activations_dists)

    # Gather this data from all MPI ranks
    this_rank_data = {
        'gids': cell_ids,
        'comp_locs': verts, 'comp_edges': edges,
        'comp_act': cell_acts, 'comp_dists': cell_dists,
    }
    if WITH_MPI:
        all_rank_data = comm.allgather(this_rank_data)
    else:
        all_rank_data = [this_rank_data]

    # Combine all data into one dictionary for exporting
    out_dict = {k: [] for k in this_rank_data.keys()}
    out_dict['population'] = pop.label
    for rank_data in all_rank_data:
        for k in rank_data.keys():
            out_dict[k].extend(rank_data[k])

    # Write to file
    if mpi_rank == 0:
        scale_units = ('um', 'mme-2', 'mme-1', 'mm',  'cm', 'dm', 'm')
        scale_index = int(-np.log10(scale))
        if 0 <= scale_index < len(scale_units):
            scale_suffix = scale_units[scale_index]
        else:
            scale_suffix = str(scale)
        space_suffix = 'dwi' if translation is None else 'anat'

        out_filename = 'comp-locs_{}_space-{}_scale-{}.pkl'.format(
            pop.label, space_suffix, scale_suffix)
        out_filepath = os.path.join(out_dir, out_filename)

        # Write as pickle with vertices grouped by cell
        logger.debug('Population %s: writing compartment coordinates.', pop.label)
        with open(out_filepath, 'wb') as fout:
            pickle.dump(out_dict, fout)

        # Write as PLY into flat datastructure
        out_filepath = out_filepath[:-4] + '.ply'
        morph_io.edges_to_PLY(out_dict['comp_locs'], out_dict['comp_edges'],
            out_filepath, multiple=True)

        # Write as TXT in units of mm
        out_filepath = out_filepath[:-4] + '.txt'
        all_vertices = np.array([v for cell_verts in out_dict['comp_locs'] for v in cell_verts])
        all_vertices += blend2anat_vec_um
        morph_io.coordinates_um_to_txt(all_vertices, out_filepath, scale=1e-3)


def simulate_model(
        pop_scale       = 1.0,
        sim_dur         = 500.0,
        sim_dt          = None,
        export_locals   = True,
        output          = None,
        report_progress = None,
        net_conf        = None,
        cell_conf       = None,
        fem_conf        = None,
        axon_coordinates    = None,
        morph_dir       = None,
        seed            = None,
        with_lfp        = None,
        with_dbs        = None,
        dopamine_depleted   = None,
        transient_period    = None,
        max_write_interval  = None,
        report_interval = 50.0,
        restore_state   = None,
        save_state      = None,
        export_compartment_coordinates = None,
        **kwargs):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.

    @param      output : str (optional)
                File path to save recordings at in following format:
                '~/storage/*.mat'


    @param      config : dict
                Dictionary with one entry per population label and one
                key 'simulation' for simulation parameters.
    """
    ############################################################################
    # READ CONFIGURATIONS
    ############################################################################

    out_dir, out_ext = output.split('*')

    sim_params = net_conf['simulation']
    emf_params = net_conf['electromagnetics']

    with_dbs = emf_params['with_dbs'] if with_dbs is None else with_dbs
    with_lfp = emf_params['with_lfp'] if with_lfp is None else with_lfp

    # Badstubner (2017), Fig. 8: 750 Ohm*cm
    # Baumanm (2010): 370 Ohm*cm
    rho_ohm_cm = 1.0 / (emf_params['sigma_extracellular_S/m'] * 1e-2)
    electrode_tip_point_um = emf_params.get('dbs_electrode_tip_point_um',
                                emf_params.get('dbs_electrode_coordinates_um', None))
    transfer_impedance_matrix = [] if fem_conf is None else fem_conf

    def intersects_encapsulation_layer(point):
        """ Check whether points is inside encapsulation layer """
        r_encap = emf_params['dbs_encapsulation_radius_um']
        p1 = np.array(electrode_tip_point_um)
        p2 = np.array(emf_params['dbs_electrode_axis_point_um'])
        p3 = np.array(point)

        v12 = p2 - p1
        v13 = p3 - p1

        # Simplest test: within tip radius
        if np.sqrt(np.dot(v13, v13)) <= r_encap:
            return True

        # Find point on electrode axis closest to test point
        u = np.dot(v12, v13) / np.dot(v12, v12)
        p_isect = p1 + u * v12
        v_i3 = p3 - p_isect
        ax_dist = np.sqrt(np.dot(v_i3, v_i3))

        if u >= 0:
            # Axis point is in cylindrical region
            return ax_dist <= r_encap

        # Axis point is below cylindrical region and we are not within tip radius
        return False



    ############################################################################
    # SIMULATOR SETUP
    ############################################################################

    if sim_dt is None:
        sim_dt = sim_params['timestep']
    else:
        logger.warning("Simulation timestep overridden from command line: dt = %f", sim_dt)

    sim.setup(timestep=sim_dt,
              min_delay=0.1, max_delay=10.0, use_cvode=False)

    if mpi_rank == 0:
        init_logging(logfile=None, debug=True)

    print("\nThis is rank {} (node {} of {})".format(
          sim.rank(), sim.rank() + 1, sim.num_processes()))

    sim.state.duration = sim_dur # not used by PyNN, only by our custom funcs
    sim.state.rec_dt = sim_params['recording_timestep']
    sim.state.mcellran4_rng_indices = {} # Keep track of MCellRan4 indices for independent random streams.
    finit_handlers = []

    # Make one random generator that is shared and should yield same results
    # for each MPI rank, and one with unique results.
    # - The shared (parallel-safe) RNGs should be used in functions that are
    #   executed on all ranks, like instantiating Population and Projection
    #   objects.
    # - The default RNG for Connectors is NumpyRNG(seed=151985012)
    if seed is None:
        seed = sim_params['shared_rng_seed']

    # Make RNG seeds accessible
    sim.state.shared_rng_seed = shared_seed = seed # original: 151985012
    sim.state.rank_rng_seed = rank_seed = sim.state.native_rng_baseseed + sim.state.mpi_rank

    # RNGs that can be passed to PyNN objects like Connector subclasses
    # Store them on simulator.state so we can access from other custom classes
    sim.state.shared_rng = shared_rng_pynn = sim.NumpyRNG(seed=shared_seed)
    sim.state.rank_rng = rank_rng_pynn = sim.NumpyRNG(seed=rank_seed)

    # Raw Numpy RNGs (numpy.random.RandomState) to be used in our own functions
    shared_rng = shared_rng_pynn.rng
    rank_rng = rank_rng_pynn.rng

    # Global physiological conditions
    DD = dopamine_depleted
    if DD is None:
        DD = dopamine_depleted = sim_params.get('DD', None)
    if DD is None:
        raise ValueError("Dopamine depleted condition not specified "
                         "in config file nor as simulation argument.")
    if mpi_rank == 0:
        print("Simulation settings are:\n"
              "    - DD = {}\n"
              "    - sim_dt = {}\n"
              "    - with_dbs = {}\n"
              "    - with_lfp = {}".format(DD, sim_dt, with_dbs, with_lfp))


    ############################################################################
    # LOCAL FUNCTIONS
    ############################################################################

    params_global_context = globals()
    params_local_context = locals() # capture function arguments

    def get_pop_parameters(pop, *param_names):
        """
        Get population parameters from config and evaluate them.
        """
        config_locals = net_conf[pop].get('local_context', {})
        param_specs = getdictvals(net_conf[pop], *param_names, as_dict=True)
        pvals = eval_params(param_specs, params_global_context,
                            [params_local_context, config_locals])
        return getdictvals(pvals, *param_names)

    def get_param_group(pop, group_name=None, mapping=None):
        """
        Get a group of parameters for a population as dictionary.
        """
        config_locals = net_conf[pop].get('local_context', {})
        if group_name is None:
            param_specs = net_conf[pop]
        else:
            param_specs = net_conf[pop][group_name]
        if mapping is not None:
            param_specs = {v: param_specs[k] for k,v in mapping.iteritems()}
        return eval_params(param_specs, params_global_context,
                           [params_local_context, config_locals])

    def get_cell_parameters(pop):
        """
        Get PyNN cell parameters as dictionary of numerical values.
        """
        config_locals = net_conf[pop].get('local_context', {})
        param_specs = net_conf[pop].get('PyNN_cell_parameters', {})
        return eval_params(param_specs, params_global_context,
                           [params_local_context, config_locals])

    def synapse_from_config(pre, post):
        """
        Make Synapse object from config dict
        """
        config_locals = net_conf[post].get('local_context', {})
        syn_type, syn_params = getdictvals(net_conf[post][pre]['synapse'],
                                           'name', 'parameters')
        if hasattr(custom_synapses, syn_type):
            syn_class = getattr(custom_synapses, syn_type)
        else:
            syn_class = getattr(sim, syn_type)
        syn_pvals = eval_params(syn_params, params_global_context,
                                [params_local_context, config_locals])
        num_contacts = net_conf[post][pre].get('num_contacts', 1)
        syntype_obj = syn_class(**syn_pvals)
        syntype_obj.num_contacts = num_contacts
        return syntype_obj

    def connector_from_config(pre, post, rng=None):
        """
        Make Connector object from config dict
        """
        config_locals = net_conf[post].get('local_context', {})
        con_type, con_params = getdictvals(net_conf[post][pre]['connector'],
                                           'name', 'parameters')
        connector_class = getattr(sim, con_type)
        con_pvals = eval_params(con_params, params_global_context,
                               [params_local_context, config_locals])
        connector = connector_class(**con_pvals)
        if rng is not None:
            connector.rng = rng
        return connector

    axon_scales = { 'mm' : 1.0, 'um': 1e-3, 'm': 1e3 }
    axon_scale = axon_scales[cell_conf['units']['axons']]

    def get_axon_coordinates(cell_def):
        """
        Get axon coordinates for cell definition
        """
        axon_spec = cell_def['axon'].split('*')

        # No glob pattern : axon is axon identifier
        if len(axon_spec) == 1:
            return np.asarray(axon_coordinates[cell_def['axon']]) * axon_scale

        # With glob pattern: axon spec matches group
        candidate_axons = [
            np.asarray(coords) * axon_scale for ax_id, coords in axon_coordinates.iteritems() if ax_id.startswith(axon_spec[0])
        ]

        # Find axon with closest end to cell center
        cell_coords = cell_def['transform'][0:3, 3]
        min_dist = 1e12
        closest_axon = None
        for ax_coords in candidate_axons:
            for i in 0, -1:
                dist_end = np.linalg.norm(cell_coords - ax_coords[i])
                if dist_end < min_dist:
                    min_dist = dist_end
                    closest_axon = ax_coords

        return closest_axon

    def get_morphology_path(morphology_id, default_morphology=None):
        """
        Get morphology file path from morphology name.
        """
        if morphology_id is None:
            if default_morphology is None:
                return ValueError('No default morphology for empty morphology.')
            morphology_id = default_morphology
        return os.path.join(morph_dir, morphology_id + '.swc')


    # Set NEURON integrator/solver options
    # if emfield_rec and not emfield_stim:
    #     sim.state.cvode.use_fast_imem(True)
    # sim.state.cvode.cache_efficient(True) # necessary for fast_imem lfp + 33% reduction in simulation time

    # Parameters to be saved to pickle file
    saved_params = {'dopamine_depleted': DD}

    ############################################################################
    # POPULATIONS
    ############################################################################
    # Define each cell population with its cell type, number of cells
    # NOTE:
    # - to query cell model attributes, use population[i]._cell
    print("rank {}: starting phase POPULATIONS.".format(mpi_rank))

    config_pop_labels = [k for k in net_conf.keys() if not k in
                            ('simulation', 'electromagnetics')]

    # 3D info for cell positioning
    gpi_cell_positions = [
        np.array(cell['transform'])[0:3, 3].reshape(-1,3) for cell in cell_conf['cells'] if
            (cell['population'] == 'GPI')
    ]

    #===========================================================================
    # STN POPULATION

    stn_ncell_base, = get_pop_parameters('STN', 'base_population_size')
    stn_ncell_biophys = int(stn_ncell_base * pop_scale)

    #---------------------------------------------------------------------------
    # STN cell model

    # List all candidate cell definitions
    all_cell_defs = [
        cell for cell in cell_conf['cells'] if (cell['population'] == 'STN')
    ]
    for cell in all_cell_defs:
        cell['transform'] = np.asarray(cell['transform'])

    # Select cells that are eligible given electrode position
    cell_defs = [
        cell for cell in all_cell_defs if 
            not intersects_encapsulation_layer(cell['transform'][0:3, 3])
    ]
    assert len(cell_defs) >= stn_ncell_biophys
    cell_defs = cell_defs[:stn_ncell_biophys]

    # Save positions for later
    stn_cell_positions = np.array([cell['transform'][0:3, 3] for cell in cell_defs]) # Nx3

    # Choose a random morphology for each cell
    candidate_morphologies = np.array(cell_conf['default_morphologies']['STN'])
    morph_indices = np.array(net_conf['STN']['morphology_indices'][:stn_ncell_biophys]) % len(candidate_morphologies)
    cells_morph_paths = [
        get_morphology_path(m) for m in candidate_morphologies[morph_indices]
    ]

    # Axons and collaterals
    cells_axon_coords = [get_axon_coordinates(cell) for cell in cell_defs]
    stn_collateralization_fraction = 7.0/8.0 # Koshimizu (2013): 7/8 Gpe-projecting neurons
    stn_collat_nbranch = np.zeros((stn_ncell_biophys, 2), dtype=int)
    stn_collat_nbranch[:, 0] = 1 # one branch at first branch point
    stn_collat_nbranch[:, 1] = 2 # two branches at second branch point
    # Select cells without collaterals
    no_collat_idx = shared_rng.choice(stn_ncell_biophys,
                        int((1.0-stn_collateralization_fraction)*stn_ncell_biophys),
                        replace=False)
    stn_collat_nbranch[no_collat_idx, 0] = 0
    stn_collat_nbranch = [nbr.reshape((-1, 2)) for nbr in stn_collat_nbranch]

    # Load default parameters from sim config
    stn_cell_params = get_cell_parameters('STN')

    # Add parameters from other sources
    stn_cell_params['with_extracellular_stim'] = with_dbs and net_conf['STN'].get('with_dbs', True)
    stn_cell_params['with_extracellular_rec'] = with_lfp and net_conf['STN'].get('with_lfp', True)
    # stn_cell_params['seclists_with_dbs'] = np.array('axonal')
    stn_cell_params['morphology_path'] = cells_morph_paths
    stn_cell_params['transform'] = [cell['transform'] for cell in cell_defs]
    ## Axon parameters
    stn_cell_params['streamline_coordinates_mm'] = cells_axon_coords
    stn_cell_params['collateral_branch_points_um'] = gpi_cell_positions[:stn_ncell_biophys]
    stn_cell_params['collateral_target_points_um'] = gpi_cell_positions[:stn_ncell_biophys]
    stn_cell_params['collateral_lvl_lengths_um'] = ArrayParameter(np.array([[100.0, 100.0]]))
    stn_cell_params['collateral_lvl_num_branches'] = stn_collat_nbranch
    ## DBS parameters
    stn_cell_params['rho_extracellular_ohm_cm'] = rho_ohm_cm
    stn_cell_params['electrode_coordinates_um'] = electrode_tip_point_um
    stn_cell_params['transfer_impedance_matrix_um'] = ArrayParameter(transfer_impedance_matrix)


    stn_type = miocinovic.StnMorphType(**stn_cell_params)

    # Trace back cell indices to morphology definitions:
    if mpi_rank == 0:
        for param_name in 'morphology_path', 'transform':
            saved_params.setdefault('STN', {})[param_name] = stn_cell_params[param_name]
        saved_params.setdefault('STN', {})['cell_defs'] = cell_defs
        # logger.debug("Mapping of STN cell indices to morphologies is:\n" +
        #     "\n".join(("{}: {}\n".format(i, c) for i,c in enumerate(cells_morph_paths))))
        # logger.debug("Mapping of STN cell indices to transforms is:\n" +
        #     "\n".join(("{}: {}\n".format(i, c) for i,c in enumerate(cells_transforms))))

    #---------------------------------------------------------------------------
    # STN population

    # Grid structure for calculating connectivity
    stn_dx, = get_pop_parameters('STN', 'grid_dx')
    stn_grid = space.Line(x0=0.0, dx=stn_dx, y=0.0, z=0.0)

    # Initial values for state variables
    vinit = stn_type.default_initial_values['v']
    initial_values = {
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    pop_stn = Population(stn_ncell_biophys,
                         cellclass=stn_type,
                         label='STN',
                         structure=stn_grid,
                         initial_values=initial_values)


    # Export 3D coordinates of compartment centers
    if export_compartment_coordinates:
        write_compartment_coordinates(pop_stn, out_dir, activating_function=with_dbs)

    # # Check coordinates
    # stn_gpe_all_nodes = []
    # for cell_id in pop_stn:
    #     model = cell_id._cell

    #     # Save compartment centers
    #     all_centers, all_n3d = morph_3d.get_segment_centers([model.get_all_sections()], samples_as_rows=True)
    #     stn_gpe_all_nodes.extend(all_centers)

    #     # Export cell
    #     morph_io.morphology_to_PLY([model.get_all_sections()],
    #         'STN_after_{}.ply'.format(int(cell_id)),
    #         segment_centers=True)

    #     # Check coordinates
    #     sec_start = np.array([[h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)]
    #                             for sec in model.icell.all])
    #     node_xyz = np.array(all_centers)
    #     ref_coord = [17113.8, 6340.13, 6848.05]
    #     node_diffs = node_xyz - ref_coord

    #     node_dists = np.linalg.norm(node_diffs, axis=1)
    #     if any(node_dists > 5000.0):
    #         raise Exception('too far from STN center! Check morphology')

    #---------------------------------------------------------------------------
    # STN Surrogate spike sources

    frac_surrogate, surr_rate = get_pop_parameters('STN',
        'surrogate_fraction', 'surrogate_rate')

    ncell_surrogate = int(stn_ncell_biophys * frac_surrogate)
    if ncell_surrogate > 0:
        pop_stn_surrogate = Population(ncell_surrogate,
                                       sim.SpikeSourcePoisson(rate=surr_rate),
                                       label='STN.surrogate')
        asm_stn = sim.Assembly(pop_stn, pop_stn_surrogate,
                               label='STN.all')
    else:
        asm_stn = sim.Assembly(pop_stn, label='STN.all')

    stn_pop_size = asm_stn.size

    #===========================================================================
    # GPE POPULATION (prototypic)

    gpe_ncell_base, frac_proto = get_pop_parameters('GPE.all',
                        'base_population_size', 'prototypic_fraction')
    gpe_ncell_biophys = int(gpe_ncell_base *frac_proto * pop_scale)

    #---------------------------------------------------------------------------
    # GPe cells parameters

    # Select cells to simulate
    pop_cell_defs = [
        cell for cell in cell_conf['cells'] if
            (cell['population'] == 'GPE') and (cell['axon'] is not None)
    ]
    cell_defs = pop_cell_defs[:gpe_ncell_biophys]

    # Get 3D morphology properties of each cell
    cells_transforms = [np.asarray(cell['transform']) for cell in cell_defs]

    # GPe axons and collaterals
    cells_axon_coords = [get_axon_coordinates(cell['axon']) for cell in cell_defs]
    gpe2gpi_collateralization = 0.5 # Kita & Kitai (1994): two out of four

    ## Number of branches at each level of the collateral, for each branch point
    gpe2gpi_nbranch = np.zeros((gpe_ncell_biophys, 2), dtype=int)
    gpe2gpi_nbranch[:, 0] = 1 # one branch at first branch point
    gpe2gpi_nbranch[:, 1] = 2 # two branches at second branch point
    no_collat_num = int((1.0-gpe2gpi_collateralization) * gpe_ncell_biophys)
    no_collat_idx = shared_rng.choice(gpe_ncell_biophys, no_collat_num, replace=False)
    gpe2gpi_nbranch[no_collat_idx, 0] = 0 # cells without collaterals
    gpe2stn_nbranch = [1, 2]
    gpe_collat_nbranch = [
        np.concatenate((gpe2gpi.reshape((-1, 2)), [gpe2stn_nbranch]), axis=0)
            for gpe2gpi in gpe2gpi_nbranch
    ]
    gpe2gpi_collat_lengths = [100.0, 100.0] # 100 um in Johson & McIntyre (2008)
    gpe2stn_collat_lengths = [250.0, 250.0]
    gpe_collat_lvl_lengths = np.array([gpe2gpi_collat_lengths, gpe2stn_collat_lengths])

    ## Branch points and target points for each collateral
    gpi_targets = gpi_cell_positions[:gpe_ncell_biophys]
    stn_targets = stn_cell_positions[shared_rng.choice(stn_ncell_biophys, gpe_ncell_biophys)]
    gpe_collat_branch_pts = [
        np.concatenate((gpi_pos, stn_pos.reshape((-1,3))), axis=0) for (gpi_pos, stn_pos) in zip(
            gpi_targets, stn_targets)
    ]


    # Load default parameters from sim config
    gpe_cell_params = get_cell_parameters('GPE.all')

    # Add parameters from other sources
    gpe_cell_params['with_extracellular_stim'] = with_dbs and net_conf['GPE.all'].get('with_dbs', True)
    gpe_cell_params['with_extracellular_rec'] = with_lfp and net_conf['GPE.all'].get('with_lfp', True)
    gpe_cell_params['transform'] = cells_transforms
    ## Axon parameters
    gpe_cell_params['termination_method'] = np.array('nodal_cutoff')
    gpe_cell_params['netcon_source_spec'] = np.array('branch_point:1:collateral')
    gpe_cell_params['streamline_coordinates_mm'] = cells_axon_coords
    gpe_cell_params['collateral_branch_points_um'] = gpe_collat_branch_pts
    gpe_cell_params['collateral_target_points_um'] = gpe_collat_branch_pts
    gpe_cell_params['collateral_lvl_lengths_um'] = ArrayParameter(gpe_collat_lvl_lengths)
    gpe_cell_params['collateral_lvl_num_branches'] = gpe_collat_nbranch
    ## FEM parameters
    gpe_cell_params['rho_extracellular_ohm_cm'] = rho_ohm_cm
    gpe_cell_params['electrode_coordinates_um'] = electrode_tip_point_um
    gpe_cell_params['transfer_impedance_matrix_um'] = ArrayParameter(transfer_impedance_matrix)

    proto_type = gunay.GpeProtoCellType(**gpe_cell_params)

    #---------------------------------------------------------------------------
    # GPe prototypic population

    # Get common parameters for GPE cells
    gpe_dx, frac_proto, = get_pop_parameters('GPE.all',
                            'grid_dx', 'prototypic_fraction',)

    # Grid structure for calculating connectivity
    gpe_grid = space.Line(x0=0.0, dx=gpe_dx,
                          y=1e6, z=0.0)

    # Initial values for state variables
    vinit = proto_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    pop_gpe_proto = Population(gpe_ncell_biophys,
                               cellclass=proto_type,
                               label='GPE.proto',
                               structure=gpe_grid,
                               initial_values=initial_values)

    # Export 3D coordinates of compartment centers
    if export_compartment_coordinates:
        write_compartment_coordinates(pop_gpe_proto, out_dir, activating_function=with_dbs)

    # # Check coordinates
    # for cell_id in pop_gpe_proto:
    #     model = cell_id._cell

    #     # Save compartment centers
    #     all_centers, all_n3d = morph_3d.get_segment_centers([model.get_all_sections()], samples_as_rows=True)
    #     stn_gpe_all_nodes.extend(all_centers)


    # # Write out analytical Ztransfer
    # def impedance_func(xyz):
    #     x1, y1, z1 = xyz
    #     x2, y2, z2 = electrode_tip_point_um
    #     dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    #     if dist == 0:
    #         dist = 0.5 # seg.diam / 2

    #     # 0.01 converts rho's cm to um and ohm to megohm
    #     return (rho_ohm_cm / 4 / np.pi) * (1 / dist) * 0.01

    # stn_gpe_Zmat = np.array(
    #     [(xyz[0], xyz[1], xyz[2], impedance_func(xyz))
    #         for xyz in stn_gpe_all_nodes])

    # np.save('transfer_impedance_matrix.npy', stn_gpe_Zmat)

    #---------------------------------------------------------------------------
    # GPE surrogate population

    frac_surrogate, surr_rate = get_pop_parameters('GPE.all',
        'surrogate_fraction', 'surrogate_rate')

    ncell_surrogate = int(gpe_ncell_base * pop_scale * frac_surrogate)
    if ncell_surrogate > 0:
        pop_gpe_surrogate = Population(ncell_surrogate,
                                       sim.SpikeSourcePoisson(rate=surr_rate),
                                       label='GPE.surrogate')
    else:
        pop_gpe_surrogate = None

    #---------------------------------------------------------------------------
    # GPE axon population (relay)

    # num_gpe_axons = (gpe_ncell_biophys + ncell_surrogate)

    # # NOTE: can use any axon per cell, since they are not electrically connected
    # gpe_conn_defs = [
    #     c for c in cell_conf['connections'] if (c['projection'] == 'GPE-STN')
    # ]

    # gpe_axon_coords = [
    #     get_axon_coordinates(connection['axon']) for connection in gpe_conn_defs
    # ][:num_gpe_axons]

    # # Trace back cell indices to axon definitions used:
    # if mpi_rank == 0:
    #     saved_params.setdefault('GPE.axons', {})['axon_definitions'] = gpe_conn_defs[:num_gpe_axons]

    # # Get axon associated with cell (not necessary if no electrical connection)
    # # gpe_axon_coords = [
    # #     get_axon_coordinates(cell['axon']) for cell in pop_cell_defs if
    # #         (cell['axon'] is not None)
    # # ][:num_gpe_axons]

    # # Re-use axon definitions if insufficient
    # while len(gpe_axon_coords) < num_gpe_axons:
    #     num_additional = num_gpe_axons - len(gpe_axon_coords)
    #     gpe_axon_coords += gpe_axon_coords[:num_additional]
    #     logger.warning('GPe: Re-using %d axon definitions', num_additional)

    # # Cell type for axons
    # gpe_axon_params = {
    #     'axon_class':                   AxonFoust2011,
    #     'streamline_coordinates_mm':    gpe_axon_coords,
    #     'termination_method':           np.array('terminal_sequence'),
    #     'with_extracellular':           with_lfp or with_dbs,
    #     'electrode_coordinates_um' :    electrode_tip_point_um,
    #     'rho_extracellular_ohm_cm' :    rho_ohm_cm,
    # }

    # gpe_axon_type = AxonRelayType(**gpe_axon_params)

    # # Initial values for state variables
    # vinit = gpe_axon_type.default_initial_values['v']
    # initial_values={
    #     'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    # }

    # pop_gpe_axons = Population(num_gpe_axons,
    #                            cellclass=gpe_axon_type,
    #                            label='GPE.axons',
    #                            initial_values=initial_values)

    #---------------------------------------------------------------------------
    # GPE Assembly (all GPe subtypes)

    if pop_gpe_surrogate is None:
        asm_gpe = sim.Assembly(pop_gpe_proto, pop_gpe_surrogate, label='GPE.all')
    else:
        asm_gpe = sim.Assembly(pop_gpe_proto, label='GPE.all')
    gpe_pop_size = asm_gpe.size


    #===========================================================================
    # CTX POPULATION

    #---------------------------------------------------------------------------
    # CTX SPIKE GENERATORS
    ctx_pop_size, = get_pop_parameters('CTX', 'base_population_size')
    ctx_burst_params = get_param_group('CTX', 'spiking_pattern')
    spikegen_name = ctx_burst_params.pop('algorithm')
    spikegen_func = getattr(spikegen, spikegen_name)

    ctx_spike_generator = spikegen_func(duration=sim_dur,
                                        rng=rank_rng,
                                        **ctx_burst_params)

    ctx_ncell = int(ctx_pop_size * pop_scale)
    pop_ctx = Population(
        ctx_ncell,
        cellclass=sim.SpikeSourceArray(spike_times=ctx_spike_generator),
        label='CTX')

    #---------------------------------------------------------------------------
    # CTX AXONS

    num_ctx_axons = ctx_ncell

    ctx_conn_defs = [
        c for c in cell_conf['connections'] if (c['projection'] == 'CTX-STN')
    ]

    num_axon_reused = 0
    while len(ctx_conn_defs) < num_ctx_axons:
        num_additional = num_ctx_axons - len(ctx_conn_defs)
        ctx_conn_defs += ctx_conn_defs[:num_additional]
        num_axon_reused += num_additional
    logger.warning('CTX: Re-using %d axon definitions', num_axon_reused)

    # Trace back cell indices to axon definitions used:
    if mpi_rank == 0:
        saved_params.setdefault('CTX.axons', {})['axon_definitions'] = ctx_conn_defs

    ctx_axon_coords = [
        get_axon_coordinates(connection['axon']) for connection in ctx_conn_defs
    ]


    # Cell type for axons
    ctx_axon_params = {
        'axon_class':                   AxonFoust2011,
        'streamline_coordinates_mm':    ctx_axon_coords,
        'termination_method':           np.array('any_cutoff'),
        'with_extracellular_stim':      with_dbs and net_conf['CTX.axons'].get('with_dbs', True),
        'with_extracellular_rec':       with_lfp and net_conf['CTX.axons'].get('with_lfp', True),
        'electrode_coordinates_um' :    electrode_tip_point_um,
        'rho_extracellular_ohm_cm' :    rho_ohm_cm,
    }

    # Axon collateral parameters
    cst_stn_branch_points = [
        stn_cell_positions[i].reshape((-1,3)) for i in shared_rng.choice(
            stn_ncell_biophys, num_ctx_axons)
    ]
    ctx_axon_params['collateral_branch_points_um'] = cst_stn_branch_points
    ctx_axon_params['collateral_target_points_um'] = cst_stn_branch_points
    ctx_axon_params['collateral_lvl_lengths_um'] = ArrayParameter(np.array([[250.0, 250.0]]))
    ctx_axon_params['collateral_lvl_num_branches'] = [np.array([[1,2]]) for i in range(num_ctx_axons)]
    ctx_axon_params['netcon_source_spec'] = np.array('branch_point:0:collateral')

    ctx_axon_type = AxonRelayType(**ctx_axon_params)

    # Initial values for state variables
    vinit = ctx_axon_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    pop_ctx_axons = Population(num_ctx_axons,
                               cellclass=ctx_axon_type,
                               label='CTX.axons',
                               initial_values=initial_values)

    # Export 3D coordinates of compartment centers
    if export_compartment_coordinates:
        write_compartment_coordinates(pop_ctx_axons, out_dir, activating_function=with_dbs)

    #===========================================================================
    # STR.MSN POPULATION

    # STR.MSN spike sources
    msn_pop_size, = get_pop_parameters(
        'STR.MSN', 'base_population_size')

    msn_burst_params = get_param_group('STR.MSN', 'spiking_pattern')
    spikegen_name = msn_burst_params.pop('algorithm')
    spikegen_func = getattr(spikegen, spikegen_name)
    msn_spike_generator = spikegen_func(duration=sim_dur,
                                        rng=rank_rng,
                                        **msn_burst_params)

    pop_msn = Population(
        int(msn_pop_size * pop_scale),
        cellclass=sim.SpikeSourceArray(spike_times=msn_spike_generator),
        label='STR.MSN')


    ############################################################################
    # EXTRACELLULAR FIELD
    ############################################################################

    if with_dbs:

        # Create DBS waveform
        pulse_train, pulse_time = stimulation.make_pulse_train(
                                    frequency=emf_params['dbs_frequency_hz'],
                                    phase_deg=emf_params.get('dbs_phase_deg', 0.0),
                                    pulse_width_ms=emf_params['dbs_pulse_width_ms'],
                                    amp0=emf_params['dbs_pulse0_amplitude_mA'],
                                    amp1=emf_params['dbs_pulse1_amplitude_mA'],
                                    dt=emf_params['dbs_sample_period_ms'],
                                    duration=sim_dur,
                                    off_intervals=emf_params['dbs_off_intervals'],
                                    coincident_discontinuities=True)

        # Play DBS waveform into GLOBAL variable for this thread
        pulse_avec = h.Vector(pulse_train)
        pulse_tvec = h.Vector(pulse_time)
        dbs_started = False
        for sec in h.allsec():
            if h.ismembrane('xtra', sec=sec):
                pulse_avec.play(h._ref_is_xtra, pulse_tvec, 1)
                dbs_started = True
                break

        if not dbs_started:
            raise Exception('Could not find mechanism "xtra" in any section.')

    ############################################################################
    # CONNECTIONS
    ############################################################################

    # All populations by label
    all_pops = {pop.label : pop for pop in Population.all_populations}
    all_asm = {asm.label: asm for asm in (asm_gpe,)}

    # All projections by population label
    all_proj = {pop.label : {} for pop in Population.all_populations}
    all_proj[asm_gpe.label] = {} # add Assembly projections manually

    # Make distinction between 'real' and surrogate subpopulations
    # (note: NativeCellType is common base class for all NEURON cells)
    biophysical_pops = [
        pop for pop in Population.all_populations if
            isinstance(pop.celltype, sim.cells.NativeCellType)
            and not isinstance(pop.celltype, AxonRelayType)
    ]

    artificial_pops = [pop for pop in Population.all_populations if not isinstance(
                        pop.celltype, sim.cells.NativeCellType)]

    # Update local context for eval() statements
    params_local_context.update(locals())

    # Make all Projections directly from (pre, post) pairs in config
    for post_label, pop_config in net_conf.iteritems():

        # Get PRE Population from label
        if post_label in all_pops.keys():
            post_pop = all_pops[post_label]
        elif post_label in all_asm.keys():
            post_pop = all_asm[post_label]
        else:
            continue
        print("rank {}: starting phase {} AFFERENTS.".format(mpi_rank, post_label))

        # Create one Projection per post-synaptic population/assembly
        for pre_label in pop_config.keys():
            # get pre-synaptic Population
            if pre_label in all_pops.keys():
                pre_pop = all_pops[pre_label]
            elif pre_label in all_asm.keys():
                pre_pop = all_asm[pre_label]
            else:
                continue
            proj_config = pop_config[pre_label]

            # make PyNN Projection
            all_proj[pre_label][post_label] = sim.Projection(
                pre_pop, post_pop,
                connector=connector_from_config(pre_label, post_label, rng=shared_rng_pynn),
                synapse_type=synapse_from_config(pre_label, post_label),
                receptor_type=proj_config['receptor_type'])

    #---------------------------------------------------------------------------
    # Post-constructional modifications

    # Reduce dendritic branching and number of GLU synapses in DD
    num_prune = net_conf['STN'].get('prune_dendritic_GLUR', 0)
    if DD and num_prune > 0:
        # PD: dendritic AMPA & NMDA-NR2B/D afferents pruned
        num_disabled = np.zeros(pop_stn.size)
        for conn in all_proj['CTX']['STN'].connections:
            if num_disabled[conn.postsynaptic_index] < num_prune:
                conn.GLUsyn_gmax_AMPA = 0.0
                conn.GLUsyn_gmax_NMDA = 0.0
                num_disabled[conn.postsynaptic_index] += 1

    # Disable somatic/proximal fast NMDA subunits
    if net_conf['STN'].get('disable_somatic_NR2A', False):
        # NOTE: config uses a separate NMDAsyn point process for somatic NMDAR
        all_proj['CTX']['STN'].set(NMDAsynTM_gmax_NMDA=0.0)

    # Only allow GABA-B currents on reported fraction of cells
    # (can also do this using separate Projections with only GABA-B/GABA-A)
    num_without_GABAB = net_conf['STN'].get('num_cell_without_GABAB', 0)
    if num_without_GABAB > 0:
        # Pick subset of cells with GABA-B disabled
        pop_sample = pop_stn.sample(num_without_GABAB, rng=shared_rng_pynn)
        stn_ids = pop_sample.all_cells  # global ids
        for pre in 'GPE.all', 'GPE.proto', 'GPE.surrogate':
            if pre in all_proj and 'STN' in all_proj[pre]:
                for conn in all_proj[pre]['STN'].connections:
                    if conn.postsynaptic_cell in stn_ids:
                        conn.gmax_GABAB = 0.0
                        print('Disabled GABAB on STN cell with id {}'.format(conn.postsynaptic_cell))

    #---------------------------------------------------------------------------
    # Sanity check: make sure all populations and projections are instantiated

    undefined_pops = [cpop for cpop in config_pop_labels if (
                        cpop not in all_pops and cpop not in all_asm)]
    undefined_proj = [(pre, post) for (post, pre) in net_conf.items() if (
                        (pre in config_pop_labels and post in config_pop_labels)
                        and (pre not in all_proj or post not in all_proj[pre]))]

    err_msg = ''
    if len(undefined_pops) > 0:
        err_msg += ("\nFollowing populations in config file were not "
                    "instantiated in simulator: {}".format(undefined_pops))

    if len(undefined_proj) > 0:
        err_msg += ("\nFollowing projections in config file were not "
                    "instantiated in simulator: {}".format(undefined_proj))

    if err_msg:
        raise Exception(err_msg)

    ############################################################################
    # RECORDING
    ############################################################################
    print("rank {}: starting phase RECORDING.".format(mpi_rank))

    # Default traces
    traces_biophys = {
        'Vm':       {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
    }

    for pop in biophysical_pops:
        pop.record(traces_biophys.items(), sampling_interval=.05)

    for pop in all_pops.values():
        pop.record(['spikes'], sampling_interval=.05)

    if with_lfp:
        for pop in Population.all_populations:
            if pop.celltype.has_parameter('with_extracellular_rec'):
                # Check if there is at least one cell with extracellular mechanisms
                record_pop_lfp = reduce(
                    lambda x,y: x or y,
                    pop.celltype.parameter_space['with_extracellular_rec'])
                if record_pop_lfp:
                    pop.record(['lfp'], sampling_interval=.05)

    # Traces defined in config file
    for pop_label, pop_config in net_conf.iteritems():
        if 'traces' not in pop_config:
            continue
        if pop_label in all_pops:
            target_pop = all_pops[pop_label]
        elif pop_label in all_asm:
            target_pop = all_asm[pop_label]
        else:
            raise ValueError("Unknown population to record from: {}".format(pop_label))

        # Translate trace group specifier to Population.record() call
        for trace_group in pop_config['traces']:
            pop_sample = trace_group['cells']
            if pop_sample in (':', 'all'):
                target_cells = target_pop
            elif isinstance(pop_sample, int):
                target_cells = target_pop.sample(pop_sample, rng=shared_rng_pynn)
            elif isinstance(pop_sample, (str, unicode)):
                slice_args = [int(i) if i!='' else None for i in pop_sample.split(':')]
                target_cells = target_pop[slice(*slice_args)]
            elif isinstance(pop_sample, list):
                target_cells = target_pop[pop_sample]
            else:
                raise ValueError("Cannot interpret cell indices '{}'".format(pop_sample))
            target_cells.record(trace_group['specs'].items(),
                                sampling_interval=trace_group['sampling_period'])


    ############################################################################
    # WRITE PARAMETERS
    ############################################################################
    print("rank {}: starting phase WRITE PARAMETERS.".format(mpi_rank))

    # NOTE: - any call to Population.get() Projection.get() does a ParallelContext.gather()
    #       - cannot perform any gather() operations before initializing MPI transfer
    #       - must do gather() operations on all nodes

    # Save cell information
    for pop in all_pops.values() + all_asm.values():
        pop_params = saved_params.setdefault(pop.label, {})
        pop_cell_gids = list(pop.all_cells.astype(int))
        pop_subcell_gids = sum((sim.state.get_spkgids(gid) for gid in pop_cell_gids), [])
        pop_params['gids'] = pop_cell_gids + pop_subcell_gids

    # Save connection information
    for pre_pop, post_pops in all_proj.iteritems():
        saved_params.setdefault(pre_pop, {})
        for post_pop, proj in post_pops.iteritems():

            # Plot connectivity matrix ('O' is connection, ' ' is no connection)
            utf_matrix, float_matrix = connection_plot(proj)
            # max_line_length = 500
            # if mpi_rank == 0 and proj.post.size < max_line_length:
            #     logger.debug("{}->{} connectivity matrix (dim[0,1] = [src,target]: \n".format(proj.pre.label, proj.post.label) + utf_matrix)

            # This does an mpi gather() on all the parameters
            conn_params = ["delay", "weight"]
            gsyn_params = ['gmax_AMPA', 'gmax_NMDA', 'gmax_GABAA', 'gmax_GABAB']
            conn_params.extend([
                p for p in gsyn_params if p in proj.synapse_type.default_parameters
            ])
            pre_post_params = np.array(proj.get(conn_params, format="list",
                                       gather='all', multiple_synapses='sum'))

            # Sanity check: minimum and maximum delays and weights
            mind = min(pre_post_params[:,2])
            maxd = max(pre_post_params[:,2])
            minw = min(pre_post_params[:,3])
            maxw = max(pre_post_params[:,3])

            if mpi_rank == 0:
                logger.debug(
                    "Error check for projection {pre}->{post}:\n"
                    "    - delay  [min, max] = [{mind}, {maxd}]\n"
                    "    - weight [min, max] = [{minw}, {maxw}]\n".format(
                        pre=pre_pop, post=post_pop, mind=mind, maxd=maxd,
                        minw=minw, maxw=maxw))

            # Save all connectivity pairs, using cell indices, and GIDs
            pop_idx_pairs = [tuple(pair) for pair in pre_post_params[:, 0:2].astype(int)]
            cell_gid_pairs = []
            for conn in proj.connections:
                if hasattr(conn, 'presynaptic_gid'):
                    cell_gid_pairs.append((conn.presynaptic_gid, conn.postsynaptic_gid))
                else:
                    cell_gid_pairs.append((int(conn.presynaptic_cell), int(conn.postsynaptic_cell)))

            # Append to saved dictionary
            proj_params = saved_params[pre_pop].setdefault(post_pop, {})
            proj_params['conn_matrix'] = float_matrix
            proj_params['conpair_pop_indices'] = pop_idx_pairs
            proj_params['conpair_gids'] = cell_gid_pairs
            proj_params['conpair_pvals'] = pre_post_params
            proj_params['conpair_pnames'] = conn_params


    # Write model parameters
    print("rank {}: starting phase WRITE PARAMETERS.".format(mpi_rank))
    if mpi_rank==0 and output is not None:


        # Save projection parameters
        extension = out_ext[:-4] + '.pkl'
        params_outfile = "{dir}pop-parameters{ext}".format(dir=out_dir, ext=extension)
        with open(params_outfile, 'wb') as fout:
            pickle.dump(saved_params, fout)


    ############################################################################
    # INITIALIZE & SIMULATE
    ############################################################################
    print("rank {}: starting phase SIMULATE.".format(mpi_rank))

    # Set physiological conditions
    h.celsius = 36.0
    h.nai0_na_ion = 15
    h.nao0_na_ion = 128.5
    h.ki0_k_ion = 140
    h.ko0_k_ion = 2.5
    h.cai0_ca_ion = 1e-04
    h.cao0_ca_ion = 2.0
    h("cli0_cl_ion = 4")
    h("clo0_cl_ion = 132.5")

    # Simulation statistics
    num_segments = sum((sec.nseg for sec in h.allsec()))
    num_cell = sum((1 for sec in h.allsec()))
    each_num_segments = comm.gather(num_segments, root=0)
    if mpi_rank == 0:
        # only rank 0 receives broadcast result
        total_num_segments = sum(each_num_segments)
        print("Entire network consists of {} segments (compartments)".format(
              total_num_segments))

    print("MPI rank {} will simulate {} segments ({} sections) for {} ms.".format(
            mpi_rank, num_segments, num_cell, sim_dur))


    tstart = time.time()
    outdir, filespec = os.path.split(output)
    progress_file = os.path.join(outdir, '{}_sim_progress.log'.format(
        datetime.fromtimestamp(tstart).strftime('%Y.%m.%d-%H.%M.%S')))


    # Times for writing out data to file
    if transient_period is None:
        transient_period = 0.0 # (ms)
    steady_period = sim_dur - transient_period
    if max_write_interval is None:
        max_write_interval = 10e3 # (ms)
    homogenize_intervals = False
    if homogenize_intervals:
        write_interval = steady_period / (steady_period // max_write_interval + 1)
    else:
        write_interval = max_write_interval
    if transient_period == 0:
        first_write_time = write_interval
    else:
        first_write_time = transient_period
    write_times = list(np.arange(first_write_time, sim_dur, write_interval)) + [sim_dur]
    last_write_time = 0.0
    last_report_time = tstart

    # Restore state
    if restore_state:
        sim.state.restore = True

    # SIMULATE
    while sim.state.t < sim_dur:
        sim.run(report_interval)

        # Report simulation progress
        if mpi_rank == 0:
            tnow = time.time()
            t_elapsed = tnow - tstart
            t_stepdur = tnow - last_report_time
            last_report_time = tnow
            # ! No newlines in progress report - passed to shell
            progress = ("Simulation time is {} of {} ms. "
                        "CPU time elapsed is {} s, last step took {} s".format(
                        sim.state.t, sim_dur, t_elapsed, t_stepdur))
            print(progress)

            if report_progress:
                stamp = datetime.fromtimestamp(tnow).strftime('%Y-%m-%d@%H:%M:%S')
                os.system("echo [{}]: {} >> {}".format(stamp, progress, progress_file))

        # Write recorded data
        if len(write_times) > 0 and abs(sim.state.t - write_times[0]) <= 5.0:
            suffix = "_{:.0f}ms-{:.0f}ms".format(last_write_time, sim.state.t)
            for pop in all_pops.values():
                write_population_data(pop, output, suffix, gather=True, clear=True)
            write_times.pop(0)
            last_write_time = sim.state.t

    # Report simulation statistics
    tstop = time.time()
    cputime = tstop - tstart
    each_num_segments = comm.gather(num_segments, root=0)
    if mpi_rank == 0:
        # only rank 0 receives broadcast result
        total_num_segments = sum(each_num_segments)
        print("Simulated {} segments for {} ms in {} s CPU time".format(
                total_num_segments, sim.state.tstop, cputime))

    # Save simulator state
    if save_state:
        sim.state.save_state()

    if export_locals:
        globals().update(locals())

    print("rank {}: SIMULATION FINISHED.".format(mpi_rank))


if __name__ == '__main__':
    # Parse arguments passed to `python model.py [args]`
    import argparse

    parser = argparse.ArgumentParser(description='Run basal ganglia network simulation')

    parser.add_argument('-d', '--dur', nargs='?', type=float, default=500.0,
                        dest='sim_dur', help='Simulation duration')

    parser.add_argument('-dt', '--simdt', nargs='?', type=float, default=None,
                        dest='sim_dt', help='Simulation time step')

    parser.add_argument('--scale', nargs='?', type=float, default=1.0,
                        dest='pop_scale', help='Scale for population sizes')

    parser.add_argument('--seed', nargs='?', type=int, default=None,
                        dest='seed', help='Seed for random number generator')

    parser.add_argument('-wi', '--writeinterval', nargs='?', type=float, default=None,
                        dest='max_write_interval',
                        help='Interval between successive write out of recording data')

    parser.add_argument('-tp', '--transientperiod', nargs='?', type=float, default=None,
                        dest='transient_period',
                        help=('Duration of transient period at start of simulation. '
                              'First data write-out is after transient period'))

    parser.add_argument('-ri', '--reportinterval', nargs='?', type=float, default=50.0,
                        dest='report_interval',
                        help='Interval between reports of simulation time.')

    parser.add_argument('--lfp',
                        dest='with_lfp', action='store_true',
                        help='Calculate Local Field Potential.')
    parser.add_argument('--nolfp',
                        dest='with_lfp', action='store_false',
                        help='Calculate Local Field Potential.')
    parser.set_defaults(with_lfp=None)

    parser.add_argument('--dbs',
                        dest='with_dbs', action='store_true',
                        help='Apply deep brain stimulation.')
    parser.add_argument('--nodbs',
                        dest='with_dbs', action='store_false',
                        help='Apply deep brain stimulation.')
    parser.set_defaults(with_dbs=None)

    parser.add_argument('--exportcomplocs',
                        dest='export_compartment_coordinates', action='store_true',
                        help='Export compartment coordinates')
    parser.set_defaults(export_compartment_coordinates=True)

    parser.add_argument('--dd',
                        dest='dopamine_depleted', action='store_true',
                        help='Set dopamine depleted condition.')
    parser.add_argument('--dnorm',
                        dest='dopamine_depleted', action='store_false',
                        help='Set dopamine normal condition.')
    parser.set_defaults(dopamine_depleted=None)


    parser.add_argument('-o', '--outdir', nargs='?', type=str,
                        default='~/storage/',
                        dest='output',
                        help='Output destination in format \'/outdir/*.ext\''
                             ' or /path/to/outdir/ with trailing slash')

    parser.add_argument('-p', '--progress',
                        dest='report_progress', action='store_true',
                        help='Report progress periodically to progress file')
    parser.set_defaults(report_progress=False)

    parser.add_argument('--restorestate',
                        dest='restore_state', action='store_true',
                        help='Restore simulator state from ./in/ directory.')
    parser.set_defaults(restore_state=False)

    parser.add_argument('--savestate',
                        dest='save_state', action='store_true',
                        help='Save simulator state to ./out/ directory.')
    parser.set_defaults(save_state=False)

    parser.add_argument('-id', '--identifier', nargs=1, type=str,
                        metavar='<job identifer>',
                        dest='job_id',
                        help='Job identifier to tag the simulation')

    parser.add_argument('-dc', '--configdir', nargs=1, type=str,
                        metavar='/path/to/circuit_config',
                        default='config',
                        dest='config_root',
                        help='Directory containing circuit configuration.'
                             ' All other configuration files will be considered'
                             ' relative to this directory if they consist only'
                             ' of a filename.')

    parser.add_argument('-cs', '--simconfig', nargs=1, type=str,
                        metavar='net_config.json',
                        dest='net_conf_path',
                        help='Simulation configuration (JSON file).'
                             ' Either provide full path or filename located'
                             ' in configdir/circuits/.')

    parser.add_argument('-cc', '--cellconfig', nargs=1, type=str,
                        metavar='cell_conf.json',
                        dest='cell_conf',
                        help='Cell configuration file (pickle file).')

    parser.add_argument('-cf', '--femconfig', nargs=1, type=str,
                        metavar='fem_conf.npy',
                        dest='fem_conf', default=None,
                        help='FEM configuration file (numpy file).')

    parser.add_argument('-ca', '--axonfile', nargs=1, type=str,
                        metavar='/path/to/axon_coordinates.pkl',
                        dest='axon_coord_file',
                        help='Axon coordiantes file (pickle file).')

    parser.add_argument('-dm', '--morphdir', nargs=1, type=str,
                        default='morphologies',
                        metavar='/path/to/morphologies_dir',
                        dest='morph_dir',
                        help='Morphologies directory.')




    args = parser.parse_args() # Namespace object
    parsed_dict = vars(args) # Namespace to dict

    # Parse config files
    config_root = os.path.expanduser(parsed_dict.pop('config_root')[0])

    # Default parent directory of each configuration file
    default_dirs = {
        'net_conf_path': os.path.join(config_root, 'circuits'),
        'cell_conf': os.path.join(config_root, 'cells'),
        'fem_conf': os.path.join(config_root, 'fem'),
        'morph_dir': config_root,
        'axon_coord_file': os.path.join(config_root, 'axons'),
    }

    # Locate each configuration file
    for conf_name, default_dir in default_dirs.items():
        if parsed_dict[conf_name] is None:
            continue
        conf_dir, conf_filename = os.path.split(parsed_dict[conf_name][0])
        # If directories are prepended, look there. Otherwise look in default dir.
        if conf_dir == '':
            parsed_dict[conf_name] = os.path.join(default_dir, conf_filename)
        else:
            parsed_dict[conf_name] = os.path.join(os.path.expanduser(conf_dir), conf_filename)

    # Read configuration files
    parsed_dict['net_conf'] = fileutils.parse_json_file(
                            parsed_dict['net_conf_path'], nonstrict=True)
    parsed_dict['cell_conf'] = fileutils.parse_json_file(
                            parsed_dict['cell_conf'], nonstrict=True)
    if parsed_dict['fem_conf'] is not None:
        parsed_dict['fem_conf'] = np.load(parsed_dict['fem_conf'])
    with open(parsed_dict['axon_coord_file'], 'rb') as axon_file:
        parsed_dict['axon_coordinates'] = pickle.load(axon_file)


    # Post process output specifier
    out_basedir = parsed_dict['output']
    if out_basedir is None or out_basedir == '': # shell can pass empty string
        out_basedir = '~/storage'
    job_id = parsed_dict.pop('job_id')[0]
    timestamp = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')

    # Default output directory
    # NOTE: don't use timestamp -> mpi ranks will make different filenames
    config_name, ext = os.path.splitext(os.path.basename(parsed_dict['net_conf_path']))
    out_subdir = 'LuNetDBS_{stamp}_job-{job_id}_{config_name}'.format(
        stamp=timestamp, job_id=job_id, config_name=config_name)

    # File names for data files
    # Default output format is hdf5 / NIX io
    filespec = '*_{stamp}_scale-{scale}_dur-{dur}_job-{job_id}.mat'.format(
        scale=parsed_dict['pop_scale'], dur=parsed_dict['sim_dur'],
        stamp=timestamp, job_id=job_id)

    # Make output directory if non-existing, but only on one host
    out_basedir = os.path.expanduser(out_basedir)
    if not os.path.isdir(out_basedir) and mpi_rank == 0:
        os.mkdir(out_basedir)

    # Don't make directory with variable timestamp -> mpi ranks will make different
    out_fulldir = os.path.join(out_basedir, out_subdir)
    if not os.path.isdir(out_fulldir) and mpi_rank == 0:
        os.mkdir(out_fulldir)
    parsed_dict['output'] = os.path.join(out_fulldir, filespec)

    # Copy config file to output directory
    if mpi_rank == 0:
        import shutil
        shutil.copytree(config_root,
                os.path.join(out_fulldir, 'simconfig'))
        shutil.copy2(parsed_dict['net_conf_path'],
                os.path.join(out_fulldir, 'simconfig', 'sim_config.json'))
        shutil.copy2(__file__,
                os.path.join(out_fulldir, 'simconfig', 'model_script_{}.py'.format(job_id)))

        print("\nFinal parsed arguments:")
        print("\n".join(
            "{:<16}: {}".format(k, v) for k,v in parsed_dict.items()
                if 'conf' not in k and k not in ('axon_coordinates',)))

    # Run the simulation
    simulate_model(**parsed_dict)
