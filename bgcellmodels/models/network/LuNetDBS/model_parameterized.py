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

>>> mpirun -n 6 <command>


Run single-threaded using IPython:

>>> ipython
>>> %run <command>

Example commands:

>>> model_parameterized.py -id testrun --dur 100 --scale 0.5 --seed 888 \
--dd --lfp --dbs \
--outdir ~/storage --transientperiod 0.0 --writeinterval 1000 \
--configdir ~/workspace/bgcellmodels/bgcellmodels/models/network/LuNetDBS/configs \
--simconfig test_simconfig.json \
--cellconfig test_cellconfig_5.json \
--axonfile axon_coordinates_cutoff.pkl \
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
from pyNN.utility import init_logging # connection_plot is bugged
import neo.io

# Custom PyNN extensions
from bgcellmodels.extensions.pynn import synapses as custom_synapses
from bgcellmodels.extensions.pynn.utility import connection_plot
from bgcellmodels.extensions.pynn.populations import Population
from bgcellmodels.extensions.pynn.axon_models import AxonRelayType
from bgcellmodels.extensions.pynn import spiketrains as spikegen

# NEURON models and mechanisms
from bgcellmodels.emfield import stimulation
from bgcellmodels.mechanisms import synapses, noise # loads MOD files
from bgcellmodels.cellpopdata import connectivity # for use in config files

from bgcellmodels.models.STN.Miocinovic2006 import miocinovic_pynn_model as miocinovic
from bgcellmodels.models.GPe.Gunay2008 import gunay_pynn_model as gunay
from bgcellmodels.models.axon.foust2011 import AxonFoust2011

from bgcellmodels.common.configutil import eval_params
from bgcellmodels.common.stdutil import getdictvals
from bgcellmodels.common import logutils, fileutils

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
    'AxonBuilder'])

logutils.setLogLevel('DEBUG', ['simulation'])

h("XTRA_VERBOSITY = 0")


def nprint(*args, **kwargs):
    """
    Print only on host with rank 0.
    """
    if mpi_rank == 0:
        print(*args, **kwargs)


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


def simulate_model(
        pop_scale       = 1.0,
        sim_dur         = 500.0,
        sim_dt          = None,
        export_locals   = True,
        output          = None,
        report_progress = None,
        config          = None,
        cell_config     = None,
        axon_coordinates = None,
        morph_dir       = None,
        seed            = None,
        with_lfp        = None,
        with_dbs        = None,
        dopamine_depleted = None,
        transient_period = None,
        max_write_interval = None,
        report_interval = 50.0,
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

    sim_params = config['simulation']
    emf_params = config['electromagnetics']

    rho_ohm_cm = 1.0 / (emf_params['sigma_extracellular_S/m'] * 1e-2)

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

    print("""\nRunning net on MPI rank {} with following settings:
    - sim_dur = {}
    - sim_dt = {}
    - output = {}""".format(mpi_rank, sim_dt, sim_dur, output))

    print("\nThis is node {} ({} of {})\n".format(
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
    nprint("Dopamine state is " + "DEPLETED" if DD else "NORMAL")

    ############################################################################
    # LOCAL FUNCTIONS
    ############################################################################

    params_global_context = globals()
    params_local_context = locals() # capture function arguments

    def get_pop_parameters(pop, *param_names):
        """
        Get population parameters from config and evaluate them.
        """
        config_locals = config[pop].get('local_context', {})
        param_specs = getdictvals(config[pop], *param_names, as_dict=True)
        pvals = eval_params(param_specs, params_global_context,
                            [params_local_context, config_locals])
        return getdictvals(pvals, *param_names)

    def get_param_group(pop, group_name=None, mapping=None):
        """
        Get a group of parameters for a population as dictionary.
        """
        config_locals = config[pop].get('local_context', {})
        if group_name is None:
            param_specs = config[pop]
        else:
            param_specs = config[pop][group_name]
        if mapping is not None:
            param_specs = {v: param_specs[k] for k,v in mapping.iteritems()}
        return eval_params(param_specs, params_global_context,
                           [params_local_context, config_locals])

    def get_cell_parameters(pop):
        """
        Get PyNN cell parameters as dictionary of numerical values.
        """
        config_locals = config[pop].get('local_context', {})
        param_specs = config[pop].get('PyNN_cell_parameters', {})
        return eval_params(param_specs, params_global_context,
                           [params_local_context, config_locals])

    def synapse_from_config(pre, post):
        """
        Make Synapse object from config dict
        """
        config_locals = config[post].get('local_context', {})
        syn_type, syn_params = getdictvals(config[post][pre]['synapse'],
                                           'name', 'parameters')
        if hasattr(custom_synapses, syn_type):
            syn_class = getattr(custom_synapses, syn_type)
        else:
            syn_class = getattr(sim, syn_type)
        syn_pvals = eval_params(syn_params, params_global_context,
                                [params_local_context, config_locals])
        num_contacts = config[post][pre].get('num_contacts', 1)
        syntype_obj = syn_class(**syn_pvals)
        syntype_obj.num_contacts = num_contacts
        return syntype_obj

    def connector_from_config(pre, post, rng=None):
        """
        Make Connector object from config dict
        """
        config_locals = config[post].get('local_context', {})
        con_type, con_params = getdictvals(config[post][pre]['connector'],
                                           'name', 'parameters')
        connector_class = getattr(sim, con_type)
        con_pvals = eval_params(con_params, params_global_context,
                               [params_local_context, config_locals])
        connector = connector_class(**con_pvals)
        if rng is not None:
            connector.rng = rng
        return connector

    axon_scales = { 'mm' : 1.0, 'um': 1e-3, 'm': 1e3 }
    axon_scale = axon_scales[cell_config['units']['axons']]
    
    def get_axon_coordinates(axon_id):
        return np.asarray(axon_coordinates[axon_id]) * axon_scale

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

    ############################################################################
    # POPULATIONS
    ############################################################################
    # Define each cell population with its cell type, number of cells
    # NOTE:
    # - to query cell model attributes, use population[i]._cell
    print("rank {}: starting phase POPULATIONS.".format(mpi_rank))

    config_pop_labels = [k for k in config.keys() if not k in 
                            ('simulation', 'electromagnetics')]

    #===========================================================================
    # STN POPULATION

    stn_ncell_base, = get_pop_parameters('STN', 'base_population_size')
    stn_ncell_biophys = int(stn_ncell_base * pop_scale)

    #---------------------------------------------------------------------------
    # STN cell model

    # Select cells to simulate
    pop_cell_defs = [
        cell for cell in cell_config['cells'] if 
            (cell['population'] == 'STN') and (cell['axon'] is not None)
    ]
    cell_defs = pop_cell_defs[:stn_ncell_biophys]
    
    # Get 3D morphology properties of each cell
    cells_transforms = [np.asarray(cell['transform']) for cell in cell_defs]
    cells_axon_coords = [get_axon_coordinates(cell['axon']) for cell in cell_defs]

    # Choose a random morphology for each cell
    candidate_morphologies = np.array(cell_config['default_morphologies']['STN'])
    candidates_sampled = shared_rng.choice(len(candidate_morphologies), stn_ncell_biophys)
    cells_morph_paths = [
        get_morphology_path(m) for m in candidate_morphologies[candidates_sampled]
    ]

    # Load default parameters from sim config
    stn_cell_params = get_cell_parameters('STN')

    # Add parameters from other sources
    stn_cell_params['with_extracellular'] = with_lfp or with_dbs
    stn_cell_params['morphology_path'] = cells_morph_paths
    stn_cell_params['transform'] = cells_transforms
    stn_cell_params['streamline_coordinates_mm'] = cells_axon_coords
    stn_cell_params['rho_extracellular_ohm_cm'] = rho_ohm_cm
    stn_cell_params['electrode_coordinates_um'] = emf_params['dbs_electrode_coordinates_um']

    
    stn_type = miocinovic.StnMorphType(**stn_cell_params)

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
    pop_cell_defs = [cell for cell in cell_config['cells'] if cell['population'] == 'GPE']
    cell_defs = pop_cell_defs[:gpe_ncell_biophys]
    
    # Get 3D morphology properties of each cell
    cells_transforms = [np.asarray(cell['transform']) for cell in cell_defs]

    # Load default parameters from sim config
    gpe_cell_params = get_cell_parameters('GPE.all')

    # Add parameters from other sources
    gpe_cell_params['with_extracellular'] = with_lfp or with_dbs
    gpe_cell_params['transform'] = cells_transforms
    # NOTE: GPe axons are NOT electrically attached to morphology, but work via relay
    # gpe_cell_params['streamline_coordinates_mm'] = cells_axon_coords
    gpe_cell_params['rho_extracellular_ohm_cm'] = rho_ohm_cm
    gpe_cell_params['electrode_coordinates_um'] = emf_params['dbs_electrode_coordinates_um']

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
    # GPE axon population

    num_gpe_axons = (gpe_ncell_biophys + ncell_surrogate)

    # NOTE: can use any axon per cell, since they are not electrically connected
    gpe_conn_defs = [
        c for c in cell_config['connections'] if (c['projection'] == 'GPE-STN')
    ]

    gpe_axon_coords = [
        get_axon_coordinates(connection['axon']) for connection in gpe_conn_defs
    ][:num_gpe_axons]

    # Get axon associated with cell (not necessary if no electrical connection)
    # gpe_axon_coords = [
    #     get_axon_coordinates(cell['axon']) for cell in pop_cell_defs if 
    #         (cell['axon'] is not None)
    # ][:num_gpe_axons]

    # Re-use axon definitions if insufficient
    while len(gpe_axon_coords) < num_gpe_axons:
        num_additional = num_gpe_axons - len(gpe_axon_coords)
        gpe_axon_coords += gpe_axon_coords[:num_additional]
        logger.warning('GPe: Re-using %d axon definitions', num_additional)

    # Cell type for axons
    gpe_axon_params = {
        'axon_class':                   AxonFoust2011,
        'streamline_coordinates_mm':    gpe_axon_coords,
        'termination_method':           np.array('terminal_sequence'),
        'with_extracellular':           with_lfp or with_dbs,
        'electrode_coordinates_um' :    emf_params['dbs_electrode_coordinates_um'],
        'rho_extracellular_ohm_cm' :    rho_ohm_cm,         
    }

    gpe_axon_type = AxonRelayType(**gpe_axon_params)

    # Initial values for state variables
    vinit = gpe_axon_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }
    
    pop_gpe_axons = Population(num_gpe_axons,
                               cellclass=gpe_axon_type,
                               label='GPE.axons',
                               initial_values=initial_values)

    #---------------------------------------------------------------------------
    # GPE Assembly (all GPe subtypes)

    if pop_gpe_surrogate is None:
        asm_gpe = sim.Assembly(pop_gpe_proto, pop_gpe_surrogate, label='GPE.all')
    else:
        asm_gpe = sim.Assembly(pop_gpe_proto, label='GPE.all')
    gpe_pop_size = asm_gpe.size


    #===========================================================================
    # CTX POPULATION

    # CTX spike sources
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
    # CTX axon population

    num_ctx_axons = ctx_ncell

    ctx_conn_defs = [
        c for c in cell_config['connections'] if (c['projection'] == 'CTX-STN')
    ]

    ctx_axon_coords = [
        get_axon_coordinates(connection['axon']) for connection in ctx_conn_defs
    ]

    while len(ctx_axon_coords) < num_ctx_axons:
        num_additional = num_ctx_axons - len(ctx_axon_coords)
        ctx_axon_coords += ctx_axon_coords[:num_additional]
        logger.warning('CTX: Re-using %d axon definitions', num_additional)

    # Cell type for axons
    ctx_axon_params = {
        'axon_class':                   AxonFoust2011,
        'streamline_coordinates_mm':    ctx_axon_coords,
        'termination_method':           np.array('terminal_sequence'),
        'with_extracellular':           with_lfp or with_dbs,
        'electrode_coordinates_um' :    emf_params['dbs_electrode_coordinates_um'],
        'rho_extracellular_ohm_cm' :    rho_ohm_cm,         
    }

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
            raise Exception('Coud not find mechanism "xtra" in any section.')

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
    for post_label, pop_config in config.iteritems():
        
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
    num_prune = config['STN'].get('prune_dendritic_GLUR', 0)
    if DD and num_prune > 0:
        # PD: dendritic AMPA & NMDA-NR2B/D afferents pruned
        num_disabled = np.zeros(pop_stn.size)
        for conn in all_proj['CTX']['STN'].connections:
            if num_disabled[conn.postsynaptic_index] < num_prune:
                conn.GLUsyn_gmax_AMPA = 0.0
                conn.GLUsyn_gmax_NMDA = 0.0
                num_disabled[conn.postsynaptic_index] += 1

    # Disable somatic/proximal fast NMDA subunits
    if config['STN'].get('disable_somatic_NR2A', False):
        # NOTE: config uses a separate NMDAsyn point process for somatic NMDAR
        all_proj['CTX']['STN'].set(NMDAsynTM_gmax_NMDA=0.0)

    # Only allow GABA-B currents on reported fraction of cells
    num_without_GABAB = config['STN'].get('num_cell_without_GABAB', 0)
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
        # TODO: for cells without GABAB, create new Projection with only GABA-A synapses
        #       - either from surrogate only or whole population (choose)

    #---------------------------------------------------------------------------
    # Sanity check: make sure all populations and projections are instantiated

    undefined_pops = [cpop for cpop in config_pop_labels if (
                        cpop not in all_pops and cpop not in all_asm)]
    undefined_proj = [(pre, post) for (post, pre) in config.items() if (
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
            if pop.celltype.has_parameter('with_extracellular'):
                # Check if there is at least one cell with extracellular mechanisms
                has_extracellular = reduce(
                    lambda x,y: x or y,
                    pop.celltype.parameter_space['with_extracellular'])
                if has_extracellular:
                    pop.record(['lfp'], sampling_interval=.05)

    # Traces defined in config file
    for pop_label, pop_config in config.iteritems():
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

    print("Will simulate {} segments ({} sections) for {} ms on MPI rank {}.".format(
            num_segments, num_cell, sim_dur, mpi_rank))


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
        print("Simulated {} segments for {} ms in {} ms CPU time".format(
                total_num_segments, sim.state.tstop, cputime))


    ############################################################################
    # WRITE PARAMETERS
    ############################################################################
    print("rank {}: starting phase INTEGRITY CHECK.".format(mpi_rank))

    # NOTE: - any call to Population.get() Projection.get() does a ParallelContext.gather()
    #       - cannot perform any gather() operations before initializing MPI transfer
    #       - must do gather() operations on all nodes
    saved_params = {'dopamine_depleted': DD}

    # Save cell information
    for pop in all_pops.values() + all_asm.values():
        saved_params.setdefault(pop.label, {})['gids'] = pop.all_cells.astype(int)

    # Save connection information
    for pre_pop, post_pops in all_proj.iteritems():
        saved_params.setdefault(pre_pop, {})
        for post_pop, proj in post_pops.iteritems():

            # Plot connectivity matrix ('O' is connection, ' ' is no connection)
            utf_matrix, float_matrix = connection_plot(proj)
            max_line_length = 500
            if mpi_rank == 0 and proj.post.size < max_line_length:
                logger.debug("{}->{} connectivity matrix (dim[0,1] = [src,target]: \n".format(
                    proj.pre.label, proj.post.label) + utf_matrix)

            # This does an mpi gather() on all the parameters
            conn_params = ["delay", "weight"]
            gsyn_params = ['gmax_AMPA', 'gmax_NMDA', 'gmax_GABAA', 'gmax_GABAB']
            conn_params.extend(
                [p for p in gsyn_params if p in proj.synapse_type.default_parameters])
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

            # Make (gid, gid) connectivity pairs
            pop_idx_pairs = [tuple(pair) for pair in pre_post_params[:, 0:2].astype(int)]
            cell_gid_pairs = [(int(proj.pre[a]), int(proj.post[b])) for a, b in pop_idx_pairs]

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
        outdir, extension = output.split('*')

        # Save projection parameters
        import pickle
        extension = extension[:-4] + '.pkl'
        params_outfile = "{dir}pop-parameters{ext}".format(dir=outdir, ext=extension)
        with open(params_outfile, 'wb') as fout:
            pickle.dump(saved_params, fout)

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
    parser.set_defaults(with_lfp=False)

    parser.add_argument('--dbs',
                        dest='with_dbs', action='store_true',
                        help='Apply deep brain stimulation.')
    parser.add_argument('--nodbs',
                        dest='with_dbs', action='store_false',
                        help='Apply deep brain stimulation.')
    parser.set_defaults(with_dbs=False)

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

    parser.add_argument('-id', '--identifier', nargs=1, type=str,
                        metavar='<job identifer>',
                        dest='job_id',
                        help='Job identifier to tag the simulation')

    parser.add_argument('-dc', '--configdir', nargs=1, type=str,
                        metavar='/path/to/circuit_config',
                        dest='config_root',
                        help='Directory containing circuit configuration.'
                             ' All other configuration files will be considered'
                             ' relative to this directory if they consist only'
                             ' of a filename.')

    parser.add_argument('-cs', '--simconfig', nargs=1, type=str,
                        metavar='sim_config.json',
                        dest='sim_config_file',
                        help='Simulation configuration (JSON file).')

    parser.add_argument('-cc', '--cellconfig', nargs=1, type=str,
                        metavar='cell_config.json',
                        dest='cell_config_file',
                        help='Cell configuration file (pickle file).')

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
        'morph_dir': config_root,
        'sim_config_file': os.path.join(config_root, 'circuits'),
        'cell_config_file': os.path.join(config_root, 'cells'),
        'axon_coord_file': os.path.join(config_root, 'axons'),
    }
    # Locate each configuration file
    for conf, parent_dir in default_dirs.items():
        conf_filedir, conf_filename = os.path.split(parsed_dict[conf][0])
        if conf_filedir == '':
            conf_filepath = os.path.join(parent_dir, conf_filename)
        else:
            conf_filepath = os.path.join(os.path.expanduser(conf_filedir), conf_filename)
        parsed_dict[conf] = conf_filepath

    # Read configuration files
    parsed_dict['config'] = fileutils.parse_json_file(
                            parsed_dict['sim_config_file'], nonstrict=True)
    parsed_dict['cell_config'] = fileutils.parse_json_file(
                            parsed_dict['cell_config_file'], nonstrict=True)
    with open(parsed_dict['axon_coord_file'], 'rb') as axon_file:
        parsed_dict['axon_coordinates'] = pickle.load(axon_file)


    # Post process output specifier
    out_basedir = parsed_dict['output']
    if out_basedir is None or out_basedir == '': # shell can pass empty string
        out_basedir = '~/storage'
    job_id = parsed_dict.pop('job_id')[0]
    time_now = time.time()
    timestamp = datetime.fromtimestamp(time_now).strftime('%Y.%m.%d_%H.%M.%S')

    # Default output directory
    # NOTE: don't use timestamp -> mpi ranks will make different filenames
    config_name, ext = os.path.splitext(os.path.basename(parsed_dict['sim_config_file']))
    out_subdir = 'LuNetStnGpe_{stamp}_job-{job_id}_{config_name}'.format(
                                            stamp=timestamp,
                                            job_id=job_id,
                                            config_name=config_name)

    # File names for data files
    # Default output format is hdf5 / NIX io
    filespec = '*_{stamp}_scale-{scale}_dur-{dur}_job-{job_id}.mat'.format(
                                            scale=parsed_dict['pop_scale'],
                                            dur=parsed_dict['sim_dur'],
                                            stamp=timestamp,
                                            job_id=job_id)

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
        shutil.copy2(parsed_dict['sim_config_file'],
                os.path.join(out_fulldir, 'simconfig', 'sim_config.json'))

    # Run the simulation
    simulate_model(**parsed_dict)
