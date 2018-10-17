# -*- coding: utf-8 -*-
"""
Basal Ganglia network model consisting of morphologically detailed
cell models for the major cell types.

Parameterized model construction based on configuration file / dictionary.

@author     Lucas Koelman

@date       31/05/2018

@see        PyNN manual for building networks:
                http://neuralensemble.org/docs/PyNN/building_networks.html
            PyNN examples of networks:
                https://github.com/NeuralEnsemble/PyNN/tree/master/examples

USAGE
-----

To run using MPI, you can use the following command:

>>> mpirun -n 8 python model_parameterized.py -n numcell -d simdur \
>>> -ng -o ~/storage -c ~/workspace/simple_config.json -id test1

To do a test run using IPython, use something like:

>>> %run model_parameterized.py --ncell 25 --dur 500 --seed 888 -id test1 \
>>> --transient-period 0.0 --write-interval 1000 --no-gui \
>>> --config myconfig.json --outdir ~/storage


NOTES
-----

- It may look like some imports are not used but they may be called dynamically
  using eval() based on the config file.

"""
from __future__ import print_function
import time
import os
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
from pyNN.parameters import Sequence
from pyNN.utility import init_logging # connection_plot is bugged
import neo.io

# Custom PyNN extensions
from bgcellmodels.extensions.pynn.synapses import GluSynapse, GabaSynapse, GabaSynTmHill
from bgcellmodels.extensions.pynn.utility import connection_plot
from bgcellmodels.extensions.pynn.populations import Population

# Monkey-patching of pyNN.neuron.Population class
# from bgcellmodels.extensions.pynn.recording import TraceSpecRecorder
# sim.Population._recorder_class = TraceSpecRecorder

# Custom NEURON mechanisms
from bgcellmodels.mechanisms import synapses, noise # loads MOD files

# Custom cell models
import bgcellmodels.models.STN.GilliesWillshaw.gillies_pynn_model as gillies
import bgcellmodels.models.GPe.Gunay2008.gunay_pynn_model as gunay
import bgcellmodels.cellpopdata.connectivity as connectivity # for use in config files
ConnectivityPattern = connectivity.ConnectivityPattern
make_connection_list = connectivity.make_connection_list
make_divergent_pattern = connectivity.make_divergent_pattern

# Our physiological parameters
# from bgcellmodels.cellpopdata.physiotypes import Populations as PopID
#from bgcellmodels.cellpopdata.physiotypes import ParameterSource as ParamSrc
# from bgcellmodels.cellpopdata.cellpopdata import CellConnector

from bgcellmodels.common.spikelib import make_oscillatory_bursts, make_variable_bursts
from bgcellmodels.common.configutil import eval_params
from bgcellmodels.common.stdutil import getdictvals
from bgcellmodels.common import logutils, fileutils

# Debug messages
logutils.setLogLevel('quiet', [
    'bpop_ext',
    'bluepyopt.ephys.parameters', 
    'bluepyopt.ephys.mechanisms', 
    'bluepyopt.ephys.morphologies'])


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


def run_simple_net(
        ncell_per_pop   = 30,
        sim_dur         = 500.0,
        export_locals   = True,
        with_gui        = True,
        output          = None,
        report_progress = None,
        config          = None,
        seed            = None,
        calculate_lfp   = None,
        burst_frequency = None,
        transient_period = None,
        max_write_interval = None):
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
    # SIMULATOR SETUP
    ############################################################################

    sim.setup(timestep=0.025, min_delay=0.1, max_delay=10.0, use_cvode=False)
    if mpi_rank == 0:
        init_logging(logfile=None, debug=True)
    

    print("""\nRunning net on MPI rank {} with following settings:
    - ncell_per_pop = {}
    - sim_dur = {}
    - output = {}""".format(mpi_rank, ncell_per_pop, sim_dur, output))
    
    print("\nThis is node {} ({} of {})\n".format(
          sim.rank(), sim.rank() + 1, sim.num_processes()))

    h = sim.h
    sim.state.duration = sim_dur # not used by PyNN, only by our custom funcs
    sim.state.rec_dt = 0.05
    sim.state.mcellran4_rng_indices = {} # Keep track of MCellRan4 indices for independent random streams.
    finit_handlers = []

    # Make one random generator that is shared and should yield same results
    # for each MPI rank, and one with unique results.
    # - The shared (parallel-safe) RNGs should be used in functions that are
    #   executed on all ranks, like instantiating Population and Projection
    #   objects.
    # - The default RNG for Connectors is NumpyRNG(seed=151985012)
    if seed is None:
        seed = config['simulation']['shared_rng_seed']
    shared_seed = seed # original: 151985012
    sim.state.rank_rng_seed = rank_seed = sim.state.native_rng_baseseed + sim.state.mpi_rank
    # RNGs that can be passed to PyNN objects like Connector subclasses
    shared_rng_pynn = sim.NumpyRNG(seed=shared_seed)
    rank_rng_pynn = sim.NumpyRNG(seed=rank_seed)
    # Raw Numpy RNGs (numpy.random.RandomState) to be used in our own functions
    shared_rng = shared_rng_pynn.rng
    rank_rng = rank_rng_pynn.rng
    


    ############################################################################
    # LOCAL FUNCTIONS
    ############################################################################

    params_global_context = globals()
    params_local_context = locals() # capture function arguments

    def get_pop_parameters(pop, *param_names):
        """
        Get parameter for population from config dict.
        """
        local_context = config[pop].get('local_context', {})
        param_specs = getdictvals(config[pop], *param_names, as_dict=True)
        pvals = eval_params(param_specs, params_global_context, local_context)
        return getdictvals(pvals, *param_names)

    def get_cell_parameters(pop):
        """
        Get PyNN cell parameters as dictionary of numerical values.
        """
        local_context = config[pop].get('local_context', {})
        param_specs = config[pop].get('PyNN_cell_parameters', {})
        return eval_params(param_specs, params_global_context, local_context)

    def synapse_from_config(pre, post):
        """
        Make Synapse object from config dict
        """
        local_context = config[post].get('local_context', {})
        syn_type, syn_params = getdictvals(config[post][pre]['synapse'],
                                           'name', 'parameters')
        syn_class = synapse_types[syn_type]
        syn_pvals = eval_params(syn_params, params_global_context, local_context)
        return syn_class(**syn_pvals)

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

    
    # LFP calculation: command line args get priority over config file
    if calculate_lfp is None:
        calculate_lfp, = get_pop_parameters('STN', 'calculate_lfp')
    
    # Set NEURON integrator/solver options
    if calculate_lfp:
        sim.state.cvode.use_fast_imem(True)
    sim.state.cvode.cache_efficient(True) # necessary for lfp, also 33% reduction in simulation time

    ############################################################################
    # POPULATIONS
    ############################################################################
    # Define each cell population with its cell type, number of cells
    print("{} start phase: POPULATIONS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # STN POPULATION
    stn_dx, gaba_mech = get_pop_parameters('STN', 'grid_dx', 'GABA_mechanism')
    stn_grid = space.Line(x0=0.0, dx=stn_dx,
                          y=0.0, z=0.0)
    ncell_stn = ncell_per_pop
    
    # FIXME: set electrode coordinates
    stn_cell_params = get_cell_parameters('STN')
    stn_type = gillies.StnCellType(
                        calculate_lfp=calculate_lfp,
                        lfp_sigma_extracellular=0.3,
                        **stn_cell_params)

    # Workaround because PyNN only allows numerical parameters
    stn_type.model.default_GABA_mechanism = gaba_mech

    pop_stn = Population(ncell_stn, 
                         cellclass=stn_type, 
                         label='STN',
                         structure=stn_grid)
    
    pop_stn.initialize(v=-63.0)


    #---------------------------------------------------------------------------
    # GPE POPULATION
    gpe_dx, gaba_mech = get_pop_parameters('GPE', 'grid_dx', 'GABA_mechanism')
    gpe_grid = space.Line(x0=0.0, dx=gpe_dx,
                          y=1e6, z=0.0)

    gpe_cell_params = get_cell_parameters('GPE')
    gpe_type = gunay.GPeCellType(**gpe_cell_params)
    gpe_type.model.default_GABA_mechanism = gaba_mech # workaround for string parameter

    ncell_gpe = ncell_per_pop
    pop_gpe = Population(ncell_gpe, 
                         cellclass=gpe_type,
                         label='GPE',
                         structure=gpe_grid)

    pop_gpe.initialize(v=-63.0)

    #---------------------------------------------------------------------------
    # GPE SURROGATE POPULATION

    num_gpe_surrogates, surr_rate = get_pop_parameters('GPE', 'num_surrogates',
                                                      'surrogate_rate')
    
    if num_gpe_surrogates > 0:
        pop_gpe_surrogate = Population(num_gpe_surrogates, 
                                       sim.SpikeSourcePoisson(rate=surr_rate),
                                       label='GpeSurrogate')
    else:
        pop_gpe_surrogate = None

    #---------------------------------------------------------------------------
    # CTX POPULATION

    # CTX spike sources
    T_burst, dur_burst, f_intra, f_inter, f_background = get_pop_parameters(
        'CTX', 'T_burst', 'dur_burst', 'f_intra', 'f_inter', 'f_background')
    synchronous, sync_fraction = get_pop_parameters(
        'CTX', 'synchronous', 'synchronized_fraction')

    # Command line args can override Beta frequency from config
    if burst_frequency is not None:
        T_burst = 1.0 / burst_frequency * 1e3


    def spiketimes_for_ctx(cell_indices):
        """
        Generate spike times for Cortex cell i.

        @param  cell_indices : list(int)
                Local indices of cells (NOT index in entire population)
        """
        if synchronous:
            make_bursts = make_oscillatory_bursts
        else:
            make_bursts = make_variable_bursts

        # Some cells don't burst but fire at inter-burst firing rate
        synchronized_cells = rank_rng.choice(
            cell_indices, int(sync_fraction * len(cell_indices)), replace=False)

        spiketimes_for_index = []
        for i in cell_indices:
            if i in synchronized_cells:
                # Spiketimes for bursting cells
                burst_gen = make_bursts(T_burst, dur_burst, f_intra, f_inter,
                                        rng=rank_rng, max_dur=sim_dur)
                spiketimes = Sequence(np.fromiter(burst_gen, float))
            else:
                # Spiketimes for background activity
                number = int(2 * sim_dur * f_background / 1e3)
                if number == 0:
                    spiketimes = Sequence([])
                else:
                    spiketimes = Sequence(np.add.accumulate(
                        rank_rng.exponential(1e3/f_background, size=number)))
            spiketimes_for_index.append(spiketimes)
        return spiketimes_for_index


    pop_ctx = Population(
                ncell_per_pop,
                sim.SpikeSourceArray(spike_times=spiketimes_for_ctx),
                label='CTX')

    #---------------------------------------------------------------------------
    # STR POPULATION

    # STR spike sources
    T_burst, dur_burst, f_intra, f_inter = get_pop_parameters(
        'STR', 'T_burst', 'dur_burst', 'f_intra', 'f_inter')

    def spiketimes_for_str(cell_indices):
        """
        Generate spike times for Striatum cell i.

        (taken from PyNN examples -> returns pyNN.Sequence generator)
        """
        def sequence_gen():
            burst_gen = make_variable_bursts(
                T_burst, dur_burst, f_intra, f_inter,
                rng=rank_rng, max_dur=sim_dur)
            bursty_spikes = np.fromiter(burst_gen, float)
            return Sequence(bursty_spikes)
        if hasattr(cell_indices, "__len__"):
            return [sequence_gen() for j in cell_indices]
        else:
            return sequence_gen()
    
    pop_str = Population(
                ncell_per_pop, 
                sim.SpikeSourceArray(spike_times=spiketimes_for_str),
                label='STR')


    # Noise sources
    # noise_gpe = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=50.0),
    #                 label='NOISE')

    all_pops = {pop.label : pop for pop in Population.all_populations}
    all_proj = {pop.label : {} for pop in Population.all_populations}

    # Register LFP calculation callbacks if manual calculation
    # if calculate_lfp:
    #     def init_lfp_calculation():
    #         """
    #         Initialize the LFP calculation (events must be added after finitialize().
    #         """
    #         lfp_rec_dt = sim.state.rec_dt
    #         for t in np.arange(lfp_rec_dt, sim.state.duration+lfp_rec_dt, lfp_rec_dt):
    #             sim.state.cvode.event(t, pop_stn.calculate_lfp)
    #     finit_handlers.append(h.FInitializeHandler(0, init_lfp_calculation))

    ############################################################################
    # CONNECTIONS
    ############################################################################

    # see notes in original model.py (non-parameterized)

    # Allowed synapse types (for creation from config file)
    synapse_types = {
        "GluSynapse": GluSynapse,
        "GabaSynapse": GabaSynapse,
        "GabaSynTmHill" : GabaSynTmHill,
    }

    ############################################################################
    # TO GPE
    print("{} start phase: GPE AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)

    stn_gpe_connector = connector_from_config('STN', 'GPE')
    stn_gpe_connector.rng = shared_rng_pynn

    stn_gpe_syn = synapse_from_config('STN', 'GPE')
    

    stn_gpe_EXC = sim.Projection(pop_stn, pop_gpe, 
                                 connector=stn_gpe_connector,
                                 synapse_type=stn_gpe_syn,
                                 receptor_type='distal.AMPA+NMDA')

    all_proj['STN']['GPE'] = stn_gpe_EXC

    #---------------------------------------------------------------------------
    # GPE -> GPE (inhibitory)

    # Distance-calculation: wrap-around in along x-axis
    x_max = ncell_gpe * gpe_grid.dx
    gpe_gpe_space = space.Space(periodic_boundaries=((0, x_max), None, None))

    # Connect to four neighboring neurons (N_PRE=8)
    gpe_gpe_connector = connector_from_config('GPE', 'GPE')
    gpe_gpe_connector.rng = shared_rng_pynn
    

    # Synapse type
    gpe_gpe_syn = synapse_from_config('GPE', 'GPE')

    gpe_gpe_INH = sim.Projection(pop_gpe, pop_gpe, 
                                 connector=gpe_gpe_connector,
                                 synapse_type=gpe_gpe_syn,
                                 receptor_type='proximal.GABAA+GABAB',
                                 space=gpe_gpe_space)

    all_proj['GPE']['GPE'] = gpe_gpe_INH

    #---------------------------------------------------------------------------
    # STR -> GPE (inhibitory)

    str_gpe_connector = connector_from_config('STR', 'GPE')
    str_gpe_connector.rng = shared_rng_pynn

    str_gpe_syn = synapse_from_config('STR', 'GPE')

    str_gpe_INH = sim.Projection(pop_str, pop_gpe,
                                 connector=str_gpe_connector,
                                 synapse_type=str_gpe_syn,
                                 receptor_type='proximal.GABAA+GABAB')

    all_proj['STR']['GPE'] = str_gpe_INH

    ############################################################################
    # TO STN
    print("{} start phase: STN AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)

    gpe_stn_connector = connector_from_config('GPE', 'STN')
    gpe_stn_connector.rng = shared_rng_pynn

    gpe_stn_syn = synapse_from_config('GPE', 'STN')

    gpe_stn_INH = sim.Projection(
                        pop_gpe, pop_stn,
                        connector=gpe_stn_connector,
                        synapse_type=gpe_stn_syn,
                        receptor_type='proximal.GABAA+GABAB')

    all_proj['GPE']['STN'] = gpe_stn_INH
    
    #---------------------------------------------------------------------------
    # GpeSurrogate -> STN (inhibitory)

    if pop_gpe_surrogate is not None:

        gpesurr_stn_INH = sim.Projection(
            pop_gpe_surrogate, pop_stn,
            connector=connector_from_config('GpeSurrogate', 'STN', shared_rng_pynn),
            synapse_type=synapse_from_config('GpeSurrogate', 'STN'),
            receptor_type='proximal.GABAA+GABAB')

        all_proj['GpeSurrogate']['STN'] = gpesurr_stn_INH

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)

    ctx_stn_syn = synapse_from_config('CTX', 'STN')

    ctx_stn_connector = connector_from_config('CTX', 'STN')
    ctx_stn_connector.rng = shared_rng_pynn

    ctx_stn_EXC = sim.Projection(
                        pop_ctx, pop_stn,
                        connector=ctx_stn_connector,
                        synapse_type=ctx_stn_syn,
                        receptor_type='distal.AMPA+NMDA')

    all_proj['CTX']['STN'] = ctx_stn_EXC

    #---------------------------------------------------------------------------
    # STN -> STN (excitatory)

    stn_stn_syn = synapse_from_config('STN', 'STN')

    stn_stn_connector = connector_from_config('STN', 'STN')
    stn_stn_connector.rng = shared_rng_pynn

    stn_stn_EXC = sim.Projection(
                        pop_stn, pop_stn,
                        connector=stn_stn_connector,
                        synapse_type=stn_stn_syn,
                        receptor_type='distal.AMPA+NMDA')

    all_proj['STN']['STN'] = stn_stn_EXC

    ############################################################################
    # RECORDING
    ############################################################################
    print("{} start phase: RECORDING.".format(mpi_rank))

    traces_allpops = {
        'Vm':       {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
        # 'gAMPA{:d}': {'syn':'GLUsyn[0]', 'var':'g_AMPA'},
        # 'gNMDA{:d}': {'syn':'GLUsyn[::2]', 'var':'g_NMDA'},
        # 'gGABAA{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAA'},
        # 'gGABAB{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAB'},
    }
    for pop in [pop_gpe, pop_stn]:
        pop.record(traces_allpops.items(), sampling_interval=.05)

    traces_stn = {
        "gGABAA{:d}": {"syn":"GABAsyn2[:]", "var":"g_GABAA"},
        "gGABAB{:d}": {"syn":"GABAsyn2[:]", "var":"g_GABAB"},
        "iGABA{:d}": {"syn":"GABAsyn2[:]", "var":"i"},
    }
    pop_stn.record(traces_stn.items(), sampling_interval=.05)

    for pop in all_pops.values():
        pop.record(['spikes'], sampling_interval=.05)

    if calculate_lfp:
        pop_stn.record(['lfp'], sampling_interval=.05)

    
    ############################################################################
    # INITIALIZE & SIMULATE
    ############################################################################
    print("{} start phase: SIMULATE.".format(mpi_rank))

    # Set physiological conditions
    h.celsius = 36.0
    h.set_aCSF(4) # Hoc function defined in Gillies code
    
    # Simulation statistics
    num_segments = sum((sec.nseg for sec in h.allsec()))
    num_cell = sum((1 for sec in h.allsec()))
    print("Will simulate {} sections ({} compartments) for {} seconds on MPI rank {}.".format(
            num_cell, num_segments, sim_dur, mpi_rank))
    tstart = time.time()
    outdir, filespec = os.path.split(output)
    progress_file = os.path.join(outdir, '{}_sim_progress.log'.format(
        datetime.fromtimestamp(tstart).strftime('%Y.%m.%d-%H.%M.%S')))
    report_interval = 50.0 # (ms) in simulator time
    last_report_time = tstart

    # Times for writing out data to file
    if transient_period is None:
        transient_period = 1000.0 # (ms)
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
    write_times = list(reversed(np.arange(first_write_time, sim_dur+write_interval,
                                          write_interval)))
    last_write_time = 0.0
    
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
        if abs(sim.state.t - write_times[-1]) <= 5.0:
            suffix = "_{:.0f}ms-{:.0f}ms".format(last_write_time, sim.state.t)
            for pop in all_pops.values():
                write_population_data(pop, output, suffix, gather=True, clear=True)
            write_times.pop()
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

    # Final write of recorded variables
    suffix = "_{:.0f}ms-{:.0f}ms".format(last_write_time, sim.state.t)
    for pop in all_pops.values():
        write_population_data(pop, output, suffix, gather=True, clear=True)


    ############################################################################
    # WRITE PARAMETERS
    ############################################################################
    print("{} start phase: INTEGRITY CHECK.".format(mpi_rank))

    # NOTE: - any call to Population.get() Projection.get() does a ParallelContext.gather()
    #       - cannot perform any gather() operations before initializing MPI transfer
    #       - must do gather() operations on all nodes
    saved_params = {}
    for pre_pop, post_pops in all_proj.iteritems():
        saved_params.setdefault(pre_pop, {})
        for post_pop, proj in post_pops.iteritems():

            # Plot connectivity matrix ('O' is connection, ' ' is no connection)
            utf_matrix, float_matrix = connection_plot(proj)
            nprint("{}->{} connectivity matrix (dim[0,1] = [src,target]: \n".format(
                proj.pre.label, proj.post.label) + utf_matrix)

            # This does an mpi gather() on all the parameters
            pre_post_params = np.array(proj.get(["delay", "weight"], format="list", 
                                       gather='all', multiple_synapses='sum'))
            mind = min(pre_post_params[:,2])
            maxd = max(pre_post_params[:,2])
            minw = min(pre_post_params[:,3])
            maxw = max(pre_post_params[:,3])
            nprint("Error check for projection {pre}->{post}:\n"
                  "    - delay  [min, max] = [{mind}, {maxd}]\n"
                  "    - weight [min, max] = [{minw}, {maxw}]\n".format(
                    pre=pre_pop, post=post_pop, mind=mind, maxd=maxd,
                    minw=minw, maxw=maxw))

            # Append to saved dictionary
            proj_params = saved_params[pre_pop].setdefault(post_pop, {})
            proj_params['conn_matrix'] = float_matrix
            proj_params['weight_delay_list'] = pre_post_params


    # Write model parameters 
    if mpi_rank==0 and output is not None:
        outdir, extension = output.split('*')

        # Save projection parameters
        import pickle
        extension = extension[:-4] + '.pkl'
        params_outfile = "{dir}pop-parameters{ext}".format(dir=outdir, ext=extension)
        with open(params_outfile, 'wb') as fout:
            pickle.dump(saved_params, fout)

    ############################################################################
    # PLOT DATA
    ############################################################################

    if mpi_rank==0 and with_gui:
        # Only plot on one process, and if GUI available
        import analysis
        pop_neo_data = {
            pop.label: pop.get_data().segments[0] for pop in all_pops.values()
        }
        analysis.plot_population_signals(pop_neo_data)

    if export_locals:
        globals().update(locals())


if __name__ == '__main__':
    # Parse arguments passed to `python model.py [args]`
    import argparse

    parser = argparse.ArgumentParser(description='Run basal ganglia network simulation')

    parser.add_argument('-d', '--dur', nargs='?', type=float, default=500.0,
                        dest='sim_dur', help='Simulation duration')

    parser.add_argument('-n', '--ncell', nargs='?', type=int, default=30,
                        dest='ncell_per_pop', help='Number of cells per population')

    parser.add_argument('-s', '--seed', nargs='?', type=int, default=None,
                        dest='seed', help='Seed for random number generator')

    parser.add_argument('-b', '--burst', nargs='?', type=float, default=None,
                        dest='burst_frequency', help='Beta bursting frequency')

    parser.add_argument('-wi', '--write-interval', nargs='?', type=float, default=None,
                        dest='max_write_interval',
                        help='Interval between successive write out of recording data')

    parser.add_argument('-tp', '--transient-period', nargs='?', type=float, default=None,
                        dest='transient_period',
                        help=('Duration of transient period at start of simulation. ' 
                              'First data write-out is after transient period'))

    parser.add_argument('--lfp',
                        dest='calculate_lfp', action='store_true',
                        help='Calculate Local Field Potential.')
    parser.add_argument('--no-lfp',
                        dest='calculate_lfp', action='store_false',
                        help='Calculate Local Field Potential.')
    parser.set_defaults(calculate_lfp=None)

    parser.add_argument('-g', '--gui',
                        dest='with_gui',
                        action='store_true',
                        help='Enable graphical output')
    parser.add_argument('-ng', '--no-gui',
                        dest='with_gui',
                        action='store_false',
                        help='Enable graphical output')
    parser.set_defaults(with_gui=False)

    parser.add_argument('-o', '--outdir', nargs='?', type=str,
                        default='~/storage/',
                        dest='output',
                        help='Output destination in format \'/outdir/*.ext\''
                             ' or /path/to/outdir/ with trailing slash')

    parser.add_argument('-p', '--progress',
                        dest='report_progress', action='store_true',
                        help='Report progress periodically to progress file')
    parser.set_defaults(report_progress=False)

    parser.add_argument('-c', '--config', nargs=1, type=str,
                        metavar='/path/to/config.json',
                        dest='config_file',
                        help='Configuration file in JSON format')

    parser.add_argument('-id', '--identifier', nargs=1, type=str,
                        metavar='<job identifer>',
                        dest='job_id',
                        help='Job identifier to tag the simulation')

    args = parser.parse_args() # Namespace object
    parsed_dict = vars(args) # Namespace to dict

    # Parse config JSON file to dict
    config_file = os.path.expanduser(parsed_dict.pop('config_file')[0])
    config_name, ext = os.path.splitext(os.path.basename(config_file))
    sim_config = fileutils.parse_json_file(config_file, nonstrict=True)
    parsed_dict['config'] = sim_config
    
    # Post process output specifier
    out_basedir = parsed_dict['output']
    if out_basedir is None or out_basedir == '': # shell can pass empty string
        out_basedir = '~/storage'
    job_id = parsed_dict.pop('job_id')[0]
    time_now = time.time()
    timestamp = datetime.fromtimestamp(time_now).strftime('%Y.%m.%d')

    # Default output directory
    # NOTE: don't use timestamp -> mpi ranks will make different filenames
    out_subdir = '{stamp}_job-{job_id}_{config_name}'.format(
                                            stamp=timestamp,
                                            job_id=job_id,
                                            config_name=config_name)

    # File names for data files
    # Default output format is hdf5 / NIX io
    filespec = '*_{stamp}_pop-{ncell}_dur-{dur}_job-{job_id}.mat'.format(
                                            ncell=parsed_dict['ncell_per_pop'],
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
    
    # Run the simulation
    run_simple_net(**parsed_dict)
