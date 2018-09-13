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

To run using MPI, use the `mpirun` or `mpiexec` command like:

`mpirun -n 4 python model_parameterized.py --scale 0.5 --dur 500  --seed 888 --transient-period 0.0 --write-interval 1000 --no-gui -id test1 --outdir ~/storage --config configs/DA-depleted_template.json`

To run from an IPyhton shell, use the %run magic function like:

`%run model_parameterized.py --scale 0.5 --dur 500 --seed 888 --transient-period 0.0 --write-interval 1000 --no-gui -id test1 --outdir ~/storage --config configs/DA-depleted_template.json`


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
from pyNN.random import RandomDistribution
from pyNN.utility import init_logging # connection_plot is bugged
import neo.io

# Custom PyNN extensions
from bgcellmodels.extensions.pynn.connection import (
    GluSynapse, GabaSynapse, GabaSynTmHill, GABAAsynTM,
    GabaSynTm2)
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
import bgcellmodels.models.striatum.Mahon2000_MSN.mahon_pynn_model as mahon
import bgcellmodels.models.interneuron.Golomb2007_FSI.golomb_pynn_model as golomb

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


def make_bursty_spike_generator(bursting_fraction, synchronous, rng,
                                T_burst, dur_burst, f_intra, f_inter,
                                f_background, duration):
    """
    Make generator function that returns bursty spike sequences.
    """
    if synchronous:
        make_bursts = make_oscillatory_bursts
    else:
        make_bursts = make_variable_bursts

    def spike_seq_gen(cell_indices):
        """
        Spike sequence generator

        @param  cell_indices : list(int)
                Local indices of cells (NOT index in entire population)

        @return spiketimes_for_cell : list(Sequence)
                Sequencie of spike times for each cell index
        """
        # Choose cell indices that will emit bursty spike trains
        num_bursting = int(bursting_fraction * len(cell_indices))
        bursting_cells = rng.choice(cell_indices, 
                                    num_bursting, replace=False)

        spiketimes_for_index = []
        for i in cell_indices:
            if i in bursting_cells:
                # Spiketimes for bursting cells
                burst_gen = make_bursts(T_burst, dur_burst, f_intra, f_inter,
                                        rng=rng, max_dur=duration)
                spiketimes = Sequence(np.fromiter(burst_gen, float))
            else:
                # Spiketimes for background activity
                number = int(2 * duration * f_background / 1e3)
                if number == 0:
                    spiketimes = Sequence([])
                else:
                    spiketimes = Sequence(np.add.accumulate(
                        rng.exponential(1e3/f_background, size=number)))
            spiketimes_for_index.append(spiketimes)
        return spiketimes_for_index

    return spike_seq_gen


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
        pop_scale       = 1.0,
        sim_dur         = 500.0,
        export_locals   = True,
        with_gui        = True,
        output          = None,
        report_progress = None,
        config          = None,
        seed            = None,
        calculate_lfp   = None,
        burst_frequency = None,
        dopamine_depleted = None,
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
    - sim_dur = {}
    - output = {}""".format(mpi_rank, sim_dur, output))
    
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
        DD = dopamine_depleted = config['simulation'].get('DD', None)
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
        config_locals = config[post].get('local_context', {})
        syn_type, syn_params = getdictvals(config[post][pre]['synapse'],
                                           'name', 'parameters')
        syn_class = synapse_types[syn_type]
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
    # NOTE:
    # - to query cell model attributes, use population[i]._cell
    print("rank {}: starting phase POPULATIONS.".format(mpi_rank))

    config_pop_labels = [k for k in config.keys() if not k in ('simulation',)]

    #===========================================================================
    # STN POPULATION
    stn_dx, = get_pop_parameters('STN', 'grid_dx')
    stn_grid = space.Line(x0=0.0, dx=stn_dx,
                          y=0.0, z=0.0)
    ncell_stn = int(50.0 * pop_scale)
    
    # FIXME: set electrode coordinates
    stn_cell_params = get_cell_parameters('STN')
    stn_type = gillies.StnCellType(
                        calculate_lfp=calculate_lfp,
                        **stn_cell_params)

    vinit = stn_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    pop_stn = Population(ncell_stn, 
                         cellclass=stn_type, 
                         label='STN',
                         structure=stn_grid,
                         initial_values=initial_values)

    #===========================================================================
    # GPE POPULATION

    # Get common parameters for GPE cells
    gpe_dx, gpe_pop_size, frac_proto, frac_arky = get_pop_parameters(
        'GPE.all', 'grid_dx', 'base_population_size',
        'prototypic_fraction', 'arkypallidal_fraction')
    
    gpe_common_params = get_cell_parameters('GPE.all')

    gpe_grid = space.Line(x0=0.0, dx=gpe_dx,
                          y=1e6, z=0.0)

    #---------------------------------------------------------------------------
    # GPE Prototypic
    
    proto_type = gunay.GpeProtoCellType(**gpe_common_params)
    
    vinit = proto_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    ncell_proto = int(gpe_pop_size * pop_scale * frac_proto)
    pop_gpe_proto = Population(ncell_proto, 
                               cellclass=proto_type,
                               label='GPE.proto',
                               structure=gpe_grid,
                               initial_values=initial_values)

    #---------------------------------------------------------------------------
    # GPE Arkypallidal

    arky_type = gunay.GpeArkyCellType(**gpe_common_params)
    
    vinit = arky_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }

    ncell_arky = int(gpe_pop_size * pop_scale * frac_arky)
    pop_gpe_arky = Population(ncell_arky, 
                              cellclass=arky_type,
                              label='GPE.arky',
                              structure=gpe_grid,
                              initial_values=initial_values)

    #---------------------------------------------------------------------------
    # GPE Surrogate spike sources

    frac_surrogate, surr_rate = get_pop_parameters('GPE.all', 
        'surrogate_fraction', 'surrogate_rate')
    
    ncell_surrogate = int(gpe_pop_size * pop_scale * frac_surrogate)
    if ncell_surrogate > 0:
        pop_gpe_surrogate = Population(ncell_surrogate, 
                                       sim.SpikeSourcePoisson(rate=surr_rate),
                                       label='GPE.surrogate')
    else:
        pop_gpe_surrogate = None

    #---------------------------------------------------------------------------
    # GPE Assembly (Proto + Arky)

    
    if pop_gpe_surrogate is None:
        asm_gpe = sim.Assembly(pop_gpe_proto, pop_gpe_arky, pop_gpe_surrogate, 
                               label='GPE.all')
    else:
        asm_gpe = sim.Assembly(pop_gpe_proto, pop_gpe_arky, label='GPE.all')
    gpe_pop_size = asm_gpe.size

    #===========================================================================
    # STR.MSN POPULATION

    msn_pop_size, = get_pop_parameters('STR.MSN', 'base_population_size')
    msn_cell_params = get_cell_parameters('STR.MSN')

    msn_type = mahon.MsnCellType(**msn_cell_params)

    vinit = msn_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }
    
    pop_msn = Population(
                int(msn_pop_size * pop_scale), 
                cellclass=msn_type,
                label='STR.MSN',
                initial_values=initial_values)

    #===========================================================================
    # STR.FSI POPULATION

    fsi_pop_size, = get_pop_parameters('STR.FSI', 'base_population_size')
    fsi_cell_params = get_cell_parameters('STR.FSI')

    fsi_type = golomb.FsiCellType(**fsi_cell_params)

    vinit = msn_type.default_initial_values['v']
    initial_values={
        'v': RandomDistribution('uniform', (vinit-5, vinit+5), rng=shared_rng_pynn)
    }
    
    pop_fsi = Population(
                int(fsi_pop_size * pop_scale), 
                cellclass=fsi_type,
                label='STR.FSI',
                initial_values=initial_values)

    #===========================================================================
    # CTX POPULATION

    # CTX spike sources
    T_burst, dur_burst, f_intra, f_inter, f_background = get_pop_parameters(
        'CTX', 'T_burst', 'dur_burst', 'f_intra', 'f_inter', 'f_background')
    synchronous, bursting_fraction, ctx_pop_size = get_pop_parameters(
        'CTX', 'synchronous', 'bursting_fraction', 'base_population_size')

    # Command line args can override Beta frequency from config
    if burst_frequency is not None:
        T_burst = 1.0 / burst_frequency * 1e3

    ctx_spike_generator = make_bursty_spike_generator(
                                bursting_fraction=bursting_fraction, 
                                synchronous=synchronous, rng=rank_rng,
                                T_burst=T_burst, dur_burst=dur_burst, 
                                f_intra=f_intra, f_inter=f_inter,
                                f_background=f_background, duration=sim_dur)

    pop_ctx = Population(
        int(ctx_pop_size * pop_scale),
        cellclass=sim.SpikeSourceArray(spike_times=ctx_spike_generator),
        label='CTX')

    #===========================================================================

    all_pops = {pop.label : pop for pop in Population.all_populations}
    all_asm = {asm.label: asm for asm in (asm_gpe,)}
    all_proj = {pop.label : {} for pop in Population.all_populations}
    all_proj[asm_gpe.label] = {} # add Assembly projections manually

    # NativeCellType is common base class for all NEURON cells
    biophysical_pops = [pop for pop in Population.all_populations if isinstance(
                        pop.celltype, sim.cells.NativeCellType)]
    artificial_pops = [pop for pop in Population.all_populations if not isinstance(
                        pop.celltype, sim.cells.NativeCellType)]

    # Update local context for eval() statements
    params_local_context.update(locals())

    ############################################################################
    # CONNECTIONS
    ############################################################################

    # see notes in original model.py (non-parameterized)
    # TODO: make sure every entry in JSON config is constructed

    # Allowed synapse types (for creation from config file)
    synapse_types = {
        "GluSynapse": GluSynapse,
        "GABAAsynTM": GABAAsynTM,
        "GabaSynTm2": GabaSynTm2,
    }

    ############################################################################
    # ALL -> STR.MSN
    print("rank {}: starting phase STR.MSN AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # CTX -> STR.MSN (excitatory)
    
    ctx_msn_EXC = sim.Projection(pop_ctx, pop_msn, 
        connector=connector_from_config('CTX', 'STR.MSN', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('CTX', 'STR.MSN'),
        receptor_type='proximal.AMPA+NMDA')

    all_proj['CTX']['STR.MSN'] = ctx_msn_EXC

    #---------------------------------------------------------------------------
    # STR.MSN -> STR.MSN (inhibitory)
    
    msn_msn_INH = sim.Projection(pop_msn, pop_msn, 
        connector=connector_from_config('STR.MSN', 'STR.MSN', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('STR.MSN', 'STR.MSN'),
        receptor_type='proximal.GABAA')

    all_proj['STR.MSN']['STR.MSN'] = msn_msn_INH

    #---------------------------------------------------------------------------
    # STR.FSI -> STR.MSN (inhibitory)
    
    fsi_msn_INH = sim.Projection(pop_fsi, pop_msn, 
        connector=connector_from_config('STR.FSI', 'STR.MSN', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('STR.FSI', 'STR.MSN'),
        receptor_type='proximal.GABAA')

    all_proj['STR.FSI']['STR.MSN'] = fsi_msn_INH

    ############################################################################
    # ALL -> STR.FSI
    print("rank {}: starting phase STR.FSI AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # CTX -> STR.FSI (excitatory)

    ctx_fsi_EXC = sim.Projection(pop_ctx, pop_fsi, 
        connector=connector_from_config('CTX', 'STR.FSI', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('CTX', 'STR.FSI'),
        receptor_type='proximal.AMPA+NMDA')

    all_proj['CTX']['STR.FSI'] = ctx_fsi_EXC

    #---------------------------------------------------------------------------
    # STR.FSI -> STR.FSI (inhibitory)

    fsi_fsi_INH = sim.Projection(pop_fsi, pop_fsi, 
        connector=connector_from_config('STR.FSI', 'STR.FSI', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('STR.FSI', 'STR.FSI'),
        receptor_type='proximal.GABAA')

    all_proj['STR.FSI']['STR.FSI'] = fsi_fsi_INH

    #---------------------------------------------------------------------------
    # GPE -> STR.FSI (inhibitory)

    gpe_fsi_INH = sim.Projection(pop_gpe_arky, pop_fsi, 
        connector=connector_from_config('GPE.arky', 'STR.FSI', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('GPE.arky', 'STR.FSI'),
        receptor_type='proximal.GABAA')

    all_proj['GPE.arky']['STR.FSI'] = fsi_fsi_INH

    ############################################################################
    # ALL -> GPE
    print("rank {}: starting phase GPE AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)

    # stn_gpe_all = sim.Projection(pop_stn, asm_gpe, 
    #     connector=connector_from_config('STN', 'GPE.all', shared_rng_pynn),
    #     synapse_type=synapse_from_config('STN', 'GPE.all'),
    #     receptor_type='distal.AMPA+NMDA')

    # all_proj['STN']['GPE.all'] = stn_gpe_all

    # Cannot connect to GPE.all because it contains spike generators
    stn_gpe_proto = sim.Projection(pop_stn, pop_gpe_proto, 
        connector=connector_from_config('STN', 'GPE.proto', shared_rng_pynn),
        synapse_type=synapse_from_config('STN', 'GPE.proto'),
        receptor_type='distal.AMPA+NMDA')

    all_proj['STN']['GPE.proto'] = stn_gpe_proto

    stn_gpe_arky = sim.Projection(pop_stn, pop_gpe_arky, 
        connector=connector_from_config('STN', 'GPE.arky', shared_rng_pynn),
        synapse_type=synapse_from_config('STN', 'GPE.arky'),
        receptor_type='distal.AMPA+NMDA')

    all_proj['STN']['GPE.arky'] = stn_gpe_arky

    #---------------------------------------------------------------------------
    # GPE -> GPE (inhibitory)

    gpe_gpe_proto = sim.Projection(asm_gpe, pop_gpe_proto, 
        connector=connector_from_config('GPE.all', 'GPE.proto', shared_rng_pynn),
        synapse_type=synapse_from_config('GPE.all', 'GPE.proto'),
        receptor_type='proximal.GABAA+GABAB')

    gpe_gpe_arky = sim.Projection(asm_gpe, pop_gpe_arky, 
        connector=connector_from_config('GPE.all', 'GPE.arky', shared_rng_pynn),
        synapse_type=synapse_from_config('GPE.all', 'GPE.arky'),
        receptor_type='proximal.GABAA+GABAB')

    all_proj['GPE.all']['GPE.proto'] = gpe_gpe_proto
    all_proj['GPE.all']['GPE.arky'] = gpe_gpe_arky

    #---------------------------------------------------------------------------
    # STR -> GPE (inhibitory)

    str_gpe_proto = sim.Projection(pop_msn, pop_gpe_proto,
        connector=connector_from_config('STR.MSN', 'GPE.proto', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('STR.MSN', 'GPE.proto'),
        receptor_type='proximal.GABAA+GABAB')

    all_proj['STR.MSN']['GPE.proto'] = str_gpe_proto

    ############################################################################
    # ALL -> STN
    print("rank {}: starting phase STN AFFERENTS.".format(mpi_rank))

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)

    gpe_proto_stn_INH = sim.Projection(
        pop_gpe_proto, pop_stn,
        connector=connector_from_config('GPE.proto', 'STN', rng=shared_rng_pynn),
        synapse_type=synapse_from_config('GPE.proto', 'STN'),
        receptor_type='proximal.GABAA+GABAB')

    all_proj['GPE.proto']['STN'] = gpe_proto_stn_INH
    
    #---------------------------------------------------------------------------
    # GPE.surrogate -> STN (inhibitory)

    if pop_gpe_surrogate is not None:

        gpesurr_stn_INH = sim.Projection(
            pop_gpe_surrogate, pop_stn,
            connector=connector_from_config('GPE.surrogate', 'STN', shared_rng_pynn),
            synapse_type=synapse_from_config('GPE.surrogate', 'STN'),
            receptor_type='proximal.GABAA+GABAB')

        all_proj['GPE.surrogate']['STN'] = gpesurr_stn_INH

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)

    ctx_stn_EXC = sim.Projection(
        pop_ctx, pop_stn,
        connector=connector_from_config('CTX', 'STN', shared_rng_pynn),
        synapse_type=synapse_from_config('CTX', 'STN'),
        receptor_type='distal.AMPA+NMDA')

    all_proj['CTX']['STN'] = ctx_stn_EXC

    #---------------------------------------------------------------------------
    # STN -> STN (excitatory)

    stn_stn_EXC = sim.Projection(
        pop_stn, pop_stn,
        connector=connector_from_config('STN', 'STN', shared_rng_pynn),
        synapse_type=synapse_from_config('STN', 'STN'),
        receptor_type='distal.AMPA+NMDA')

    all_proj['STN']['STN'] = stn_stn_EXC

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

    traces_biophys = {
        'Vm':       {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
        # Can use SynMech[a:b:c]
        # 'gAMPA{:d}': {'syn':'GLUsyn[0]', 'var':'g_AMPA'},
        # 'gNMDA{:d}': {'syn':'GLUsyn[::2]', 'var':'g_NMDA'},
        # 'gGABAA{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAA'},
        # 'gGABAB{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAB'},
    }

    for pop in biophysical_pops:
        pop.record(traces_biophys.items(), sampling_interval=.05)

    for pop in all_pops.values():
        pop.record(['spikes'], sampling_interval=.05)

    if calculate_lfp:
        pop_stn.record(['lfp'], sampling_interval=.05)

    
    ############################################################################
    # INITIALIZE & SIMULATE
    ############################################################################
    print("rank {}: starting phase SIMULATE.".format(mpi_rank))

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
        if len(write_times) > 0 and abs(sim.state.t - write_times[-1]) <= 5.0:
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
    if sim.state.t - last_write_time > (report_interval + 5.0):
        for pop in all_pops.values():
            write_population_data(pop, output, suffix, gather=True, clear=True)


    ############################################################################
    # WRITE PARAMETERS
    ############################################################################
    print("rank {}: starting phase INTEGRITY CHECK.".format(mpi_rank))

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
            conn_params = ["delay", "weight"]
            gsyn_params = ['gmax_AMPA', 'gmax_NMDA', 'gmax_GABAA', 'gmax_GABAB']
            conn_params.extend([p for p in gsyn_params if p in proj.synapse_type.default_parameters])
            pre_post_params = np.array(proj.get(conn_params, format="list", 
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

    print("rank {}: SIMULATION FINISHED.".format(mpi_rank))


if __name__ == '__main__':
    # Parse arguments passed to `python model.py [args]`
    import argparse

    parser = argparse.ArgumentParser(description='Run basal ganglia network simulation')

    parser.add_argument('-d', '--dur', nargs='?', type=float, default=500.0,
                        dest='sim_dur', help='Simulation duration')

    parser.add_argument('--scale', nargs='?', type=float, default=1.0,
                        dest='pop_scale', help='Scale for population sizes')

    parser.add_argument('--seed', nargs='?', type=int, default=None,
                        dest='seed', help='Seed for random number generator')

    parser.add_argument('--burst', nargs='?', type=float, default=None,
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

    parser.add_argument('--dd',
                        dest='dopamine_depleted', action='store_true',
                        help='Set dopamine depleted condition.')
    parser.add_argument('--dnorm',
                        dest='dopamine_depleted', action='store_false',
                        help='Set dopamine normal condition.')
    parser.set_defaults(dopamine_depleted=None)

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
    timestamp = datetime.fromtimestamp(time_now).strftime('%Y.%m.%d_%H:%M:%S')

    # Default output directory
    # NOTE: don't use timestamp -> mpi ranks will make different filenames
    out_subdir = 'LuNetSGS_{stamp}_job-{job_id}_{config_name}'.format(
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
    
    # Run the simulation
    run_simple_net(**parsed_dict)
