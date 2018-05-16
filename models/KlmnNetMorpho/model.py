"""
Basal Ganglia network model consisting of morphologically detailed
cell models for the major cell types.

@author     Lucas Koelman

@date       20/03/2017

@see        PyNN manual for building networks:
                http://neuralensemble.org/docs/PyNN/building_networks.html
            PyNN examples of networks:
                https://github.com/NeuralEnsemble/PyNN/tree/master/examples

USAGE
-----

To run using MPI, you can use the following command:

    >>> mpiexec -np 8 python model.py

"""
import time
import numpy as np

# PyNN library
import pyNN.neuron as sim
from pyNN import space
from pyNN.utility import init_logging, connection_plot

# Custom PyNN extensions
from extensions.pynn.connection import NativeSynapse, GluSynapse, GabaSynapse # , SynapseFromDB
from extensions.pynn.recording import TraceSpecRecorder
sim.Population._recorder_class = TraceSpecRecorder

# Custom NEURON mechanisms
import os.path
script_dir = os.path.dirname(__file__)
sim.simulator.load_mechanisms(os.path.join('..', '..', 'mechanisms', 'synapses'))

# Custom cell models
import models.GilliesWillshaw.gillies_pynn_model as gillies
import models.Gunay2008.gunay_pynn_model as gunay

# Our physiological parameters
from cellpopdata.physiotypes import Populations as PopID, ParameterSource as ParamSrc
# from cellpopdata.cellpopdata import CellConnector


def test_STN_population(ncell_per_pop=5, sim_dur=500.0, export_locals=True):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.
    """

    init_logging(logfile=None, debug=False)
    mpi_rank = sim.setup(use_cvode=False)
    h = sim.h

    seed = sim.state.mpi_rank + sim.state.native_rng_baseseed
    numpy_rng = np.random.RandomState(seed)

    # STN cell population
    pop_stn = sim.Population(ncell_per_pop, gillies.StnCellType(), label='STN')
    pop_stn.pop_id = PopID.STN
    pop_stn.initialize(v=-63.0)

    # RECORDING
    traces = {
        'Vm': {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
    }
    pop_stn.record(traces.items(), sampling_interval=.05)


    # INITIALIZE + RUN

    # Set physiological conditions
    h.celsius = 36.0
    h.set_aCSF(4) # Hoc function defined in Gillies code

    print("CVode state: {}".format(sim.state.cvode.active()))
    tstart = time.time()
    sim.run(sim_dur)
    tstop = time.time()
    cputime = tstop - tstart
    num_segments = sum((sec.nseg for sec in h.allsec()))
    print("Simulated {} segments for {} ms in {} ms CPU time".format(
            num_segments, sim.state.tstop, cputime))


    # PLOTTING
    import matplotlib.pyplot as plt

    # Fetch recorded data
    pop = pop_stn
    pop_data = pop.get_data() # Neo Block object
    segment = pop_data.segments[0] # segment = all data with common time basis

    # Plot each signal
    for signal in segment.analogsignals:
        # one figure per trace type
        fig, axes = plt.subplots(signal.shape[1], 1)
        fig.suptitle("Population {} - trace {}".format(
                        pop.label, signal.name))

        time_vec = signal.times
        y_label = "{} ({})".format(signal.name, signal.units._dimensionality.string)

        for i_cell in range(signal.shape[1]):
            ax = axes[i_cell]
            label = "cell {}".format(signal.annotations['source_ids'][i_cell])
            ax.plot(time_vec, signal[:, i_cell], label=label)
            ax.set_ylabel(y_label)
            ax.legend()

    plt.show(block=False)

    if export_locals:
        globals().update(locals())


def run_simple_net(ncell_per_pop=30, sim_dur=500.0, export_locals=True):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.
    """

    init_logging(logfile=None, debug=False)
    mpi_rank = sim.setup(use_cvode=False)
    h = sim.h

    seed = sim.state.mpi_rank + sim.state.native_rng_baseseed
    numpy_rng = np.random.RandomState(seed)

    # Parameters database
    # params_db = CellConnector(
    #     'Parkinsonian', numpy_rng,
    #     preferred_sources=[ParamSrc.Chu2015, ParamSrc.Fan2012, ParamSrc.Atherton2013],
    #     preferred_mechanisms=['GLUsyn', 'GABAsyn'])

    # NOTE: if wrapped synapses like GluSynapse and GabaSynapse are used, any
    #       parameter can be a function of distance or cell index. For complex
    #       functions using math and numpy modules, write a function. For
    #       simple ones, you can use a string as as well.

    ############################################################################
    # POPULATIONS
    ############################################################################
    # Define each cell population with its cell type, number of cells

    # STN cell population
    stn_grid = space.Line(x0=0.0, dx=50.0,
                          y=0.0, z=0.0)
    
    stn_type = gillies.StnCellType(with_receptors=['GLUsyn', 'GABAsyn'])

    ncell_stn = ncell_per_pop
    pop_stn = sim.Population(ncell_stn, 
                             cellclass=stn_type, 
                             label='STN',
                             structure=stn_grid)
    
    pop_stn.pop_id = PopID.STN
    pop_stn.initialize(v=-63.0)


    # GPe cell population
    gpe_grid = space.Line(x0=0.0, dx=50.0,
                          y=1e6, z=0.0)

    gpe_type = gunay.GPeCellType(with_receptors=['GLUsyn', 'GABAsyn'])

    ncell_gpe = ncell_per_pop
    pop_gpe = sim.Population(ncell_gpe, 
                             cellclass=gpe_type,
                             label='GPE',
                             structure=gpe_grid)

    pop_gpe.pop_id = PopID.GPE
    pop_gpe.initialize(v=-63.0)


    # CTX spike sources
    pop_ctx = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=20.0),
                    label='CTX')
    pop_ctx.pop_id = PopID.CTX


    # STR spike sources
    pop_str = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=20.0),
                    label='STR')
    pop_str.pop_id = PopID.STR


    # Noise sources
    noise_gpe = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=50.0),
                    label='NOISE')

    all_pops = {pop.pop_id : pop for pop in [pop_gpe, pop_stn, pop_ctx]}

    ############################################################################
    # CONNECTIONS
    ############################################################################

    # Create connection
    conn_all2all = sim.AllToAllConnector()
    conn_allp05 = sim.FixedProbabilityConnector(0.5)

    # db_syn = SynapseFromDB(parameter_database=params_db) # our custom synapse class
    # TODO: set get_physiological_parameters(adjust_gsyn_msr=XXX)
    #       - you can implement it in get_physiological_parameters() so that
    #         the returned conductance is adjusted if the param is included
    #         in custom_params. E.g. if it's in custom_params and adjust_gsyn_msr
    #         is truthy: use that param to scale, else use adjust_gsyn_msr.

    ############################################################################
    # TO GPE

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)

    stn_gpe_syn = GluSynapse(**{
        'weight':       1.0,
        'delay':        2.0, # [ms] delay from literature
        'U1':           0.1, # baseline release probability
        'tau_rec':      200.0, # [ms] recovery from STD
        'tau_facil':    800.0, # [ms] recovery from facilitation
        # AMPA receptor
        'gmax_AMPA':    0.025 / 0.1 * 1e-3, # [uS], adjusted for U1
        'tau_r_AMPA':   1.0, # [ms] rise time
        'tau_d_AMPA':   4.0, # [ms] decay time
        # NMDA receptor
        'gmax_NMDA':    0.0,
    })
    # stn_gpe_syn = sim.StaticSynapse(weight='1/(1+d)', delay=1.0)
    # stn_gpe_syn = NativeSynapse(
    #     mechanism='GLUsyn',
    #     mechanism_parameters={
    #         'netcon:weight[0]': 1.0,
    #         'netcon:delay':     2.0, # [ms] delay from literature
    #         'syn:U1':           0.1, # baseline release probability
    #         'syn:tau_rec':      200.0, # [ms] recovery from STD
    #         'syn:tau_facil':    800.0, # [ms] recovery from facilitation
    #         # AMPA receptor
    #         'syn:gmax_AMPA':    0.025 / 0.1 * 1e-3, # [uS], adjusted for U1
    #         'syn:tau_r_AMPA':   1.0, # [ms] rise time
    #         'syn:tau_d_AMPA':   4.0, # [ms] decay time
    #         # NMDA receptor
    #         'syn:gmax_NMDA':    0.0,
    #     })

    # TODO: distance-based, use variable weight or variable number of synapses
    stn_gpe_connector = conn_allp05

    stn_gpe_EXC = sim.Projection(pop_stn, pop_gpe, 
                                 connector=stn_gpe_connector,
                                 synapse_type=stn_gpe_syn,
                                 receptor_type='distal_dend.AMPA+NMDA')

    stn_gpe_EXC.set(gmax_NMDA=1.0)

    #---------------------------------------------------------------------------
    # GPE -> GPE (inhibitory)

    # Distance-calculation: wrap-around in along x-axis
    x_max = ncell_gpe * gpe_grid.dx
    gpe_gpe_space = space.Space(periodic_boundaries=((0, x_max), None, None))

    # Connect to four neighboring neurons
    dist_fun_conn = 'd < 205'
    gpe_gpe_connector = sim.DistanceDependentProbabilityConnector(
                                dist_fun_conn,
                                allow_self_connections=False)
    # NOTE: for math and numpy functions: use function instead of string
    dist_fun_weight = '(d<50)*1.0 + (d>=50)*exp(-(d-50)/100)'

    # Calibrated using Miguelez (2012)
    gpe_gpe_syn = NativeSynapse(
        mechanism='GABAsyn',
        mechanism_parameters={
            'netcon:weight[0]': dist_fun_weight,
            'netcon:delay':     0.5, # [ms] delay from literature
            'syn:U1':           0.2, # baseline release probability
            'syn:tau_rec':      400.0, # [ms] recovery from STD
            'syn:tau_facil':    1.0, # [ms] recivery from facilitation
            # AMPA receptor
            'syn:gmax_GABAA':   0.1 / 0.2 * 1e-3, # [uS], adjusted for U1
            'syn:tau_r_GABAA':  2.0, # [ms] rise time
            'syn:tau_d_GABAA':  6.0, # [ms] decay time
            # NMDA receptor
            'syn:gmax_GABAB':   0.1 * 1e-3, # [uS]
            'syn:tau_r_GABAB':  5.0, # [ms] rise time of first species in cascade
            'syn:tau_d_GABAB':  10.0, # [ms] decay time of first species cascade
            
        })

    # TODO: set connector and weight dependence
    gpe_gpe_INH = sim.Projection(pop_gpe, pop_gpe, 
                                 connector=gpe_gpe_connector,
                                 synapse_type=gpe_gpe_syn,
                                 receptor_type='proximal_dend.GABAA+GABAB',
                                 space=gpe_gpe_space)

    #---------------------------------------------------------------------------
    # STR -> GPE (inhibitory)

    # TODO: calibrate using refs Steiner BG HB, facilation params
    str_gpe_syn = NativeSynapse(
        mechanism='GABAsyn',
        mechanism_parameters={
            'netcon:weight[0]': 1.0,
            'netcon:delay':     5.0, # [ms] delay from literature
            'syn:U1':           0.2, # baseline release probability
            'syn:tau_rec':      1.0, # [ms] recovery from STD
            'syn:tau_facil':    500.0, # [ms] recivery from facilitation
            # GABA-A receptor
            'syn:gmax_GABAA':   0.2 * 1e-3, # [uS], adjusted for U1
            'syn:tau_r_GABAA':  2.0, # [ms] rise time
            'syn:tau_d_GABAA':  6.0, # [ms] decay time
            # GABA-B receptor
            'syn:gmax_GABAB':   0.0, # [uS], adjusted for U1
        })

    # TODO connection pattern STR -> GPE
    str_gpe_connector = conn_allp05
    str_gpe_INH = sim.Projection(pop_str, pop_gpe,
                                 connector=str_gpe_connector,
                                 synapse_type=str_gpe_syn,
                                 receptor_type='proximal_dend.GABAA+GABAB')


    #---------------------------------------------------------------------------
    # NOISE -> GPE (excitatory)
    noise_syn = NativeSynapse(
        mechanism='GLUsyn',
        mechanism_parameters={
            'netcon:weight[0]': 1.0,
            'netcon:delay':     5.0, # [ms] delay from literature
        })

    noise_connector = sim.FixedProbabilityConnector(0.5)
    
    noise_gpe_EXC = sim.Projection(noise_gpe, pop_gpe, 
                                   connector=noise_connector,
                                   synapse_type=noise_syn,
                                   receptor_type='proximal_dend.AMPA+NMDA')
    
    noise_gpe_EXC.preferred_param_sources = [
        ParamSrc.Chu2015, ParamSrc.Fan2012, ParamSrc.Atherton2013]

    ############################################################################
    # TO STN

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)
    gpe_stn_syn = NativeSynapse(
        mechanism='GABAsyn',
        mechanism_parameters={
            'netcon:weight[0]': 1.0,
            'netcon:delay':     4.0, # [ms] delay from literature
            'syn:U1':           0.2, # baseline release probability
            'syn:tau_rec':      17300.0, # [ms] recovery from STD
            'syn:tau_facil':    1.0, # [ms] recivery from facilitation
            # AMPA receptor
            'syn:gmax_GABAA':   7.0 / 0.2 * 1e-3, # [uS], adjusted for U1
            'syn:tau_r_GABAA':  2.0, # [ms] rise time
            'syn:tau_d_GABAA':  6.0, # [ms] decay time
            # NMDA receptor
            'syn:gmax_GABAB':   7.0 * 1e-3, # [uS]
            'syn:tau_r_GABAB':  5.0, # [ms] rise time of cascade first species
            'syn:tau_d_GABAB':  10.0, # [ms] decay time of cascade first species
            
        })

    # TODO: GPE -> STN projection pattern
    gpe_stn_INH = sim.Projection(
                        pop_gpe, pop_stn,
                        connector=conn_allp05,
                        synapse_type=gpe_stn_syn,
                        receptor_type='proximal_dend.GABAA+GABAB')

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)

    ctx_stn_syn = NativeSynapse(
        mechanism='GLUsyn',
        mechanism_parameters={
            'netcon:weight[0]': 1.0,
            'netcon:delay':     5.9, # [ms] delay from literature
            'syn:U1':           0.7, # baseline release probability
            'syn:tau_rec':      200.0, # [ms] recovery from STD
            'syn:tau_facil':    1.0, # [ms] recivery from facilitation
            # AMPA receptor
            'syn:gmax_AMPA':    3.44 / 0.7 * 1e-3, # [uS], adjusted for U1
            'syn:tau_r_AMPA':   1.0, # [ms] rise time
            'syn:tau_d_AMPA':   4.0, # [ms] decay time
            # NMDA receptor
            'syn:gmax_NMDA':    7.0 / 0.7 * 1e-3, # [uS], adjusted for U1
            'syn:tau_r_NMDA':   3.7, # [ms] rise time
            'syn:tau_d_NMDA':   80.0, # [ms] decay time
            
        })

    # TODO: CTX -> STN projection pattern
    ctx_stn_EXC = sim.Projection(
                        pop_ctx, pop_stn,
                        connector=conn_allp05,
                        synapse_type=ctx_stn_syn,
                        receptor_type='distal_dend.AMPA+NMDA')

    #---------------------------------------------------------------------------
    # Plot connectivity matrix
    for prj in [stn_gpe_EXC, gpe_stn_INH]:
        print(u"{} connectivity matrix: \n".format(prj) + connection_plot(prj))

    ############################################################################
    # RECORDING
    ############################################################################

    traces_allpops = {
        'Vm':       {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
        'gAMPA{:d}': {'syn':'GLUsyn[0]', 'var':'g_AMPA'},
        # 'gNMDA{:d}': {'syn':'GLUsyn[::2]', 'var':'g_NMDA'},
        # 'gGABAA{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAA'},
        # 'gGABAB{:d}': {'syn':'GABAsyn[1]', 'var':'g_GABAB'},
    }
    for pop in [pop_gpe, pop_stn]:
        pop.record(traces_allpops.items(), sampling_interval=.05)

    for pop in all_pops.values():
        pop.record(['spikes'], sampling_interval=.05)


    ############################################################################
    # INITIALIZE + RUN
    ############################################################################

    # Set physiological conditions
    h.celsius = 36.0
    h.set_aCSF(4) # Hoc function defined in Gillies code

    print("CVode state: {}".format(sim.state.cvode.active()))
    tstart = time.time()
    sim.run(sim_dur)
    tstop = time.time()
    cputime = tstop - tstart
    num_segments = sum((sec.nseg for sec in h.allsec()))
    print("Simulated {} segments for {} ms in {} ms CPU time".format(
            num_segments, sim.state.tstop, cputime))

    ############################################################################
    # PLOTTING
    ############################################################################
    import matplotlib.pyplot as plt

    # Plot spikes
    fig_spikes, axes_spikes = plt.subplots(len(all_pops), 1)
    fig_spikes.suptitle('Spikes for each population')

    for i_pop, pop in enumerate(all_pops.values()):

        pop_data = pop.get_data() # Neo Block object
        segment = pop_data.segments[0] # segment = all data with common time basis

        ax = axes_spikes[i_pop]
        for spiketrain in segment.spiketrains:
            y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
            ax.plot(spiketrain, y, '.')
            ax.set_ylabel(pop.label)


        for signal in segment.analogsignals:
            # one figure per trace type
            fig, axes = plt.subplots(signal.shape[1], 1)
            fig.suptitle("Population {} - trace {}".format(
                            pop.label, signal.name))

            # signal matrix has one cell signal per column
            time_vec = signal.times
            y_label = "{} ({})".format(signal.name, signal.units._dimensionality.string)

            for i_cell in range(signal.shape[1]):
                ax = axes[i_cell]
                label = "cell {}".format(signal.annotations['source_ids'][i_cell])
                ax.plot(time_vec, signal[:, i_cell], label=label)
                ax.set_ylabel(y_label)
                ax.legend()

    plt.show(block=False)

    if export_locals:
        globals().update(locals())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run basal ganglia network simulation')

    parser.add_argument('-d', '--dur', nargs='?', type=float, default=500.0,
                        dest='sim_dur', help='Simulation duration')

    parser.add_argument('-n', '--ncell', nargs='?', type=int, default=5,
                        dest='ncell_per_pop', help='Number of cells per population')

    args = parser.parse_args() # Namespace object

    run_simple_net(**vars(args))
    # test_STN_population(**vars(args))
