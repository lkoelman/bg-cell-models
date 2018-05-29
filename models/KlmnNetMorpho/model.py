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

    >>> mpiexec -np 8 python model.py [-n numcell -d simdur]

"""
import time
import numpy as np

# PyNN library
import pyNN.neuron as sim
from pyNN import space
from pyNN.utility import init_logging, connection_plot
from pyNN.parameters import Sequence

# Custom PyNN extensions
from extensions.pynn.connection import GluSynapse, GabaSynapse # , SynapseFromDB
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
from cellpopdata.physiotypes import Populations as PopID
#from cellpopdata.physiotypes import ParameterSource as ParamSrc
# from cellpopdata.cellpopdata import CellConnector

from common.spikelib import make_oscillatory_bursts
from common import logutils

# Debug messages
logutils.setLogLevel('quiet', ['bpop_ext'])


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


def run_simple_net(ncell_per_pop=30, sim_dur=500.0, export_locals=True, 
                   with_gui=True):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.
    """

    mpi_rank = sim.setup(use_cvode=False)
    if mpi_rank == 1:
        init_logging(logfile=None, debug=False)

    print("Running net on MPI rank {}\n".format(mpi_rank))
    print("This is node {} ({} of {})\n".format(
          sim.rank(), sim.rank() + 1, sim.num_processes()))

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
    
    stn_type = gillies.StnCellType()

    ncell_stn = ncell_per_pop
    pop_stn = sim.Population(ncell_stn, 
                             cellclass=stn_type, 
                             label='STN',
                             structure=stn_grid)
    
    pop_stn.initialize(v=-63.0)


    # GPe cell population
    gpe_grid = space.Line(x0=0.0, dx=50.0,
                          y=1e6, z=0.0)

    gpe_type = gunay.GPeCellType()

    ncell_gpe = ncell_per_pop
    pop_gpe = sim.Population(ncell_gpe, 
                             cellclass=gpe_type,
                             label='GPE',
                             structure=gpe_grid)

    pop_gpe.initialize(v=-63.0)


    # CTX spike sources
    def spiketimes_for_pop(i):
        """ PyNN wants a Sequence generator """
        def sequence_gen():
            burst_gen = make_oscillatory_bursts(3500.0, 545.0, 50.0, 5.0, rng=numpy_rng)
            bursty_spikes = np.fromiter(burst_gen, float)
            return Sequence(bursty_spikes)
        if hasattr(i, "__len__"):
            return [sequence_gen() for j in i]
        else:
            return sequence_gen()
    
    pop_ctx = sim.Population(
                    ncell_per_pop,
                    sim.SpikeSourceArray(spike_times=spiketimes_for_pop),
                    label='CTX')


    # STR spike sources
    str_base_firing_rate = 1.0
    str_num_poisson_combined = 3
    str_combined_firing_rate = str_base_firing_rate * str_num_poisson_combined
    pop_str = sim.Population(
                    ncell_per_pop, 
                    sim.SpikeSourcePoisson(rate=str_combined_firing_rate),
                    label='STR')


    # Noise sources
    noise_gpe = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=50.0),
                    label='NOISE')

    all_pops = {pop.label : pop for pop in [pop_gpe, pop_stn, pop_ctx]}

    ############################################################################
    # CONNECTIONS
    ############################################################################

    # NOTE: for the different types of connectors available, see:
    #   - overview: https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/neuron/connectors.py
    #   - classes with docstrings: https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/connectors.py
    #   - usage: http://neuralensemble.org/docs/PyNN/connections.html

    # NOTE: distance-based expressions can be used for any parameter
    #   - string expression can only use global functions
    #   - lambda expression can use any function in its closure
    
    # # Example connectors
    # conn_all_to_all =   sim.AllToAllConnector()
    # # Probability based
    # conn_all_p05 =      sim.FixedProbabilityConnector(0.5)
    # conn_all_pdist =    sim.DistanceDependentProbabilityConnector('d<3')
    # # Fixed number
    # conn_5_to_all =     FixedNumberPreConnector(5)
    # conn_all_to_5 =     FixedNumberPostConnector(5)
    # # Indices specifying sparse matrix
    # connector =         sim.FromListConnector([(0, 1), (0, 2), (2, 5)])
    # # Indices as flags in full matrix
    # connector =         sim.ArrayConnector(np.array([[0, 1, 1, 0],
    #                                                  [1, 1, 0, 1],
    #                                                  [0, 0, 1, 0]], dtype=bool)

    # db_syn = SynapseFromDB(parameter_database=params_db) # our custom synapse class

    ############################################################################
    # TO GPE

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)

    # Number of afferent axons:
    #   - 135 [boutons] / 20? [boutons/(axon*cell)] ~= 6 [axons on one cell]
    # Spatial structure:
    #   - TODO: spatial structure of STN->GPE connection
    stn_gpe_connector = sim.FixedNumberPreConnector(10)

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
        'gmax_NMDA':    0.025 / 0.1 * 1e-3, # [uS], adjusted for U1
        'tau_r_NMDA':   3.7,    # [ms] rise time
        'tau_d_NMDA':   80.0,   # [ms] decay time
    })
    

    stn_gpe_EXC = sim.Projection(pop_stn, pop_gpe, 
                                 connector=stn_gpe_connector,
                                 synapse_type=stn_gpe_syn,
                                 receptor_type='distal.AMPA+NMDA')

    #---------------------------------------------------------------------------
    # GPE -> GPE (inhibitory)

    # Number of afferent axons:
    #   - 2000 [boutons] / 264-581 [boutons/(axon*cell)] = 3.4-7.6 [axons on one cell]
    # Spatial structure:
    #   - distance-based probability and strength
    #       + 581 [boutons/(axon*cell)] for close neighbors
    #       + 264 [boutons/(axon*cell)] for far neighbors

    # TODO: calibrate again with both weight and gmax modified
    #   - (see what BBP uses in examples)
    #   - may cause different dynamics & saturation behaviour

    # Distance-calculation: wrap-around in along x-axis
    x_max = ncell_gpe * gpe_grid.dx
    gpe_gpe_space = space.Space(periodic_boundaries=((0, x_max), None, None))

    # Connect to four neighboring neurons (N_PRE=8)
    N_PRE = 6
    con_prob = 'd < ({})'.format(gpe_grid.dx * N_PRE / 2.0)
    gpe_gpe_connector = sim.DistanceDependentProbabilityConnector(
                                con_prob,
                                allow_self_connections=False)
    
    # weight 1 for distance < 50 um and exponentially decreasing from there
    weight_expression = lambda d: (d<50)*1.0 + (d>=50)*np.exp(-(d-50)/100)

    gpe_gpe_syn = GabaSynapse(**{
        'weight':       weight_expression,
        'delay':        0.5, # [ms] delay from literature
        # STP parameters
        'U1':           0.2, # baseline release probability
        'tau_rec':      400.0, # [ms] recovery from STD
        'tau_facil':    1.0, # [ms] recovery from facilitation
        # GABA-A receptor
        'gmax_GABAA':    0.1 / 0.2 * 1e-3, # [uS], adjusted for U1
        'tau_r_GABAA':   2.0, # [ms] rise time
        'tau_d_GABAA':   5.0, # [ms] decay time
        # GABA-B receptor
        'gmax_GABAB':    0.1 * 1.0 * 1e-3, # [uS], adjusted for U1
        'tau_r_GABAB':   5.0,   # [ms] rise time initial species of signaling cascade
        'tau_d_GABAB':   10.0,  # [ms] decay time initial species of signaling cascade
    })

    gpe_gpe_INH = sim.Projection(pop_gpe, pop_gpe, 
                                 connector=gpe_gpe_connector,
                                 synapse_type=gpe_gpe_syn,
                                 receptor_type='proximal.GABAA+GABAB',
                                 space=gpe_gpe_space)

    #---------------------------------------------------------------------------
    # STR -> GPE (inhibitory)

    # Number of afferent axons:
    #   - 10622 [boutons] / 123-226 [boutons/(axon*cell)] = 47-86 [axons on one cell]
    #       + If we use additive property of Poisson distribution we can divide 
    #         the number of synapses by N, with Poisson spikers firing at N*f_mean
    #       + => num_afferents ~= 60 / num_poisson_combined
    # Spatial structure:
    #   - TODO: if we re-use spike sources than connection pattern matters, else not
    str_gpe_connector = sim.FixedNumberPreConnector(66 // str_num_poisson_combined)

    str_gpe_syn = GabaSynapse(**{
        'weight':       1.0,
        'delay':        5.0, # [ms] delay from literature
        # STP parameters
        'U1':           0.2, # baseline release probability
        'tau_rec':      400.0, # [ms] recovery from STD
        'tau_facil':    1.0, # [ms] recovery from facilitation
        # GABA-A receptor
        'gmax_GABAA':    0.2 / 0.2 * 1e-3, # [uS], adjusted for U1
        'tau_r_GABAA':   2.0, # [ms] rise time
        'tau_d_GABAA':   5.0, # [ms] decay time
        # GABA-B receptor
        'gmax_GABAB':    0.0, # [uS], adjusted for U1
        'tau_r_GABAB':   5.0,   # [ms] rise time initial species of signaling cascade
        'tau_d_GABAB':   10.0,  # [ms] decay time initial species of signaling cascade
    })

    str_gpe_INH = sim.Projection(pop_str, pop_gpe,
                                 connector=str_gpe_connector,
                                 synapse_type=str_gpe_syn,
                                 receptor_type='proximal.GABAA+GABAB')


    #---------------------------------------------------------------------------
    # NOISE -> GPE (excitatory)
    # noise_syn = NativeSynapse(
    #     mechanism='GLUsyn',
    #     mechanism_parameters={
    #         'netcon:weight[0]': 1.0,
    #         'netcon:delay':     5.0, # [ms] delay from literature
    #     })

    # noise_connector = sim.FixedProbabilityConnector(0.5)
    
    # noise_gpe_EXC = sim.Projection(noise_gpe, pop_gpe, 
    #                                connector=noise_connector,
    #                                synapse_type=noise_syn,
    #                                receptor_type='proximal_dend.AMPA+NMDA')

    ############################################################################
    # TO STN

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)

    # Number of afferent axons:
    #   - 300 [boutons] * 0.33 [GPE fraction] / 6-18 [boutons/afferent] 
    #     = 100 / 12 = 8.33 [afferent axons]
    # Spatial structure:
    #   - TODO: spatial pattern of GPe->STN connection
    gpe_stn_connector = sim.FixedNumberPreConnector(8)

    gpe_stn_syn = GabaSynapse(**{
        'weight':       1.0,
        'delay':        4.0, # [ms] delay from literature
        # STP parameters
        'U1':           0.2, # baseline release probability
        'tau_rec':      17300.0, # [ms] recovery from STD
        'tau_facil':    1.0, # [ms] recovery from facilitation
        # GABA-A receptor
        'gmax_GABAA':    7.0 * 250 * 1e-3, # [uS], adjusted for U1
        'tau_r_GABAA':   2.0, # [ms] rise time
        'tau_d_GABAA':   6.0, # [ms] decay time
        # GABA-B receptor
        'gmax_GABAB':    7.0 * 5 * 1e-3, # [uS], adjusted for U1
        'tau_r_GABAB':   5.0,   # [ms] rise time initial species of signaling cascade
        'tau_d_GABAB':   10.0,  # [ms] decay time initial species of signaling cascade
    })

    # TODO: GPE -> STN projection pattern
    gpe_stn_INH = sim.Projection(
                        pop_gpe, pop_stn,
                        connector=gpe_stn_connector,
                        synapse_type=gpe_stn_syn,
                        receptor_type='proximal.GABAA+GABAB')

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)

    # Number of afferent axons:
    #   - 300 [boutons] * 0.5 [CTX fraction] / 6-18 [boutons/afferent] 
    #     = 150 / 10 = 15 [afferent axons]
    # Spatial structure:
    #   - TODO: spatial pattern of CTX->STN connection

    ctx_stn_syn = GluSynapse(**{
        'weight':       1.0,
        'delay':        5.9, # [ms] delay from literature
        # STP parameters
        'U1':           0.1, # baseline release probability
        'tau_rec':      200.0, # [ms] recovery from STD
        'tau_facil':    800.0, # [ms] recovery from facilitation
        # AMPA receptor
        'gmax_AMPA':    3.44 * 4 * 1e-3, # [uS], adjusted for U1
        'tau_r_AMPA':   1.0, # [ms] rise time
        'tau_d_AMPA':   4.0, # [ms] decay time
        # NMDA receptor
        'gmax_NMDA':    7.0 * 2.0 * 1e-3, # [uS], adjusted for U1
        'tau_r_NMDA':   3.7,    # [ms] rise time
        'tau_d_NMDA':   80.0,   # [ms] decay time
    })

    ctx_stn_connector = sim.FixedNumberPreConnector(14)

    ctx_stn_EXC = sim.Projection(
                        pop_ctx, pop_stn,
                        connector=ctx_stn_connector,
                        synapse_type=ctx_stn_syn,
                        receptor_type='distal.AMPA+NMDA')

    #---------------------------------------------------------------------------
    # STN -> STN (excitatory)

    # Number of afferent axons:
    #   - 300 [boutons] * 0.17? [STN fraction] / 14.4? [boutons/afferent] 
    #     = 50 / 14.4 ~= 3.5 [afferent axons]

    stn_stn_syn = GluSynapse(**{
        'weight':       1.0,
        'delay':        0.5, # [ms] delay from literature
        # STP parameters
        'U1':           0.1, # baseline release probability
        'tau_rec':      200.0, # [ms] recovery from STD
        'tau_facil':    800.0, # [ms] recovery from facilitation
        # AMPA receptor
        'gmax_AMPA':    3.44 * 4 * 1e-3, # [uS], adjusted for U1
        'tau_r_AMPA':   1.0, # [ms] rise time
        'tau_d_AMPA':   4.0, # [ms] decay time
        # NMDA receptor
        'gmax_NMDA':    7.0 * 2.0 * 1e-3, # [uS], adjusted for U1
        'tau_r_NMDA':   3.7,    # [ms] rise time
        'tau_d_NMDA':   80.0,   # [ms] decay time
    })

    stn_stn_connector = sim.FixedNumberPreConnector(6,
                            allow_self_connections=False)

    stn_stn_EXC = sim.Projection(
                        pop_stn, pop_stn,
                        connector=stn_stn_connector,
                        synapse_type=stn_stn_syn,
                        receptor_type='distal.AMPA+NMDA')

    #---------------------------------------------------------------------------
    # Plot connectivity matrix ('O' is connection, ' ' is no connection)
    if mpi_rank == 0:
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
    num_segments = sum((sec.nseg for sec in h.allsec()))
    num_cell = sum((1 for sec in h.allsec()))
    print("Will simulate {} cells ({} segments) for {} seconds on MPI rank {}.".format(
            num_cell, num_segments, sim_dur, mpi_rank))
    
    tstart = time.time()
    # sim.run() -> pyNN.common.control.run()
    # sim.state.run() -> pyNN.neuron.simulator._State.run()

    report_interval = 25.0
    while sim.state.t < sim_dur:
        print("Simulated {} ms. Advancing next {} ms ...".format(sim.state.t,
              report_interval))
        sim.state.run(report_interval)

    tstop = time.time()
    cputime = tstop - tstart
    
    print("Simulated {} segments for {} ms in {} ms CPU time".format(
            num_segments, sim.state.tstop, cputime))

    ############################################################################
    # PLOTTING
    ############################################################################
    if mpi_rank==0 and with_gui:
        # Only plot on one process, and if GUI available
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
    # Parse arguments passed to `python model.py [args]`
    import argparse
    parser = argparse.ArgumentParser(description='Run basal ganglia network simulation')

    parser.add_argument('-d', '--dur', nargs='?', type=float, default=500.0,
                        dest='sim_dur', help='Simulation duration')

    parser.add_argument('-n', '--ncell', nargs='?', type=int, default=30,
                        dest='ncell_per_pop', help='Number of cells per population')

    parser.add_argument('-g', '--gui', nargs='?', type=bool, default=True,
                        dest='with_gui', help='Enable graphical output')

    args = parser.parse_args() # Namespace object

    run_simple_net(**vars(args))
    # test_STN_population(**vars(args))
