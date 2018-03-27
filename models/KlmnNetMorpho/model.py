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

from pyNN.utility import init_logging, connection_plot
import pyNN.neuron as sim
from extensions.pynn.connection import SynapseFromDB, NativeSynapse
from extensions.pynn.recording import TraceSpecRecorder
sim.Population._recorder_class = TraceSpecRecorder

# Load custom synapse mechanisms
import os.path
script_dir = os.path.dirname(__file__)
sim.simulator.load_mechanisms(os.path.join('..', '..', 'mechanisms', 'synapses'))

import models.GilliesWillshaw.gillies_pynn_model as gillies
import models.Gunay2008.gunay_pynn_model as gunay

from cellpopdata.physiotypes import Populations as PopID, ParameterSource as LitRef
from cellpopdata.cellpopdata import CellConnector


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


def run_simple_net(ncell_per_pop=5, sim_dur=500.0, export_locals=True):
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
    params_db = CellConnector('Parkinsonian', numpy_rng,
        preferred_sources=[LitRef.Chu2015, LitRef.Fan2012, LitRef.Atherton2013],
        preferred_mechanisms=['GLUsyn', 'GABAsyn'])

    ############################################################################
    # POPULATIONS
    ############################################################################

    # STN cell population
    stn_type = gillies.StnCellType(with_receptors=['GLUsyn', 'GABAsyn'])
    pop_stn = sim.Population(ncell_per_pop, stn_type, label='STN')
    pop_stn.pop_id = PopID.STN
    pop_stn.initialize(v=-63.0)

    # GPe cell population
    gpe_type = gunay.GPeCellType(with_receptors=['GLUsyn', 'GABAsyn'])
    pop_gpe = sim.Population(ncell_per_pop, gpe_type, label='GPE')
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

    db_syn = SynapseFromDB(parameter_database=params_db) # our custom synapse class

    ############################################################################
    # TO GPE 

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)
    stn_gpe_EXC = sim.Projection(pop_stn, pop_gpe, conn_allp05, db_syn, 
        receptor_type='distal_dend.AMPA+NMDA')

    #---------------------------------------------------------------------------
    # STR -> GPE (inhbitory)
    str_gpe_INH = sim.Projection(pop_str, pop_gpe, conn_allp05, db_syn, 
        receptor_type='proximal_dend.GABAA+GABAB')

    #---------------------------------------------------------------------------
    # NOISE -> GPE (excitatory)
    noise_syn = NativeSynapse(weight=1.0, delay=5.0, mechanism='GLUsyn') # default params
    noise_connector = sim.FixedProbabilityConnector(0.5)
    noise_gpe_EXC = sim.Projection(noise_gpe, pop_gpe, noise_connector, noise_syn,
                                    receptor_type='proximal_dend.AMPA+NMDA')

    ############################################################################
    # TO STN 

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)
    gpe_stn_INH = sim.Projection(pop_gpe, pop_stn, conn_allp05, db_syn, 
        receptor_type='proximal_dend.GABAA+GABAB')

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)
    ctx_stn_EXC = sim.Projection(pop_ctx, pop_stn, conn_allp05, db_syn, 
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