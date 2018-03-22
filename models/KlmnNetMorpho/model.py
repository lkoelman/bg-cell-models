"""
Basal Ganglia network model consisting of morphologically detailed
cell models for the major cell types.

@author     Lucas Koelman

@date       20/03/2017

@see        PyNN manual for building networks:
                http://neuralensemble.org/docs/PyNN/building_networks.html
            PyNN examples of networks:
                https://github.com/NeuralEnsemble/PyNN/tree/master/examples
"""
import time
import numpy as np

from pyNN.utility import init_logging
import pyNN.neuron as sim
from extensions.pynn.connection import SynapseFromDB
from extensions.pynn.recording import TraceSpecRecorder
sim.Population._recorder_class = TraceSpecRecorder

# Load custom synapse mechanisms
import os.path
script_dir = os.path.dirname(__file__)
sim.simulator.load_mechanisms(os.path.join('..', '..', 'mechanisms', 'synapses'))

import models.GilliesWillshaw.gillies_pynn_model as gillies
import models.Gunay2008.gunay_pynn_model as gunay

from cellpopdata.physiotypes import Populations as PopID
from cellpopdata.cellpopdata import CellConnector


def run_simple_net(ncell_per_pop=5, export_locals=True):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.
    """

    init_logging(logfile=None, debug=True)
    mpi_rank = sim.setup(use_cvode=False)
    
    seed = sim.state.mpi_rank + sim.state.native_rng_baseseed
    numpy_rng = np.random.RandomState(seed)

    # Parameters database
    params_db = CellConnector('Parkinsonian', numpy_rng,
                                preferred_mechanisms=['GLUsyn', 'GABAsyn'])

    ############################################################################
    # POPULATIONS
    ############################################################################

    # GPe cell population
    pop_stn = sim.Population(ncell_per_pop, gillies.StnCellType())
    pop_stn.pop_id = PopID.STN
    pop_stn.initialize(v=-63.0)

    # GPe cell population
    pop_gpe = sim.Population(ncell_per_pop, gunay.GPeCellType())
    pop_gpe.pop_id = PopID.GPE
    pop_gpe.initialize(v=-63.0)

    # CTX spike sources
    pop_ctx = sim.Population(ncell_per_pop, sim.SpikeSourcePoisson(rate=50.0))
    pop_ctx.pop_id = PopID.CTX

    all_pops = {pop.pop_id : pop for pop in [pop_gpe, pop_stn, pop_ctx]}

    ############################################################################
    # CONNECTIONS
    ############################################################################
    
    # Create connection    
    conn_all2all = sim.AllToAllConnector()
    conn_allp05 = sim.FixedProbabilityConnector(0.5)

    syn = SynapseFromDB(parameter_database=params_db) # our custom synapse class

    ############################################################################
    # XXX -> GPE 

    #---------------------------------------------------------------------------
    # STN -> GPE (excitatory)
    stn_gpe_EXC = sim.Projection(pop_stn, pop_gpe, conn_all2all, syn, 
        receptor_type='distal_dend.AMPA')

    # stn_gpe_NMDA = sim.Projection(pop_stn, pop_gpe, all_to_all, syn, 
    #     receptor_type='distal_dend.NMDA')

    #---------------------------------------------------------------------------
    # STR -> GPE (inhbitory)

    ############################################################################
    # XXX -> STN 

    #---------------------------------------------------------------------------
    # GPe -> STN (inhibitory)
    gpe_stn_INH = sim.Projection(pop_gpe, pop_stn, conn_all2all, syn, 
        receptor_type='proximal_dend.GABAA')

    #---------------------------------------------------------------------------
    # CTX -> STN (excitatory)
    ctx_stn_EXC = sim.Projection(pop_ctx, pop_stn, conn_all2all, syn, 
        receptor_type='distal_dend.AMPA+NMDA')

    ############################################################################
    # RECORDING
    ############################################################################

    traces_allpops = {
        'Vm':       {'sec':'soma[0]', 'loc':0.5, 'var':'v'},
    }
    for pop in [pop_gpe, pop_stn]:
        pop.record(traces_allpops.items(), sampling_interval=.05)
    for pop in all_pops.values():
        pop.record(['spikes'], sampling_interval=.05)


    print("CVode state: {}".format(sim.state.cvode.active()))
    tstart = time.time()
    sim.run(1000.0)
    tstop = time.time()
    cputime = tstop - tstart
    print("Simulated {} ms neuron time in {} ms CPU time".format(
            sim.state.tstop, cputime))

    ############################################################################
    # PLOTTING
    ############################################################################
    import matplotlib.pyplot as plt

    # Plot spikes
    for pop in all_pops.values():
        data_block = pop.get_data() # Neo Block object
        segment = data_block.segments[0] # segment = all data with common time basis
        plt.figure()
        for spiketrain in segment.spiketrains:
            y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
            plt.plot(spiketrain, y, '.')
            plt.ylabel(segment.name)
            plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.show(block=False)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    run_simple_net(ncell_per_pop=5)