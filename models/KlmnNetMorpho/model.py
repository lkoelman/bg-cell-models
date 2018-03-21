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

import numpy as np

from pyNN.utility import init_logging
import pyNN.neuron as pynn
from extensions.pynn.connection import SynapseFromDB
# TODO: test recording using trace specs
# from extensions.pynn.recording import TraceSpecRecorder
# nrn.Population._recorder_class = TraceSpecRecorder

# Load custom synapse mechanisms
import os.path
script_dir = os.path.dirname(__file__)
pynn.simulator.load_mechanisms(os.path.join('..', '..', 'mechanisms', 'synapses'))

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
    pynn.setup()
    
    seed = pynn.state.mpi_rank + pynn.state.native_rng_baseseed
    numpy_rng = np.random.RandomState(seed)

    # Parameters database
    params_db = CellConnector('Parkinsonian', numpy_rng,
                                preferred_mechanisms=['GLUsyn', 'GABAsyn'])

    # GPe cell population
    pop_stn = pynn.Population(ncell_per_pop, gillies.StnCellType())
    pop_stn.pop_id = PopID.STN
    pop_stn.initialize(v=-63.0)

    # GPe cell population
    pop_gpe = pynn.Population(ncell_per_pop, gunay.GPeCellType())
    pop_gpe.pop_id = PopID.GPE
    pop_gpe.initialize(v=-63.0)

    
    # Create connection    
    connector = pynn.AllToAllConnector()
    syn = SynapseFromDB(parameter_database=params_db) # our custom synapse class

    ## STN -> GPe
    prj_stn_gpe = pynn.Projection(pop_stn, pop_gpe, connector, syn, 
        receptor_type='distal_dend.AMPA')

    ## GPe -> STN
    prj_gpe_stn = pynn.Projection(pop_gpe, pop_stn, connector, syn, 
        receptor_type='distal_dend.AMPA')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    pynn.run(1000.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    run_simple_net(ncell_per_pop=5)