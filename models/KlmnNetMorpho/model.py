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

from pyNN.utility import init_logging
import pyNN.neuron as nrn

import models.GilliesWillshaw.gillies_pynn_model as gillies
import models.Gunay2008.gunay_pynn_model as gunay


def run_simple_net(ncell_per_pop=10, export_locals=True):
    """
    Run a simple network consisting of an STN and GPe cell population
    that are reciprocally connected.
    """

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # GPe cell population
    pop_stn = nrn.Population(ncell_per_pop, gillies.StnCellType())
    pop_stn.initialize(v=-63.0)

    # GPe cell population
    pop_gpe = nrn.Population(ncell_per_pop, gunay.GPeCellType())
    pop_gpe.initialize(v=-63.0)

    
    # Create connection    
    connector = nrn.AllToAllConnector()
    syn = nrn.StaticSynapse(weight=0.1, delay=2.0)

    ## STN -> GPe
    prj_stn_gpe = nrn.Projection(pop_stn, pop_gpe, connector, syn, 
        receptor_type='distal_dend.Exp2Syn')

    ## GPe -> STN
    prj_gpe_stn = nrn.Projection(pop_gpe, pop_stn, connector, syn, 
        receptor_type='distal_dend.Exp2Syn')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    run_simple_net(ncell_per_pop=10)