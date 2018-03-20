"""
PyNN compatible cell models for Gillies STN cell model.

@author     Lucas Koelman

@date       15/03/2018

"""

import neuron
h = neuron.h

# Load NEURON libraries, mechanisms
import os, os.path
script_dir = os.path.dirname(__file__)
neuron.load_mechanisms(os.path.join(script_dir, 'mechanisms'))

# Load STN cell Hoc code
prev_cwd = os.getcwd()
os.chdir(script_dir)
# os.environ['HOC_LIBRARY_PATH'] = os.environ.get('HOC_LIBRARY_PATH', '') + ':' + script_dir
h.xopen("gillies_createcell.hoc")
os.chdir(prev_cwd)

# from pyNN.standardmodels import StandardCellType
from pyNN.neuron.cells import NativeCellType

import extensions.pynn.ephys_models as ephys_pynn
from extensions.pynn.ephys_locations import SomaDistanceRangeLocation

# Debug messages
from common import logutils
logutils.setLogLevel('quiet', ['bluepyopt.ephys.parameters', 'bluepyopt.ephys.mechanisms'])


def define_locations():
    """
    Define locations / regions on the cell that will function as the target
    of synaptic connections.

    @return     list(SomaDistanceRangeLocation)
                List of location / region definitions.
    """

    proximal_dend = SomaDistanceRangeLocation(
        name='proximal_dend',
        seclist_name='basal',
        min_distance=5.0,
        max_distance=100.0,
        syn_mech_names=['Exp2Syn'])

    distal_dend = SomaDistanceRangeLocation(
        name='distal_dend',
        seclist_name='basal',
        min_distance=100.0,
        max_distance=1000.0,
        syn_mech_names=['Exp2Syn'])

    return [proximal_dend, distal_dend]


class StnCellModel(ephys_pynn.EphysModelWrapper):
    # TODO: adapt model wrapper for cells without morphology
    #   - A: maintain the class, just do checks to see which of _ephys_xxx
    #     is present in class definition, and adapt instantiation procedure.
    #       => makes most sense, since we need to subclass CellModel anyway for
    #          Hoc models without morphology etc., then we might as well override
    #          instantiate etc.
    #
    #   - B: make a new wrapper class called HocModelWrapper that inherits from
    #     ephys.CellModel (see bpop_cellmodels) + does the pyNN-related modifications.
    #
    #   - C: create two subclasses of intermediate base classEphysModelWrapper: 
    #        one with and one without morphology


    _ephys_locations = define_locations()


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @override       ephys.models.CellModel.instantiate()
                        
                        Since the wrapped model is a pure Hoc model completely
                        defined by its Hoc template, i.e. without ephys
                        morphology, parameters, or mechanisms definitions,
                        we have to override instantiate().
        """

        cell_idx = h.make_stn_cell_global()
        cell_idx = int(cell_idx)
        self.icell = h.SThcells[cell_idx]


class StnCellType(NativeCellType):
    """
    Encapsulates an STN model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.
    """

    # The encapsualted model available as class attribute 'model'
    model = StnCellModel

    default_parameters = {}
    # extra_parameters = {}
    # default_initial_values = {'v': -65.0}

    # Synapse receptor types per region
    receptor_types = [
        "{}.{}".format(loc.name, syn_mech) for loc in StnCellModel._ephys_locations 
            for syn_mech in loc.syn_mech_names
    ]


def test_stn_cells_multiple(export_locals=True):
    """
    Test creation of multiple instances of the Gillies STN model.
    """
    stn_cells = []
    for i in range(5):
        cell_idx = h.make_stn_cell_global()
        cell_idx = int(cell_idx)
        stn_cells.append(h.SThcells[cell_idx])

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


def test_stn_pynn_population(export_locals=True):
    """
    Test creation of PyNN population of STN cells.
    """
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # STN cell population
    cell_type = StnCellType()
    p1 = nrn.Population(5, cell_type)

    p1.initialize(v=-63.0)

    # Stimulation electrode
    current_source = nrn.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                           amplitudes=[0.4, 0.6, -0.2, 0.2])
    p1.inject(current_source)

    # Stimulation spike source
    p2 = nrn.Population(1, nrn.SpikeSourcePoisson(rate=100.0))
    connector = nrn.AllToAllConnector()
    syn = nrn.StaticSynapse(weight=0.1, delay=2.0)
    prj_alpha = nrn.Projection(p2, p1, connector, syn, 
        receptor_type='distal_dend.Exp2Syn')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    # test_stn_cells_multiple()
    test_stn_pynn_population()
