"""
PyNN compatible cell models for Gillies STN cell model.

@author     Lucas Koelman

@date       15/03/2018

"""

import neuron
h = neuron.h

# Load NEURON libraries, mechanisms
import os.path
script_dir = os.path.dirname(__file__)
neuron.load_mechanisms(os.path.join(script_dir, 'mechanisms'))

# Load STN cell Hoc code
h.xopen("gillies_createcell.hoc")


# from pyNN.standardmodels import StandardCellType
from pyNN.neuron.cells import NativeCellType

import extensions.pynn.ephys_models as ephys_pynn
from extensions.pynn.ephys_locations import SomaDistanceRangeLocation

# Debug messages
from common import logutils
logutils.setLogLevel('quiet', ['bluepyopt.ephys.parameters', 'bluepyopt.ephys.mechanisms'])


def define_stn_locations():
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

    _ephys_locations = define_stn_locations()

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
    receptor_types = [ # prefixes are ephys model secarray names
        'proximal_dend.Exp2Syn', 'distal_dend.Exp2Syn'
    ]


if __name__ == '__main__':
    cell_idx = h.make_stn_cell_global()
    cell_idx = int(cell_idx)
    stn_cell = h.SThcells[cell_idx]
