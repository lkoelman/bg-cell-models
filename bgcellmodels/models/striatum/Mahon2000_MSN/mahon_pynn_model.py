"""
PyNN compatible version of Corbit/Mahon Striatal MSN model.

@author     Lucas Koelman

@date       14/08/2018
"""

import neuron
h = neuron.h

# Load NEURON libraries, mechanisms
import os, os.path
script_dir = os.path.dirname(__file__)
neuron.load_mechanisms(os.path.join(script_dir, 'mechanisms'))

# Load Hoc functions for cell model
prev_cwd = os.getcwd()
os.chdir(script_dir)
h.xopen("mahon_createcell.hoc") # instantiates all functions & data structures on Hoc object
os.chdir(prev_cwd)

import bgcellmodels.extensions.pynn.ephys_models as ephys_pynn

class MsnCellModel(ephys_pynn.EphysModelWrapper):
    """
    Model class for Mahon/Corbit MSN cell.


    EXAMPLE
    -------

    >>> cell = MsnCellModel()
    >>> nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    >>> icell = cell.instantiate(sim=nrnsim)
    
    """

    regions = ['proximal']

    # Must define 'default_parameters' in associated cell type
    # parameter_names = []

    # Workaround: set directly as property on the class because
    # PyNN only allows numerical parameters
    GABA_synapse_mechanism = 'GABAsyn'
    GLU_synapse_mechanism = 'GLUsyn'


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @override       ephys.models.CellModel.instantiate()
                        
                        Since the wrapped model is a pure Hoc model completely
                        defined by its Hoc template, i.e. without ephys
                        morphology, parameters, or mechanisms definitions,
                        we have to override instantiate().
        """
        self.icell = h.MahonMSN()
        for seg in self.icell.soma[0]:
            seg.el_Leakm = -90 # Corbit (2016) changes -75 to -90


    def memb_init(self):
        """
        Initialization function required by PyNN.

        @override     EphysModelWrapper.memb_init()
        """
        for seg in self.icell.soma:
            seg.v = self.v_init


    def get_threshold(self):
        """
        Get spike threshold for soma membrane potential (used for NetCon)
        """
        return 0.0


    def _init_synapses(self):
        """
        Initialize synapses on this neuron.

        @override   EphysModelWrapper._init_synapses()
        """
        pass # raise NotImplementedError() # TODO: add synapses on neuron model



class MsnCellType(ephys_pynn.EphysCellType):
    """
    Encapsulates an MSN model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.
    """

    # The encapsualted model available as class attribute 'model'
    model = MsnCellModel

    # NOTE: default_parameters is used to make 'schema' for checking & converting datatypes
    default_parameters = {}
    # extra_parameters = {}
    default_initial_values = {'v': -77.4}
    # recordable = ['spikes', 'v']

    # Combined with self.model.regions by EphysCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB',
                      'Exp2Syn']


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(MsnCellType, self).can_record(variable)


def test_msn_population(export_locals=True):
    """
    Test creation of PyNN population of MSN cells.
    """
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # STN cell population
    cell_type = MsnCellType()
    p1 = nrn.Population(5, cell_type)

    p1.initialize(v=-63.0)

    # Stimulation electrode
    current_source = nrn.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                           amplitudes=[0.4, 0.6, -0.2, 0.2])
    p1.inject(current_source)

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    test_msn_population()