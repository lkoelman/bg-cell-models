"""
PyNN compatible version of Fujita, Kitano et al. (2011) GPe cell model.

@author     Lucas Koelman

@date       14/09/2018
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
h.xopen("fujita_createcell.hoc") # instantiates all functions & data structures on Hoc object
os.chdir(prev_cwd)

from bgcellmodels.extensions.pynn.ephys_models import PynnCellModelBase, EphysCellType

class GpeCellModel(PynnCellModelBase):
    """
    Model class for Mahon/Corbit MSN cell.


    EXAMPLE
    -------

    >>> from fujita_pynn_model import GpeCellModel
    >>> GpeCellModel.default_GABA_mechanism = 'MyMechanism'
    >>> cell = GpeCellModel()
    >>> nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)
    >>> icell = cell.instantiate(sim=nrnsim)
    
    """

    # Combined with celltype.receptors in EphysCellType constructor
    # to make celltype.receptor_types in format 'region.receptor'
    regions = ['proximal']

    # Must define 'default_parameters' in associated cell type
    # parameter_names = []

    # Workaround: set directly as property on the class because
    # PyNN only allows numerical parameters
    default_GABA_mechanism = 'GABAsyn'
    default_GLU_mechanism = 'GLUsyn'
    allow_synapse_reuse = False


    def instantiate(self):
        """
        Instantiate cell in simulator

        @override       ephys.models.CellModel.instantiate()

                        Since the wrapped model is a pure Hoc model completely
                        defined by its Hoc template, i.e. without ephys
                        morphology, parameters, or mechanisms definitions,
                        we have to override instantiate().
        """
        self.icell = h.FujitaGPE()
        self.icell.setparams_corbit_2016()


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


    def get_synapses(self, region, receptors, num_contacts, **kwargs):
        """
        Get synapse in subcellular region for given receptors.
        Called by Connector object to get synapse for new connection.

        @override   PynnCellModelBase.get_synapse()
        """
        syns = [self.make_new_synapse(receptors, self.icell.soma[0](0.5), **kwargs) for i in xrange(num_contacts)]
        synmap_key = tuple(sorted(receptors))
        self._synapses['proximal'].setdefault(synmap_key, []).extend(syns)
        return syns



class GpeProtoType(EphysCellType):
    """
    Encapsulates an MSN model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.
    """

    # The encapsualted model available as class attribute 'model'
    model = GpeCellModel

    # NOTE: default_parameters is used to make 'schema' for checking & converting datatypes
    default_parameters = {}
    # extra_parameters = {}
    default_initial_values = {'v': -65.0}
    # recordable = ['spikes', 'v']

    # Combined with self.model.regions by EphysCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(GpeProtoType, self).can_record(variable)


class GpeArkyType(EphysCellType):
    """
    Encapsulates an MSN model described as a BluePyOpt Ephys model 
    for interoperability with PyNN.
    """

    # The encapsualted model available as class attribute 'model'
    model = GpeCellModel

    # NOTE: default_parameters is used to make 'schema' for checking & converting datatypes
    default_parameters = {}
    # extra_parameters = {}
    default_initial_values = {'v': -65.0}
    # TODO: decrease NaP for arky type
    # recordable = ['spikes', 'v']

    # Combined with self.model.regions by EphysCellType constructor
    receptor_types = ['AMPA', 'NMDA', 'AMPA+NMDA',
                      'GABAA', 'GABAB', 'GABAA+GABAB']


    def can_record(self, variable):
        """
        Override or it uses pynn.neuron.record.recordable_pattern.match(variable)
        """
        return super(GpeArkyType, self).can_record(variable)


def test_gpe_population(export_locals=True):
    """
    Test creation of PyNN population of MSN cells.
    """
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # STN cell population
    cell_type = GpeProtoType()
    p1 = nrn.Population(5, cell_type)

    # Stimulation electrode
    # Fujita et al. (2011): "Change the IClamp delay to 500, duration to
    # 1000 and current to either -.003, 0, .003, .006 nA to correspond to
    # -1, 0, 1, 2 uA/cm2."
    current_source = nrn.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                                           amplitudes=[0.4, 0.6, -0.2, 0.2])
    p1.inject(current_source)

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(500.0)

    if export_locals:
        print("Adding to global namespace: {}".format(locals().keys()))
        globals().update(locals())


if __name__ == '__main__':
    test_msn_population()