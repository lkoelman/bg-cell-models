"""
Morphological axon models for PyNN

@author     Lucas Koelman

@date       07/05/2019

"""

import logging

# Third party modules
import numpy as np
from neuron import h
from pyNN.neuron.cells import NativeCellType
from pyNN.parameters import ArrayParameter

# Our custom modules
from bgcellmodels.extensions.pynn import cell_base
from bgcellmodels import emfield # Hoc code
from bgcellmodels.extensions.pynn import cell_base
from bgcellmodels.models.axon.foust2011 import AxonFoust2011

logger = logging.getLogger('ext.pynn.cell_base')


class AxonalRelay(object):
    """
    Axon that relays incoming spikes to its terminal section as a propagating
    action potential.
    """

    def __init__(self, **kwargs):
        """
        As opposed to the original CellModel class,
        this class instantiates the cell in its __init__ method.

        ARGUMENTS
        ---------

        @param      **kwargs : dict(str, object)
                    Parameter name, value pairs
        """
        # Save parameters as attributes
        for param_name, param_val in kwargs.items():
            if isinstance(param_val, ArrayParameter):
                param_val = param_val.value
            setattr(self, param_name, param_val)

        # Build axon
        axon_builder = self.axon_class(
            without_extracellular=not self.with_extracellular)


        self.icell = axon_builder.build_along_streamline(
                        self.streamline_coordinates_mm,
                        termination_method=self.termination_method,
                        interp_method='arclength',
                        tolerance_mm=1e-4)

        initial_sec = self.icell.ordered[0]
        terminal_sec = self.icell.ordered[-1]


        # Change source for NetCons (see pyNN.neuron.simulator code)
        self.source_section = terminal_sec
        self.source = terminal_sec(0.5)._ref_v


        # Attributes required for recording (see pyNN.neuron.recording.Recorder._record())
        self.rec = h.NetCon(self.source, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self.source_section)
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = False

        # Create targets for NetCons
        # (property names must correspond to entries in cell_type.receptor_types)
        self.excitatory = h.Exp2Syn(initial_sec(0.5))
        # Synapse properties together with weight of 1 cause good following up to 200 Hz
        self.excitatory.tau1 = 0.1
        self.excitatory.tau2 = 0.2
        self.excitatory.e = 0.0

        # Initialize extracellular stim & rec
        if self.with_extracellular:
            self._init_emfield()


    def memb_init(self):
        """
        Set initial values for all variables in this cell.

        @override   memb_init() required by PyNN interface for cell models.
        """
        for sec in self.icell.all:
            for seg in sec:
                seg.v = self.v_init # set using pop.init(v=v_init) or default_initial_values


    def get_threshold(self):
        """
        Get spike threshold for self.source variable (usually points to membrane
        potential). This threshold is used when creating NetCon connections.

        @override   get_threshold() required by pyNN interface for cell models.

        @return     threshold : float
        """
        return -10.0


    # Bind interface method
    resolve_section = cell_base.irec_resolve_section


    def _init_emfield(self):
        """
        Set up extracelullar stimulation and recording.

        @pre    mechanism 'extracullular' must be inserted and parameters set
                in all compartments that should contribute to the LFP and are
                targets for stimulation
        """
        # Insert mechanism that mediates between extracellular variables and
        # recording & stimulation routines.
        for sec in self.icell.all:
            if h.ismembrane('extracellular', sec=sec):
                sec.insert('xtra')

        # Calculate coordinates of each compartment's (segment) center
        h.xtra_segment_coords_from3d(self.icell.all)
        h.xtra_setpointers(self.icell.all)

        # Set transfer impedance between electrode and compartment centers
        x_elec, y_elec, z_elec = self.electrode_coordinates_um
        h.xtra_set_impedances_pointsource(
            self.icell.all, self.rho_extracellular_ohm_cm, x_elec, y_elec, z_elec)

        # Set up LFP calculation
        if logger.level <= logging.WARNING:
            h.XTRA_VERBOSITY = 1
        self.lfp_summator = h.xtra_sum(self.icell.ordered[0](0.5))
        self.lfp_tracker = h.ImembTracker(self.lfp_summator, self.icell.all, "xtra")


    def get_all_sections(self):
        """
        Get all neuron.Section objects that make up this cell.

        @return     neuron.SectionList containing all sections
        """
        return self.icell.all


class AxonRelayType(NativeCellType):
    """
    PyNN cell type for use with AxonalRelay cell model.
    """
    model = AxonalRelay

    # Queried by Population.find_units() for recordings
    units = cell_base.UnitFetcherPlaceHolder()

    # Properties of cell model containing NetCon targets
    receptor_types = ['excitatory']

    default_parameters = {
        # 3D specification
        'transform': ArrayParameter([]),
        'streamline_coordinates_mm': ArrayParameter([]),
        'termination_method': np.array('terminal_sequence'),
        # Extracellular stim & rec
        'with_extracellular': False,
        'electrode_coordinates_um' : ArrayParameter([]),
        'rho_extracellular_ohm_cm' : 0.03, 
        
    }

    # NOTE: extra_parameters supports non-numpy types. 
    extra_parameters = {
        'axon_class': AxonFoust2011,
    }

    default_initial_values = {'v': -68.0}

    def __init__(self, **parameters):
        """
        Trick for allowing extra parameters as kwargs.
        """
        self.extra_parameters = {
            k: parameters.pop(k, v) for k,v in AxonRelayType.extra_parameters.items()
        }
        NativeCellType.__init__(self, **parameters)


################################################################################
# Testing
################################################################################


def test_simulate_population(export_locals=False):
    """
    Test PyNN model creation, running, and recording.

    @see    Based on test in:
            https://github.com/NeuralEnsemble/PyNN/blob/master/test/system/test_neuron.py
    """
    from pyNN.utility import init_logging
    import pyNN.neuron as nrn
    import numpy as np

    init_logging(logfile=None, debug=True)
    nrn.setup()

    # Artificial axon trajectory
    axon_coords = np.concatenate([np.arange(20).reshape((-1,1))]*3, axis=1)
    elec_coords = np.array([1e6, 1e6, 1e6])

    # GPe cell population
    num_cell = 5
    axon_params = {
        'transform': ArrayParameter(np.eye(4)), # [np.eye(4)] * num_cell,
        'streamline_coordinates_mm': ArrayParameter(axon_coords), # [axon_coords] * num_cell,
        'axon_class': AxonFoust2011,
        'with_extracellular': True,
        'electrode_coordinates_um': ArrayParameter(elec_coords), # [elec_coords] * num_cell,
    }
    cell_type = AxonRelayType(**axon_params)
    pop_dest = nrn.Population(5, cell_type)
    pop_dest.initialize(v=-68.0)

    # Spike source population
    pop_src = nrn.Population(5, nrn.SpikeSourcePoisson(rate=100.0))
    
    # Connect populations
    connector = nrn.OneToOneConnector()
    syn = nrn.StaticSynapse(weight=1.0, delay=2.0)
    prj_alpha = nrn.Projection(pop_src, pop_dest,
                               connector, syn, 
                               receptor_type='excitatory')

    # Recording
    # p1.record(['apical(1.0).v', 'soma(0.5).ina'])
    nrn.run(250.0)

    if export_locals:
        globals().update(locals())


if __name__ == '__main__':
    test_simulate_population(export_locals=True)