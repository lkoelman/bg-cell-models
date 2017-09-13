"""
Extensions to BluePyOpt classes for optimizing Gillies STN neuron

@author Lucas Koelman

@date   13/09/2017


ARCHITECTURE

Different approaches to creating a cell model

- create own ephys.Model or ephys.models.CellModel

    + see CellModel class
        * https://github.com/BlueBrain/BluePyOpt/blob/master/bluepyopt/ephys/models.py
    
    + see external use cases
        * https://github.com/BlueBrain/BluePyOpt/wiki
        * particularly the example at https://github.com/apdavison/BluePyOpt/blob/pynn-models/bluepyopt/ephys_pyNN/models.py

    + see example of complex model at https://github.com/BlueBrain/BluePyOpt/tree/master/examples/l5pc

    + see dummy cell model at https://github.com/BlueBrain/BluePyOpt/blob/master/bluepyopt/tests/test_ephys/testmodels/dummycells.py

"""

import os

import bluepyopt.ephys as ephys

import gillies_model
from reducemodel import redutils, reduce_cell


import logging
logger = logging.getLogger(__name__)


class StnReducedModel(ephys.models.Model):
    '''
    Wraps the Gillies & Willshaw model so it can be used by BluePyOpt

    Based on dummy cell template at https://github.com/BlueBrain/BluePyOpt/blob/master/bluepyopt/tests/test_ephys/testmodels/dummycells.py

    '''

    def __init__(self, name=None, fold_method=None, num_passes=1):
        """Constructor"""

        super(StnReducedModel, self).__init__(name)
        self.persistent = []
        self.icell = None

        self._fold_method = fold_method
        self._num_passes = num_passes

        self._reduction = None


    def freeze(self, param_values):
        """Freeze model"""
        pass

    def unfreeze(self, param_names):
        """Freeze model"""
        pass

    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator
        """

        class Cell(object):
            """
            Empty cell class, substitute for Hoc template.
            """

            def __init__(self):
                """Constructor"""
                # list(Section) for each region
                self.soma = None
                self.dendrite = None
                

                # SectionList for each region
                self.somatic = None
                self.dendritic = None

        # Make sure original STN cell is built
        if self._reduction is None:
            
            # Do initial model reduction
            if self._fold_method == 'marasco':
                reduction = reduce_cell.gillies_marasco_reduction()
                reduction.reduce_model(num_passes=self._num_passes)

            elif self._fold_method == 'stratford':
                reduction = reduce_cell.gillies_stratford_reduction()
                reduction.reduce_model(num_passes=self._num_passes)

            self._reduction = reduction

            # Save initial cell parameters
            for ref in reduction.all_sec_refs:
                redutils.store_seg_props(ref, gillies_model.gillies_gdict, attr_name='initial_params')

        else:
            # Reset cell parameters
            for ref in self._reduction.all_sec_refs:
                redutils.set_range_props(ref, ref.initial_params)

        # Create a cell
        self.icell = Cell()

        # TODO: control instantiation across optimization procedure
        #   - check if instantiate called for each individual
        #       - see ephys.protocols.SweepProtocol._run_func
        #       - operations: model.freeze(params) > model.instantiate() proto.instantiate() > sim.run () > model.unfreeze()
        #           - model.freeze()
        #               - calls param.freeze() on each param
        #               - this calls param.value.setter
        #               - how this value is used: see param.instantiate()
        #           - model.instantiate()
        #               - should call mechanism.instantiate(), param.instantiate()
        #   - save initial param values with redutils.store_seg_props
        #   - let instantiate restore initial values

        #TODO: see ephys.CellModel.instantiate() : copy relevant calls/functionality

        # Copy sections to instantiated cell object
        self.icell.soma = [ref.sec for ref in self._reduction._soma_refs]
        self.icell.dendrite = [ref.sec for ref in self._reduction._dend_refs]

        # Make SectionLists (for identification of regions)
        ## Somatic region
        self.icell.somatic = sim.neuron.h.SectionList()
        for sec in self.soma:
            self.icell.somatic.append(sec=sec)

        ## Dendritic region
        self.icell.dendritic = sim.neuron.h.SectionList()
        for sec in self.icell.dendrite:
            self.icell.dendritic.append(sec=sec)

        self.persistent.append(self.icell)
        self.persistent.append(self.icell.soma[0])

        return self.icell

    def destroy(self, sim=None):
        """Destroy cell from simulator"""

        self.persistent = []



class StnProtoModel(ephys.models.HocCellModel):
    '''
    Wraps the Gillies & Willshaw Hoc Template so it can be used by BluePyOpt

    TODO: use HocCellModel
        - this requires changes to the Gillies cell template.
        - use predefined morphology file (export Gillies model as SWC)
        - see https://github.com/BlueBrain/BluePyOpt/blob/master/bluepyopt/ephys/models.py
        - see complex cell model example

    '''

    def __init__(self, name, hoc_path=None, hoc_string=None):
        """
        Constructor

        Args: see HocCellModel documentation
        """
        super(StnProtoModel, self).__init__(name,
                                           morph=None,
                                           mechs=[],
                                           params=[])

        if hoc_path is not None and hoc_string is not None:
            raise TypeError('HocCellModel: cant specify both hoc_string '
                            'and hoc_path argument')
        if hoc_path is not None:
            with open(hoc_path) as hoc_file:
                self.hoc_string = hoc_file.read()
        else:
            self.hoc_string = hoc_string

        # TODO: hardcoded morphology path (need to add 3d info that is compatible with layout)
        raise NotImplementedError("STN template-based model not yet implemented")
        morphology_path = '...'

        self.morphology = ephys.models.HocMorphology(morphology_path)
        self.cell = None
        self.icell = None


    def instantiate(self, sim=None):
        sim.neuron.h.load_file('stdrun.hoc')
        template_name = self.load_hoc_template(sim, self.hoc_string)

        morph_path = self.morphology.morphology_path
        assert os.path.exists(morph_path), \
            'Morphology path does not exist: %s' % morph_path
        
        if os.path.isdir(morph_path):
            # will use the built in morphology name, if the init() only
            # gets one parameter
            self.cell = getattr(sim.neuron.h, template_name)(morph_path)
        
        else:
            morph_dir = os.path.dirname(morph_path)
            morph_name = os.path.basename(morph_path)
            self.cell = getattr(sim.neuron.h, template_name)(morph_dir,
                                                             morph_name)
        self.icell = self.cell.CellRef


    def destroy(self, sim=None):
        self.cell = None
        self.icell = None


    def check_nonfrozen_params(self, param_names):
        pass


    def __str__(self):
        """Return string representation"""
        return (
            '%s: %s of %s(%s)' %
            (self.__class__,
             self.name,
             self.get_template_name(self.hoc_string),
             self.morphology.morphology_path,))