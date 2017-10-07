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
import collections

import bluepyopt.ephys as ephys

import gillies_model
from reducemodel import redutils, reduce_cell


import logging
logger = logging.getLogger('bpop_ext')


class Cell(object):
    """
    Insntantiated cell class, substitute for Hoc template.
    """

    def __init__(self):
        """Constructor"""
        # list(Section) for each region
        self.soma = None
        self.dendrite = None

        # SectionList for each region
        self.somatic = None
        self.dendritic = None

        # list(SectionRef) for each region
        self._soma_refs = None
        self._dend_refs = None
        self._all_refs = None


class StnBaseModel(ephys.models.Model):
    '''
    Wraps the Gillies & Willshaw model so it can be used by BluePyOpt.

    This is version of ephys.models.CellModel without a morphology file,
    that instantiates the STN cell using the Gillies STN module.

    Based on:
        - ephys.models.CellModel
        - dummy cell template at https://github.com/BlueBrain/BluePyOpt/blob/master/bluepyopt/tests/test_ephys/testmodels/dummycells.py

    '''

    def __init__(self, 
            name=None,
            mechs=None,
            params=None,
            gid=0):
        """
        Constructor

        Args:
            mechs (list of Mechanisms):
                Mechanisms associated with the cell
            
            params (list of Parameters):
                Parameters of the cell model
        """

        super(StnBaseModel, self).__init__(name)

        # BluePyOpt variables
        self.mechanisms = [] if (mechs is None) else mechs
        self.params = collections.OrderedDict()
        
        if params is not None:
            for param in params:
                self.params[param.name] = param

        self.param_values = None
        self.gid = gid

        self.seclist_names = ['all', 'somatic', 'dendritic'] # SectionList variables defining regions
        self.secarray_names = ['soma', 'dendrite']

        self.icell = None

    def set_params(self, params):
        """
        Convenience function to be able to set parameters after making the model
        """
        self.params = collections.OrderedDict()
        if params is not None:
            for param in params:
                self.params[param.name] = param


    def set_mechs(self, mechs):
        """
        Convenience function to be able to set mechanisms after making the model
        """
        self.mechanisms = mechs 


    def params_by_names(self, param_names):
        """
        Get parameter objects by name
        """

        return [self.params[param_name] for param_name in param_names]


    def freeze(self, param_dict):
        """
        Set params
        """

        for param_name, param_value in param_dict.items():
            self.params[param_name].freeze(param_value)


    def unfreeze(self, param_names):
        """
        Unset params
        """

        for param_name in param_names:
            self.params[param_name].unfreeze()


    def check_nonfrozen_params(self, param_names):
        """
        Check if all nonfrozen params are set
        """

        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator

        @note   this method should only be called after the subclass instantiate function

        @post   all mechanisms and parameters are instantiated

        @post   stored all all Sections referred to by icell's SectionRef objects
                in following attributes:

                    soma        list(Section)
                    dendrite    list(Section)

                    somatic     Hoc.SectionList()
                    dendritic   Hoc.SectionList()
                    all         Hoc.SectionList()
        """

        # instantiation across optimization procedure:
        #       - see ephys.protocols.SweepProtocol._run_func
        #       - operations: model.freeze(params) > model.instantiate() > proto.instantiate() > sim.run () > model.unfreeze()
        #           - model.freeze()
        #               - calls param.freeze() on each param
        #               - this calls param.value.setter
        #               - how this value is used: see param.instantiate()
        #           - model.instantiate()
        #               - should call mechanism.instantiate(), param.instantiate()

        # Copy sections to instantiated cell object
        self.icell.soma = [ref.sec for ref in self.icell._soma_refs]
        self.icell.dendrite = [ref.sec for ref in self.icell._dend_refs]

        # Make SectionLists (for identification of regions)
        self.icell.all = sim.neuron.h.SectionList()
        
        ## Somatic region
        self.icell.somatic = sim.neuron.h.SectionList()
        for sec in self.icell.soma:
            self.icell.somatic.append(sec=sec)
            self.icell.all.append(sec=sec)

        ## Dendritic region
        self.icell.dendritic = sim.neuron.h.SectionList()
        for sec in self.icell.dendrite:
            self.icell.dendritic.append(sec=sec)
            self.icell.all.append(sec=sec)

        # Instantiate mechanisms and parameters
        for mechanism in self.mechanisms:
            mechanism.instantiate(sim=sim, icell=self.icell)
        
        for param in self.params.values():
            param.instantiate(sim=sim, icell=self.icell)

        return self.icell


    def destroy(self, sim=None):
        """Destroy cell from simulator"""

        self.icell = None

        for mechanism in self.mechanisms:
            mechanism.destroy(sim=sim)
        
        for param in self.params.values():
            param.destroy(sim=sim)


class StnFullModel(StnBaseModel):
    
    def __init__(self, 
            name=None,
            mechs=None,
            params=None,
            gid=0):
        """
        Constructor

        Args: see StnBaseModel
        """

        super(StnFullModel, self).__init__(name, mechs, params, gid)
        self._persistent_refs = None


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator
        """

        # Create a cell
        self.icell = Cell()
        self.icell.gid = self.gid
        
        # Make sure original STN cell is built
        h = sim.neuron.h
        stn_exists = hasattr(h, 'SThcell')
        if (not stn_exists) or (self._persistent_refs is None):
            
            # Create Gillies & Willshaw STN model or get refs to existing
            soma, dendL, dendR = gillies_model.get_stn_refs()

            # If existed: reset variables
            if stn_exists:
                h.set_gbar_stn() # see createcell.hoc

            self.icell._soma_refs = [soma]
            self.icell._dend_refs = dendL + dendR
            self.icell._all_refs = self.icell._soma_refs + self.icell._dend_refs

            # save SectionRef across model instantiation (icell will be destroyed)
            self._persistent_refs = self.icell._all_refs

            # Save initial cell parameters
            for ref in self.icell._all_refs:
                redutils.store_seg_props(ref, gillies_model.gillies_gdict, attr_name='initial_params')

        else:
            # Restore persistent SectionRef
            self.icell._all_refs = self._persistent_refs
            self.icell._soma_refs = gillies_model.get_soma_refs(self._persistent_refs)
            self.icell._dend_refs = [ref for ref in self._persistent_refs if ref not in self.icell._soma_refs]

            # Reset cell parameters
            for ref in self.icell._all_refs:
                # Restore parameter dict stored on SectionRef
                redutils.set_range_props(ref, ref.initial_params)

        # Set physiological conditions
        # h.celsius = 30 # will be overriden by possible global parameter in super.instantiate
        # h.v_init = -60
        # h.set_aCSF(4)

        # Call base class method
        return super(StnFullModel, self).instantiate(sim)


class StnReducedModel(StnBaseModel):
    
    def __init__(self, 
            name=None,
            fold_method=None,
            num_passes=1,
            mechs=None,
            params=None,
            gid=0):
        """
        Constructor

        Args: see StnBaseModel
        """

        super(StnReducedModel, self).__init__(name, mechs, params, gid)
        self._persistent_refs = None

        # Reduction variables
        self._fold_method = fold_method
        self._num_passes = num_passes

        self._reduction = None


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator
        """

        # Create a cell
        self.icell = Cell()
        self.icell.gid = self.gid
        

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

            self.icell._soma_refs = reduction._soma_refs
            self.icell._dend_refs = reduction._dend_refs
            self.icell._all_refs = self.icell._soma_refs + self.icell._dend_refs

            # save SectionRef across model instantiation (icell will be destroyed)
            self._persistent_refs = self.icell._all_refs

            # Save initial cell parameters
            for ref in self.icell._all_refs:
                redutils.store_seg_props(ref, gillies_model.gillies_gdict, attr_name='initial_params')

        else:
            # Restore persistent SectionRef
            self.icell._all_refs = self._persistent_refs
            self.icell._soma_refs = gillies_model.get_soma_refs(self._persistent_refs)
            self.icell._dend_refs = [ref for ref in self._persistent_refs if ref not in self.icell._soma_refs]

            # Reset cell parameters to initial values after reduction
            for ref in self.icell._all_refs:
                redutils.set_range_props(ref, ref.initial_params)

        # Set physiological conditions
        # h = sim.neuron.h
        # h.celsius = 30 # will be overriden by possible global parameter in super.instantiate
        # h.v_init = -60
        # h.set_aCSF(4)

        # Call base class method
        return super(StnReducedModel, self).instantiate(sim)



class StnProtoModel(ephys.models.HocCellModel):
    '''
    Wraps the Gillies & Willshaw Hoc Template so it can be used by BluePyOpt

    TODO: use HocCellModel
        - this requires changes to the Gillies cell template to conform to BluePyOpt/HBP standards.
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