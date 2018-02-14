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

import collections

import bluepyopt.ephys as ephys

import gillies_model
from reducemodel import redutils, reduce_cell
from evalmodel import cellpopdata as cpd


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

        # SelfContainedProtocol interaction
        self.proto_setup_funcs = []
        self.proto_setup_kwargs = {}


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


    def exec_proto_setup_funcs(self, icell=None):
        """
        Execute protocol setup functions passed by SelfContainedProtocol.
        """
        setup_kwargs = { 'icell': icell }
        setup_kwargs.update(self.proto_setup_kwargs)

        for setup_func in self.proto_setup_funcs:
            setup_func(**setup_kwargs)


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

        # FIXME: uncommenting below causes crash/hang
        # self.proto_setup_kwargs = None
        # self.proto_setup_funcs = None

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

        # Call base class method
        icell = super(StnFullModel, self).instantiate(sim)

        # Call protocol setup functions if any
        self.exec_proto_setup_funcs(icell=icell)

        return icell


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

        # Reduction variables
        self._fold_method = fold_method
        self._num_passes = num_passes

        self._persistent_refs = None


    def first_instantiate(self, sim=None):
        """
        First instantiation of the cell model.
        """

        # Create Reduction object (creates full model)
        if self._fold_method == 'marasco':
            reduction = reduce_cell.gillies_marasco_reduction(tweak=False)
        elif self._fold_method == 'stratford':
            reduction = reduce_cell.gillies_stratford_reduction()

        # Apply pre-reduction protocol setup functions
        full_cell = Cell() # dummy cell
        full_cell.dendritic = [ref.sec for ref in reduction.dend_refs]
        self.exec_proto_setup_funcs(icell=full_cell)

        # Prepare synapse mapping
        if 'do_map_synapses' in self.proto_setup_kwargs:
            # Get synapses on full model
            syns_tomap = cpd.get_synapse_data(
                                self.proto_setup_kwargs['connector'],
                                self.proto_setup_kwargs['stim_data']['synapses'],
                                self.proto_setup_kwargs['stim_data']['syn_NetCons'])
            reduction.set_syns_tomap(syns_tomap)

        
        # Do reduction
        reduction.reduce_model(num_passes=self._num_passes, map_synapses=False)
        
        # Set all Sections on instantiated cell
        self.icell._soma_refs = reduction._soma_refs
        self.icell._dend_refs = reduction._dend_refs
        self.icell._all_refs = self.icell._soma_refs + self.icell._dend_refs
        # save SectionRef across model instantiation (icell will be destroyed)
        self._persistent_refs = self.icell._all_refs

        # Apply parameters _before mapping_ (mapping measures electrotonic properties)
        icell = super(StnReducedModel, self).instantiate(sim)

        # Do synapse mapping
        if 'do_map_synapses' in self.proto_setup_kwargs:

            reduction.map_synapses()

            # Update stim data
            self.proto_setup_kwargs['stim_data']['synapses'] = [s.mapped_syn for s in reduction.map_syn_info]
            self.proto_setup_kwargs['stim_data']['syn_NetCons'] = sum((s.afferent_netcons for s in reduction.map_syn_info), [])

        # Save initial cell parameters
        for ref in self.icell._all_refs:
            redutils.store_seg_props(ref, gillies_model.gillies_gdict, attr_name='initial_params')

        return icell


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator
        """

        # Create a cell
        self.icell = Cell()
        self.icell.gid = self.gid
        

        # Make sure original STN cell is built
        if self._persistent_refs is None:
            # first instantiation does cell reduction
            return self.first_instantiate(sim=sim)

        else:
            # Restore persistent SectionRef
            self.icell._all_refs = self._persistent_refs
            self.icell._soma_refs = gillies_model.get_soma_refs(self._persistent_refs)
            self.icell._dend_refs = [ref for ref in self._persistent_refs if ref not in self.icell._soma_refs]

            # Reset cell parameters to initial values after reduction
            for ref in self.icell._all_refs:
                redutils.set_range_props(ref, ref.initial_params)

            # Call base class method
            return super(StnReducedModel, self).instantiate(sim)


    def destroy(self, sim=None):
        """
        Destroy cell from simulator
        """
        logger.debug("Destroying Reduced cell model")
        self._persistent_refs = None
        return super(StnReducedModel, self).destroy(sim=sim)