"""
STN cell as BluePyOpt CellModel for use with CERSEI toolbox

@author Lucas Koelman

@date   30/01/2018
"""

from cersei.collapse.optimize.cellmodels import Cell, CollapsableCell

import gillies_model
import cersei_reduce
from reducemodel import redutils
import cellpopdata as cpd




import logging
logger = logging.getLogger('gillies_cersei')


class StnCellReduced(CollapsableCell):
    
    def __init__(self, 
            name=None,
            reduction_method=None,
            reduction_params=None,
            mechs=None,
            params=None,
            gid=0):
        """
        Constructor

        Args: see StnBaseModel
        """

        super(StnCellReduced, self).__init__(name, mechs, params, gid)

        # Reduction variables
        self.reduction_method = reduction_method
        self.reduction_params = {} if reduction_params is None else reduction_params

        self._persistent_refs = None


    def instantiate(self, sim=None):
        """
        Instantiate cell in simulator. This is the method called by BluePyOpt.
        """

        # Create a cell
        self.icell = Cell()
        self.icell.gid = self.gid
        

        # Make sure original STN cell is built
        if self.reduction_method in [None, 'original', 'none', 'nonreduced']:
            return self.instantiate_nonreduced(sim=sim)
        
        elif self._persistent_refs is None:
            # first instantiation does cell reduction
            return self.instantiate_reduced(sim=sim)

        else:
            return self.reset_instantiated(sim=sim)


    def reset_instantiated(self, sim=None):
        """
        Reset previously instantiated cell.
        """
        # Restore persistent SectionRef
        self.icell._all_refs = self._persistent_refs
        self.icell._soma_refs = gillies_model.get_soma_refs(self._persistent_refs)
        self.icell._dend_refs = [ref for ref in self._persistent_refs if ref not in self.icell._soma_refs]

        # Reset cell parameters to initial values after reduction
        for ref in self.icell._all_refs:
            redutils.set_range_props(ref, ref.initial_params)

        # Call base class method
        return super(StnCellReduced, self).instantiate(sim)


    def instantiate_nonreduced(self, sim=None):
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
        icell = super(StnCellReduced, self).instantiate(sim)

        # Call protocol setup functions if any
        self.exec_proto_setup_funcs(icell=icell)

        return icell


    def instantiate_reduced(self, sim=None):
        """
        First instantiation of the cell model.
        """

        # Create Reduction object (creates full model)
        reduction = cersei_reduce.make_reduction(self.reduction_method, tweak=False)

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
        num_passes = self.reduction_params.get('num_collapse_passes', 1)
        reduction.reduce_model(num_passes=num_passes, map_synapses=False)
        
        # Set all Sections on instantiated cell
        self.icell._soma_refs = reduction._soma_refs
        self.icell._dend_refs = reduction._dend_refs
        self.icell._all_refs = self.icell._soma_refs + self.icell._dend_refs
        # save SectionRef across model instantiation (icell will be destroyed)
        self._persistent_refs = self.icell._all_refs

        # Apply parameters _before mapping_ (mapping measures electrotonic properties)
        icell = super(StnCellReduced, self).instantiate(sim)

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


    def destroy(self, sim=None):
        """
        Destroy cell from simulator
        """
        logger.debug("Destroying Reduced cell model")
        self._persistent_refs = None
        return super(StnCellReduced, self).destroy(sim=sim)