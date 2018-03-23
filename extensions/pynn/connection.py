"""
Create synaptic connections using custom NEURON synapse models
and get their parameters from centralized parameter databvase.

@author     Lucas Koelman

@date       21/03/2018
"""

from pyNN.neuron import StaticSynapse
from pyNN.neuron.simulator import Connection, state, h
from common import nrnutil


class ConnectionFromDB(Connection):

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.

        @pre        The synapse_type passed to Projection must have an attribute
                    'parameter_database' providing a method
                    make_parallel_connection()

        @pre        Each Population object needs to have an attribute
                    'pop_id' containing the population identifier recognized
                    by cellpopdata
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]    # Population[index] -> ID
        self.postsynaptic_cell = projection.post[post] # Population[index] -> ID

        # Get our population identifiers
        pre_pop_id = projection.pre.pop_id
        post_pop_id = projection.post.pop_id

        # Get the target region on the cell and the receptor type
        region, receptor = projection.receptor_type.split(".")
        receptors = receptor.split("+")
        segment = getattr(self.postsynaptic_cell._cell, region)
        pre_gid = int(self.presynaptic_cell)
        
        # TODO: allow custom_conpar/custom_synpar in **parameters, and pass them on
        params_db = projection.synapse_type.parameter_database
        syn, nc = params_db.make_parallel_connection(
                    state.parallel_context, 
                    pre_pop_id, post_pop_id,
                    segment, pre_gid, receptors)

        self.synapse = syn
        self.nc = nc

        # Also store synapse on postsynaptic cell
        syn_mech_name = nrnutil.get_mod_name(syn)
        post_syns = self.postsynaptic_cell._cell.synapses.setdefault(syn_mech_name, [])
        post_syns.append(syn)

        
        if projection.synapse_type.model is not None:
            self._setup_plasticity(projection.synapse_type, parameters)
        # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested


class SynapseToCellRegion(Connection):
    """
    Connectt a synaptic mechanism (MOD file name)
    to a cell region specified as Ephys location.

    USAGE
    -----

        >>> syn = sim.StaticSynapse(**syn_mech_params)
        >>> syn.connection_type = SynapseToCellRegion
    
    """

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.

        @pre    The region specified in the first part of the
                projection.receptor_type must be present as an attribute
                on the post-synaptic cell.
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]
        self.postsynaptic_cell = projection.post[post]

        # Get the target region on the cell and the receptor type
        region, syn_mech_name = projection.receptor_type.split(".")
        segment = getattr(self.postsynaptic_cell._cell, region)
        
        # Find a synapse to connect to or create one
        seg_pps = segment.point_processes()
        target_syn = next((syn for syn in seg_pps if 
                        nrnutil.get_mod_name(syn)==syn_mech_name), None)
        if target_syn is None:
            constructor = getattr(h, syn_mech_name)
            target_syn = constructor(segment)
            existing_syn = True
        else:
            existing_syn = False
        
        pre_gid = int(self.presynaptic_cell)
        self.nc = state.parallel_context.gid_connect(pre_gid, target_syn)
        self.synapse = target_syn

        # Also store synapse on postsynaptic cell
        if not existing_syn:
            post_syns = self.postsynaptic_cell._cell.synapses.setdefault(syn_mech_name, [])
            post_syns.append(target_syn)

        # TODO: set parameters using interpretParamSpec (synmech.py)
        self.nc.weight[0] = parameters.pop('weight')
        # if we have a mechanism (e.g. from 9ML) that includes multiple
        # synaptic channels, need to set nc.weight[1] here
        if self.nc.wcnt() > 1 and hasattr(self.postsynaptic_cell._cell, "type"):
            self.nc.weight[1] = self.postsynaptic_cell._cell.type.receptor_types.index(projection.receptor_type)
        
        self.nc.delay = parameters.pop('delay')
        if projection.synapse_type.model is not None:
            self._setup_plasticity(projection.synapse_type, parameters)
        # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested


# We have to define a custom Synapse model so we can make the connection_type
# point to our custom Connection class
class SynapseFromDB(StaticSynapse):
    """
    Same as StaticSynapse except we specify our own connection_type.

    This synapse figures out the synapse parameters from the
    Projection.receptor_type and pre- and post-synaptic cell types.

    @see        pyNN.standardmodels.synapses.StaticSynapse
    """
    connection_type = ConnectionFromDB

    def __init__(self, **parameters):
        """
        Make new synapse type for use with a Projection.

        @param      parameters_database : object

        """
        self.parameter_database = parameters.pop('parameter_database')
        # Insert dummy variables so pyNN doesn't complain
        parameters['delay'] = 1.0
        parameters['weight'] = 1.0
        super(SynapseFromDB, self).__init__(**parameters)