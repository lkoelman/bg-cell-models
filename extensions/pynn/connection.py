"""
Create synaptic connections using custom NEURON synapse models
and get their parameters from centralized parameter databvase.

@author     Lucas Koelman

@date       21/03/2018
"""

from pyNN.neuron import StaticSynapse
from pyNN.neuron.simulator import Connection, state


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
        segment = getattr(self.postsynaptic_cell._cell, region)
        pre_gid = int(self.presynaptic_cell)
        
        # self.nc = state.parallel_context.gid_connect(int(self.presynaptic_cell), target_pp)
        params_db = projection.synapse_type.parameter_database
        pp, nc = params_db.make_parallel_connection(
                    state.parallel_context, 
                    pre_pop_id, post_pop_id,
                    segment, pre_gid, [receptor])

        self.pp = pp
        self.nc = nc
        
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