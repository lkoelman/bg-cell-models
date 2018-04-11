"""
Create synaptic connections using custom NEURON synapse models
and get their parameters from centralized parameter databvase.

@author     Lucas Koelman

@date       21/03/2018
"""

import re
import pyNN.neuron
from pyNN.neuron.simulator import Connection, state, h
from common import nrnutil


class ConnectionFromDB(Connection):
    """
    Same as NativeSynToRegion except it looks up its parameters in
    the synapse type's attached database, using the Projection's 
    region + receptor type, presynaptic and postsynaptic cell types.
    """
    #    The call graph is:
    #
    #    -> Projection.__init__(..., Connector
    #    `-> Connector.connect(..., Projection)
    #    `-> MapConnector._standard_connect(...)
    #        `-> Connector._parameters_from_synapse_type(...)
    #            - parameters are fetched from Projection.synapse_type
    #            - parameters are optionally calculated from distance map
    #    `-> Projection._convergent_connect(..., **connection_parameters)
    #    `-> Projection.synapse_type.connection_type(...)
    #    `-> Connection.__init__(...)

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.


        @pre        The synapse_type passed to Projection must have an attribute
                    'parameter_database' providing a method
                    make_parallel_connection()
        
        @pre        The projection must have an attribute 'preferred_param_sources'
                    containing the preferred sources used for looking up
                    parameters in the database.

        @pre        Each Population object needs to have an attribute
                    'pop_id' containing the population identifier recognized
                    by cellpopdata

        @param      parameters : **dict
                    Parameters retrieved from projection synapse type
        
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
        
        params_db = projection.synapse_type.parameter_database
        custom_physio_params = parameters.get('custom_physio_params', None)
        custom_nrn_params = parameters.get('custom_nrn_params', None)

        # Get suitable parameters for populations and receptors
        syn_mech_name = params_db.get_synaptic_mechanism(receptors)

        physio_params = params_db.get_physiological_parameters(
                            pre_pop_id, post_pop_id,
                            custom_params=custom_physio_params,
                            use_sources=projection.preferred_param_sourcesr)

        # Make a synapse and NetCon
        synapse_type = getattr(h, syn_mech_name)
        syn = synapse_type(segment)
        nc = state.parallel_context.gid_connect(pre_gid, syn)

        # Set the mechanism and NetCon parameters from phsyiological parameters
        params_db.set_connection_params_from_physio(
                    syn, nc, syn_mech_name, receptors, 
                    con_par_data=physio_params,
                    custom_synpar=custom_nrn_params)

        self.synapse = syn
        self.nc = nc

        # Also store synapse on postsynaptic cell
        syn_mech_name = nrnutil.get_mod_name(syn)
        post_syns = self.postsynaptic_cell._cell.synapses.setdefault(syn_mech_name, [])
        post_syns.append(syn)

        
        if projection.synapse_type.model is not None:
            self._setup_plasticity(projection.synapse_type, parameters)
        # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested


class NativeSynToRegion(Connection):
    """
    Connect a synaptic mechanism (MOD file name) to a cell region
    pecified as Ephys location.

    This connection type is only for use with NativeSynapse defined in 
    this module.

    USAGE
    -----

        >>> syn_params = {'netcon:weight[0]': 1.0, 'netcon:weight[1]: 0.5', 
        >>>               'netcon:delay': 3.0, 'syn:e_rev': 80.0,
        >>>               'syn:tau_fall': 15.0, 'syn:tau_rise': 5.0}
        >>> syn = sim.StaticSynapse(**syn_params)
        >>> syn.connection_type = NativeSynToRegion
    
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
        # syn_mech_name = parameters.pop('mechanism', None)
        syn_mech_name = projection.synapse_type.mechanism
        region, receptor_name = projection.receptor_type.split(".")
        if syn_mech_name is None:
            syn_mech_name = receptor_name
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

        # Interpret connection parameters
        param_targets = {'syn': self.synapse, 'netcon': self.nc}
        parameters.update(projection.synapse_type.mechanism_parameters)
        
        for param_spec, param_value in parameters.iteritems():
            # Default PyNN parameters
            if param_spec == 'weight':
                self.nc.weight[0] = param_value
            elif param_spec == 'delay':
                self.nc.delay = param_value
            else:
                # read param spec in format "target:attribute[index]"
                matches = re.search(
                    r'^(?P<target>\w+):(?P<parname>\w+)(\[(?P<index>\d+)\])?', 
                    param_spec)
    
                target_name = matches.group('target')
                param_name = matches.group('parname')
                param_index = matches.group('index')

                # Determine target (synapse or NetCon)
                target = param_targets[target_name]

                # Set attribute
                if param_index is None:
                    setattr(target, param_name, param_value)
                else:
                    getattr(target, param_name)[int(param_index)] = param_value


# We have to define a custom Synapse model so we can make the connection_type
# point to our custom Connection class
class SynapseFromDB(pyNN.neuron.StaticSynapse):
    """
    A NEURON native synapse that looks up the correct MOD mechanism
    and all its synapse parameters in the provided database, using the
    Projection's receptor type, presynaptic cell and postsynaptic cell.

    @see        pyNN.standardmodels.synapses.StaticSynapse

    USAGE
    -----

        >>> from cellpopdata import cellpopdata
        >>> params_db = cellpopdata.Cellconnector(...)
        >>> syn = SynapseFromDB(parameter_database=params_db)
        >>> proj = sim.Projection(pop_stn, pop_gpe, connector, db_syn,
        >>>>                      receptor_type="distal_dend.AMPA+NMDA")
    
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


class NativeSynapse(pyNN.neuron.StaticSynapse):
    """
    A NEURON native synapse model with an explicit MOD mechanism
    and mechanism parameters.

    You can specify any mechanism name as the 'mechanism' argument,
    and its parameters as 'mechanism_parameters'

    @see        pyNN.standardmodels.synapses.StaticSynapse
    """
    connection_type = NativeSynToRegion

    # default_parameters = {
    #     'weight': 0.0,
    #     'delay': None,
    #     'mechanism': 'Exp2Syn',
    #     'mechanism_parameters': None,
    # }

    def __init__(self, **parameters):
        """
        Make new NEURON native synapse

        @param      mechanism : str
                    NEURON synapse mechanism name

        @param      **parameters : (keyword arguments)
                    Only 'weight' 'delay' and 'mechanism_parameters'
                    are recognized keywords.

        @param      mechanism_parameters : dict(str, float)
                    Parameters of neuron mechanism and NetCon, in format
                    "syn:attribute[index]" and "netcon:attribute[index]".
        """
        # Don't pass native mechanism parameters
        self.mechanism = parameters.pop('mechanism')
        self.mechanism_parameters = parameters.pop('mechanism_parameters', {})
        # self.physiological_parameters = parameters.pop('physiological_parameters', {})
        
        # Insert dummy variables so pyNN doesn't complain
        parameters.setdefault('delay', None)
        parameters.setdefault('weight', 0.0)
        super(NativeSynapse, self).__init__(**parameters)