"""
Create synaptic connections using custom NEURON synapse models
and get their parameters from centralized parameter databvase.

@author     Lucas Koelman

@date       21/03/2018
"""

import re

import pyNN.neuron
from pyNN.neuron.simulator import Connection, state, h
from pyNN.standardmodels import synapses, build_translations
from bgcellmodels.common import nrnutil

#    The call graph for creating Connections is:
#
#    -> Projection.__init__(..., Connector)
#    `-> Connector.connect(..., Projection)
#    `-> MapConnector._standard_connect(...)
#        `-> Connector._parameters_from_synapse_type(...)
#            - parameters are fetched from Projection.synapse_type
#            - parameters are optionally calculated from distance map
#    `-> Projection._convergent_connect(..., **connection_parameters)
#    `-> Projection.synapse_type.connection_type(...)
#    `-> Connection.__init__(...)


class ConnectionFromDB(Connection):
    """
    Same as NativeSynToRegion except it looks up its parameters in
    the synapse type's attached database, using the Projection's 
    region + receptor type, presynaptic and postsynaptic cell types.
    """

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.


        @pre        The synapse_type passed to Projection must have an attribute
                    'parameter_database' providing a method
                    make_parallel_connection()
        
        @pre        The projection must have an attribute 'preferred_param_sources'
                    containing the preferred sources used for looking up
                    parameters in the database.

        @pre        The 'label' attribute of each population must be an identifier
                    recognized by cellpopdata

        @param      parameters : **dict
                    Parameters retrieved from projection synapse type
        
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]    # Population[index] -> ID
        self.postsynaptic_cell = projection.post[post] # Population[index] -> ID

        # Get our population identifiers
        pre_pop_id = projection.pre.label
        post_pop_id = projection.post.label

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

    This Connection class can represent a multi-synaptic connection and hence 
    encapsulate multiple NEURON synapse and NetCon objects.

    This Connection class can obly be used with NativeSynapse defined in 
    this module.

    USAGE
    -----

        >>> syn_params = {'netcon:weight[0]': 1.0, 'netcon:weight[1]: 0.5', 
        >>>               'netcon:delay': 3.0, 'syn:e_rev': 80.0,
        >>>               'syn:tau_fall': 15.0, 'syn:tau_rise': 5.0}
        >>>
        >>> syn = NativeSynapse(mechanism='GLUsyn',
        >>>                     mechanism_parameters=syn_params)
        >>>
        >>> proj = sim.Projection(pop_pre, pop_post, connector, syn,
                                  receptor_type='distal_region.AMPA+NMDA')
    
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
        post_cell = self.postsynaptic_cell._cell # CellType.model instance

        # Get the target region on the cell and the receptor type
        # syn_mech_name = parameters.pop('mechanism', None)
        # syn_mech_name = projection.synapse_type.mechanism
        region, receptor = projection.receptor_type.split(".")
        receptors = receptor.split("+")


        # TODO: handle distribution of synapses better
        #   - cell is responsible for maintaining realistic distribution of synapses
        #       - maintain spacing, etc.
        #   - function to get multiple segments or synapses
        #       - args: pre_cell, receptor, region, number of synaptic contacts
        #   - can give existing synapses if parameters are the same
        #       - no overwrite of params in this case
        #       - check by comparing same pre-cel, receptor, region,


        # Ask cell for segment in target region
        synapse, used = post_cell.get_synapse(region, receptors, True)
        self.synapse = synapse
        if used > 0:
            raise Exception("No unused synapses on target cell {}".format())
        
        
        # Create NEURON synapse and NetCon
        pre_gid = int(self.presynaptic_cell)
        self.nc = state.parallel_context.gid_connect(pre_gid, self.synapse)

        # Interpret connection parameters
        param_targets = {'syn': self.synapse, 'netcon': self.nc}
        parameters.update(projection.synapse_type.mechanism_parameters)

        # Don't allow weight and delay in **parameters
        pynn_delay = parameters.pop('delay')
        parameters.setdefault('netcon:delay', pynn_delay)

        pynn_weight = parameters.pop('weight')
        parameters.setdefault('netcon:weight[0]', pynn_weight)
        
        for param_spec, value_spec in parameters.iteritems():
            # Convert value specification to actual parameter value
            if callable(value_spec):
                # Generate distance map lazily and cache it
                distance_map = getattr(projection, '_distance_map', None)
                if distance_map is None:
                    distance_map = projection._connector._generate_distance_map(
                                                            projection)
                    projection._distance_map = distance_map
                # distance_map has projection.shape
                distance = distance_map[pre, post]
                param_value = value_spec(distance)
            elif isinstance(value_spec, (float, int)):
                param_value = value_spec
            else:
                raise ValueError("Cannot interpret parameter specification "
                    "<{}> for parameter {}".format(value_spec, param_spec))

            # interpret parameter specification in format "target:attribute[index]"
            matches = re.search(
                r'^(?P<target>\w+):(?P<parname>\w+)(\[(?P<index>\d+)\])?', 
                param_spec)
            target_name = matches.group('target')
            param_name = matches.group('parname')
            param_index = matches.group('index')

            # Set attribute
            target = param_targets[target_name]
            if param_index is None:
                setattr(target, param_name, param_value)
            else:
                getattr(target, param_name)[int(param_index)] = param_value


class ConnectionNrnWrapped(Connection):
    """
    Connection to a wrapped NEURON synapse that exposes certain
    parameters to PyNN

    The advantage is that you can specify parameters using PyNN's built-in
    mechanisms. E.g.
    """

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]
        self.postsynaptic_cell = projection.post[post]
        post_cell = self.postsynaptic_cell._cell # CellType.model instance

        # Get the target region on the cell and the receptor type
        # syn_mech_name = projection.synapse_type.mechanism
        region, receptor = projection.receptor_type.split(".")
        receptors = receptor.split("+")

        # Ask cell for segment in target region
        synapse, used = post_cell.get_synapse(region, receptors, True)
        self.synapse = synapse
        if used > 0 and not post_cell.allow_synapse_reuse:
            raise Exception("No unused synapses on target cell {}".format(type(post_cell)))
        
        # Create NEURON NetCon
        pre_gid = int(self.presynaptic_cell)
        self.nc = state.parallel_context.gid_connect(pre_gid, synapse)
        self.nc.weight[0] = parameters.pop('weight')
        self.nc.delay = parameters.pop('delay')
        
        # if we have a mechanism (e.g. from 9ML) that includes multiple
        # synaptic channels, need to set nc.weight[1] here
        if self.nc.wcnt() > 1 and hasattr(self.postsynaptic_cell._cell, "type"):
            self.nc.weight[1] = self.postsynaptic_cell._cell.type.receptor_types.index(projection.receptor_type)
        
        # Plastic synapses use SynapseType.model to store weight adjuster,
        # we use it for the synapse model
        if projection.synapse_type.model is not None:
            self._setup_nrn_synapse(projection.synapse_type, parameters)
            # self._setup_plasticity(projection.synapse_type, parameters)


    def __setattr__(self, name, value):
        """
        Support properties of associated Synapse object.

        This is required to support Projection.set(param_name=param_val)

        NOTES
        -----

        In PyNN.neuron.simulator.py, this is solved by generating Connector
        properties for each attribute of the associated synapse type 
        and weight adjuster. We catch the property assignments in this method
        so we don't have to create explicit properties.
        """
        if hasattr(self, 'synapse_type') and name in self.synapse_type.get_parameter_names():
            pinfo = self.synapse_type.translations[name]
            pname = pinfo['translated_name']
            setattr(self.synapse, pname, value)
        else:
            super(ConnectionNrnWrapped, self).__setattr__(name, value)


    def _setup_nrn_synapse(self, synapse_type, parameters):
        """
        Set parameters on the NEURON synapse object.

        @param      parameters : dict(str, float)
                    Parameters are already evaluated by PyNN
        """
        # parameters = synapse_type.translate(parameters<ParameterSpace>, copy=copy)
        # parameters.evaluate(simplify=True)
        # for name, value in parameters.items():
        #   setattr(target, name, value)
        synapse_param_names = synapse_type.get_parameter_names()
        for name, value in parameters.items():
            if name in synapse_param_names:
                pinfo = synapse_type.translations[name]
                pname = pinfo['translated_name']
                setattr(self.synapse, pname, value)


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

        >>> from bgcellmodels.cellpopdata import cellpopdata
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
        self.multi_synapse_rule = parameters.pop('multi_synapse_rule', 1)
        # self.physiological_parameters = parameters.pop('physiological_parameters', {})

        # Convert distance-based string expressions to callable functions
        converted_expressions = {}
        for param_spec, value_spec in self.mechanism_parameters.iteritems():
            if not isinstance(value_spec, str):
                continue
            d_expression = value_spec
            try:
                # Check for singularities at large and small distances
                d = 0; assert 0 <= eval(d_expression), eval(d_expression)
                d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
            except ZeroDivisionError as err:
                raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
            converted_expressions[param_spec]= eval("lambda d: {}".format(d_expression))
        self.mechanism_parameters.update(converted_expressions)
        
        # Insert dummy variables so pyNN doesn't complain
        parameters.setdefault('delay', 1.0)
        parameters.setdefault('weight', 0.0)
        super(NativeSynapse, self).__init__(**parameters)



class GluSynapse(pyNN.neuron.BaseSynapse, synapses.StaticSynapse):
    """
    Wrapper for NEURON GLUsyn mechanism defined in .mod file
    """

    connection_type = ConnectionNrnWrapped
    model = 'GLUsyn'

    # PyNN internal name to NEURON name
    translations = build_translations(
        # NetCon parameters
        ('weight', 'weight'),
        ('delay', 'delay'),
        # Conductance time course
        ('gmax_AMPA', 'gmax_AMPA'),     # Weight conversion factor (from nS to uS)
        ('gmax_NMDA', 'gmax_NMDA'),     # Weight conversion factor (from nS to uS)
        ('tau_r_AMPA', 'tau_r_AMPA'),   # Dual-exponential conductance profile
        ('tau_d_AMPA', 'tau_d_AMPA'),   # IMPORTANT: tau_r < tau_d
        ('tau_r_NMDA', 'tau_r_NMDA'),   # Dual-exponential conductance profile
        ('tau_d_NMDA', 'tau_d_NMDA'),    # IMPORTANT: tau_r < tau_d
        # Short-term Depression/Facilitation
        ('tau_rec', 'tau_rec'),         # time constant of recovery from depression
        ('tau_facil', 'tau_facil'),     # time constant of facilitation
        ('U1', 'U1'),                   # baseline release probability
        # Magnesium block for NMDA
        ('e', 'e'),                     # AMPA and NMDA reversal potential
        ('mg', 'mg'),                   # Initial concentration of mg2+
    )

    default_parameters = {
        'weight':       1.0,
        'delay':        0.5,
        'tau_r_AMPA':   0.2,
        'tau_d_AMPA':   1.7,
        'tau_r_NMDA':   0.29,
        'tau_d_NMDA':   43.0,
        'e':            0.0,
        'mg':           1.0,
        'gmax_AMPA':    0.001,
        'gmax_NMDA':    0.001,
        'tau_rec':      200.0,
        'tau_facil':    200.0,
        'U1':           0.5,
    }

    def _get_minimum_delay(self):
        return state.min_delay


class GabaSynapse(pyNN.neuron.BaseSynapse, synapses.StaticSynapse):
    """
    Wrapper for NEURON GLUsyn mechanism defined in .mod file
    """

    connection_type = ConnectionNrnWrapped
    model = 'GABAsyn' # defined in GABAsyn.mod

    # PyNN internal name to NEURON name
    translations = build_translations(
        # NetCon parameters
        ('weight', 'weight'),
        ('delay', 'delay'),
        # Conductance time course
        ('gmax_GABAA', 'gmax_GABAA'),     # Weight conversion factor (from nS to uS)
        ('gmax_GABAB', 'gmax_GABAB'),     # Weight conversion factor (from nS to uS)
        ('tau_r_GABAA', 'tau_r_GABAA'),   # Dual-exponential conductance profile
        ('tau_d_GABAA', 'tau_d_GABAA'),   # IMPORTANT: tau_r < tau_d
        ('tau_r_GABAB', 'tau_r_GABAB'),   # Dual-exponential conductance profile
        ('tau_d_GABAB', 'tau_d_GABAB'),    # IMPORTANT: tau_r < tau_d
        # Short-term Depression/Facilitation
        ('tau_rec', 'tau_rec'),         # time constant of recovery from depression
        ('tau_facil', 'tau_facil'),     # time constant of facilitation
        ('U1', 'U1'),                   # baseline release probability
        # Reversal potentials
        ('Erev_GABAA', 'Erev_GABAA'),                     # GABAA and GABAB reversal potential
        ('Erev_GABAB', 'Erev_GABAB'),                   # Initial concentration of mg2+
        # TODO: include GABA-B signaling cascade if necessary
    )

    default_parameters = {
        'weight':     1.0,
        'delay':      0.5,
        'tau_r_GABAA':   0.2,
        'tau_d_GABAA':   1.7,
        'tau_r_GABAB':   0.2,
        'tau_d_GABAB':   1.7,
        'Erev_GABAA':   -80.0,
        'Erev_GABAB':   -95.0,
        'gmax_GABAA':    0.001,
        'gmax_GABAB':    0.001,
        'tau_rec':      200.0,
        'tau_facil':    200.0,
        'U1':           0.5,
    }

    def _get_minimum_delay(self):
        return state.min_delay


class GabaSynTmHill(pyNN.neuron.BaseSynapse, synapses.StaticSynapse):
    """
    Tsodyks-Markram synapse for GABA-A and GABA-B receptors with GABA-B
    conductance expressed as hill function applied to Tsodyks-Markram
    conductance variable.
    
    @see    mechanism GABAsyn2.mod
    """

    connection_type = ConnectionNrnWrapped
    model = 'GABAsyn2' # defined in GABAsyn.mod

    # PyNN internal name to NEURON name
    translations = build_translations(
        # NetCon parameters
        ('weight', 'weight'),
        ('delay', 'delay'),
        # Conductance time course
        ('gmax_GABAA', 'gmax_GABAA'),     # Weight conversion factor (from nS to uS)
        ('gmax_GABAB', 'gmax_GABAB'),     # Weight conversion factor (from nS to uS)
        ('tau_r_GABAA', 'tau_r_GABAA'),   # Dual-exponential conductance profile
        ('tau_d_GABAA', 'tau_d_GABAA'),   # IMPORTANT: tau_r < tau_d
        ('tau_r_GABAB', 'tau_r_GABAB'),   # Dual-exponential conductance profile
        ('tau_d_GABAB', 'tau_d_GABAB'),    # IMPORTANT: tau_r < tau_d
        # Short-term Depression/Facilitation
        ('tau_rec', 'tau_rec'),         # time constant of recovery from depression
        ('tau_facil', 'tau_facil'),     # time constant of facilitation
        ('U1', 'U1'),                   # baseline release probability
        # Reversal potentials
        ('Erev_GABAA', 'Erev_GABAA'),                     # GABAA and GABAB reversal potential
        ('Erev_GABAB', 'Erev_GABAB'),                   # Initial concentration of mg2+
        # GABA-B signaling cascade
        ('K3', 'K3'),
        ('K4', 'K4'),
        ('KD', 'KD'),
        ('n', 'n'),
    )

    default_parameters = {
        'weight':       1.0,
        'delay':        0.5,
        'tau_r_GABAA':  0.2,
        'tau_d_GABAA':  1.7,
        'tau_r_GABAB':  0.2,
        'tau_d_GABAB':  1.7,
        'Erev_GABAA':   -80.0,
        'Erev_GABAB':   -95.0,
        'gmax_GABAA':   0.001,
        'gmax_GABAB':   0.001,
        'tau_rec':      200.0,
        'tau_facil':    200.0,
        'U1':           0.5,
        'K3':           0.098,
        'K4':           0.033,
        'KD':           100.0,
        'n':            4,
    }

    def _get_minimum_delay(self):
        return state.min_delay