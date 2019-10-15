"""
Optimization of reduced neuron model using BluePyOpt.

@author Lucas Koelman

@date   12/09/2017


NOTES
-----

- see comments marked 'SETPARAM' for script/optimization parameters

"""

# BluePyOpt modules
import bluepyopt as bpop
import bluepyopt.ephys as ephys

# Our custom BluePyOpt modules
import bgcellmodels.extensions.bluepyopt.bpop_extensions as bpop_extensions
from bgcellmodels.cellpopdata import StnModel
from bgcellmodels.models.STN.GilliesWillshaw.optimize import (
    bpop_cellmodels as stn_models,
    bpop_protocols_stn as stn_protos,
    bpop_parameters_stn as stn_params,
    bpop_features_stn as stn_feats,
    bpop_analysis_stn as stn_anls,
)

# Gillies & Willshaw model mechanisms
from bgcellmodels.models.STN.GilliesWillshaw import gillies_model
gleak_name = gillies_model.gleak_name

# Physiology parameters

from bgcellmodels.common.stimprotocols import StimProtocol
CLAMP_PLATEAU = StimProtocol.CLAMP_PLATEAU
CLAMP_REBOUND = StimProtocol.CLAMP_REBOUND
MIN_SYN_BURST = StimProtocol.MIN_SYN_BURST
SYN_BACKGROUND_HIGH = StimProtocol.SYN_BACKGROUND_HIGH

# Adjust verbosity of loggers
import logging
logger = logging.getLogger('__main__')
# logutils.setLogLevel('quiet', ['marasco', 'folding', 'redops', 'bluepyopt.ephys.parameters'])



# SETPARAM: filepath of saved responses
PROTO_RESPONSES_FILE = "/home/luye/cloudstore_m/simdata/Gillies2005_fullmodel/responses/STN_Gillies2005_proto_responses_3.pkl" # backup is in filename.old.pkl


################################################################################
# OPTIMIZATION EXPERIMENTS
################################################################################


def test_protocol(stim_proto, model_type, export_locals=True):
    """
    Test stimulation protocol applied to full cell model.

    @param  model_type      either a CellModel object, StnModel enum instance,
                            or string "full" / "reduced"

    EXAMPLE:

        test_protocol(StimProtocol.MIN_SYN_BURST, 'full')
    
    """
    if model_type in ('full', StnModel.Gillies2005):
        model_type = StnModel.Gillies2005
        fullmodel = True
    elif model_type in ('reduced', StnModel.Gillies_FoldMarasco):
        model_type = StnModel.Gillies_FoldMarasco
        fullmodel = False


    # instantiate protocol
    proto = stn_protos.BpopProtocolWrapper.make(stim_proto, model_type)

    # Get protocol mechanisms that need to be isntantiated
    proto_mechs = proto.proto_vars.get('pp_mechs', []) + \
                  proto.proto_vars.get('range_mechs', [])

    proto_params = proto.proto_vars.get('pp_mech_params', [])

    # Make cell model
    if fullmodel:
        cellmodel = stn_models.StnFullModel(
                        name        = 'StnGillies',
                        mechs       = proto_mechs,
                        params      = proto_params)
    else:
        cellmodel = stn_models.StnReducedModel(
                        name        = 'StnFolded',
                        fold_method = 'marasco',
                        num_passes  = 7,
                        mechs       = proto_mechs,
                        params      = proto_params)

    nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

    # Apply protocol and simulate
    responses = proto.ephys_protocol.run(
                    cell_model      = cellmodel, 
                    param_values    = {},
                    sim             = nrnsim,
                    isolate         = False) # allows us to query cell with h.allsec()

    ## Plot protocol responses
    from matplotlib import pyplot as plt
    for resp_name, traces in responses.iteritems():
        plt.figure()
        plt.plot(traces['time'], traces['voltage'])
        plt.suptitle(resp_name)
    plt.show(block=False)

    if export_locals:
        globals().update(locals())


def inspect_protocol(stim_proto, model_type, export_locals=True):
    """
    Test stimulation protocol applied to full cell model.

    @param  model_type      either a CellModel object, StnModel enum instance,
                            or string "full" / "reduced"

    EXAMPLE USAGE:

    proto = synburst_protocol
    test_protocol(proto, 'full')
    
    """
    if model_type in ('full', StnModel.Gillies2005):
        model_type = StnModel.Gillies2005
        fullmodel = True
    elif model_type in ('reduced', StnModel.Gillies_FoldMarasco):
        model_type = StnModel.Gillies_FoldMarasco
        fullmodel = False


    # instantiate protocol
    proto = stn_protos.BpopProtocolWrapper.make(stim_proto, model_type)

    # Get protocol mechanisms that need to be isntantiated
    proto_mechs = proto.proto_vars.get('pp_mechs', []) + \
                  proto.proto_vars.get('range_mechs', [])

    proto_params = proto.proto_vars.get('pp_mech_params', [])

    # Make cell model
    if fullmodel:
        cellmodel = stn_models.StnFullModel(
                        name        = 'StnGillies',
                        mechs       = proto_mechs,
                        params      = proto_params)
    else:
        cellmodel = stn_models.StnReducedModel(
                        name        = 'StnFolded',
                        fold_method = 'marasco',
                        num_passes  = 7,
                        mechs       = proto_mechs,
                        params      = proto_params)

    nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

    # protocol.run() cleans up after running: therefore instantiate everything ourselves
    cellmodel.freeze({})
    cellmodel.instantiate(sim=nrnsim)
    proto.instantiate(sim=nrnsim, icell=cellmodel.icell)


    if export_locals:
        globals().update(locals())


def make_optimisation(red_model=None, parallel=False, export_locals=False):
    """
    Optimization routine for reduced cell model.

    @param  red_model : ephys.models.Model
            Cell model that instantiates the reduced cell model.
    """
    nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

    ############################################################################
    # Parameters & Cell Model

    # Make parameters
    # SETPARAM: passive parameters fit using PRAXIS
    gleak_orig = 7.84112e-05
    gleak_fit = 12.43169e-5 # fit to match Zin_DC (see praxis_passive.py)
    dend_gl_param = ephys.parameters.NrnSectionParameter(
                        name        = 'gleak_dend_param',
                        param_name  = gleak_name,
                        locations   = [stn_params.dendritic_region],
                        bounds      = [gleak_fit*1e-1, gleak_fit*1e1],
                        value       = gleak_orig, # SETPARAM: use fitted gl value
                        frozen      = True)

    cm_orig = 1.0
    cm_fit = cm_orig * (gleak_fit / gleak_orig) # preserve membrane time constant
    dend_cm_param = ephys.parameters.NrnSectionParameter(
                        name        = 'cm_dend_param',
                        param_name  = 'cm',
                        locations   = [stn_params.dendritic_region],
                        bounds      = [1.0, 1.0],
                        value       = cm_orig, # SETPARAM: use fitted cm value
                        frozen      = True)

    # FROZEN PARAMETERS are passive parameters fit previously in passive model
    frozen_params = [] # SETPARAM: frozen params from previous optimisations

    # FREE PARAMETERS are active conductances with large impact on response
    free_params = stn_params.dend_active_params # SETPARAM: parameters that are optimised (must be not frozen)

    # NOTE: we don't need to define NrnModMechanism for these parameters, since they are not inserted by BluePyOpt but by our custom model setup code

    ############################################################################
    # Protocols for optimisation

    stn_model_type = StnModel.Gillies_FoldMarasco_Legacy # SETPARAM: model type to optimise

    # Protocols to use for optimisation
    opt_stim_protocols = [CLAMP_REBOUND, MIN_SYN_BURST]

    # Make all protocol data
    red_protos = {
        stim_proto: stn_protos.BpopProtocolWrapper.make(stim_proto, stn_model_type) 
            for stim_proto in opt_stim_protocols
    }

    # Collect al frozen mechanisms and parameters required for protocols to work
    proto_mechs, proto_params = stn_protos.BpopProtocolWrapper.all_mechs_params(red_protos.values())

    # Distinguish between sets of parameters (used, frozen, free/optimised)
    frozen_params += proto_params
    used_params = frozen_params + free_params
    for param in frozen_params:
        assert param.frozen
    for param in free_params:
        assert (not param.frozen)

    # Create reduced model
    if red_model is None:
        red_model = stn_models.StnReducedModel(
                        name        = 'StnFolded',
                        fold_method = 'marasco',
                        num_passes  = 7,
                        mechs       = proto_mechs,
                        params      = used_params)
    else:
        red_model.set_mechs(proto_mechs)
        red_model.set_params(used_params)


    ############################################################################
    # Features

    # Get protocol responses for full model
    if PROTO_RESPONSES_FILE is not None:
        full_responses = stn_anls.load_proto_responses(PROTO_RESPONSES_FILE)
    else:
        full_protos = [
            stn_protos.BpopProtocolWrapper.make(stim_proto, stn_model_type)
                for stim_proto in opt_stim_protocols
        ]
        
        full_mechs, full_params = stn_protos.BpopProtocolWrapper.all_mechs_params(full_protos)
        full_model = stn_models.StnFullModel(
                        name        = 'StnGillies',
                        mechs       = full_mechs,
                        params      = full_params)
        
        full_responses = stn_anls.run_proto_responses(full_model, full_protos)

    # Make EFEL feature objects
    stimprotos_feats = stn_feats.make_opt_features(red_protos.values())

    # Calculate target values from full model responses
    stn_feats.calc_feature_targets(stimprotos_feats, full_responses)

    ############################################################################
    # TEST

    # default_params = {k : v*1.1 for k,v in stn_params.default_params.items() if k in [p.name for p in used_params]}

    # stim_proto = MIN_SYN_BURST
    # e_proto = red_protos[stim_proto]

    # # Compute protocol responses
    # responses = e_proto.ephys_protocol.run(
    #               cell_model      = red_model, 
    #               param_values    = default_params,
    #               sim             = nrnsim,
    #               isolate         = False) # allows us to query cell with h.allsec()

    # # Plot responses
    # from matplotlib import pyplot as plt
    # for resp_name, traces in responses.iteritems():
    #   plt.figure()
    #   plt.plot(traces['time'], traces['voltage'])
    #   plt.suptitle(resp_name)
    # plt.show(block=False)

    # # Calculate response scores
    # featdict = stimprotos_feats[stim_proto]
    # feats, weights = zip(*featdict.values())
    # for feat in feats:
    #   score = feat.calculate_score(responses)
    #   logger.debug('Score = <{}> for feature {}.{}'.format(score, stim_proto.name, feat.name))

    # import ipdb; ipdb.set_trace()

    ############################################################################
    # Objective / Fitness calculation

    # Collect characteristic features for all protocols used in evaluation
    all_opt_features, all_opt_weights = stn_feats.all_features_weights(
                                                    stimprotos_feats.values())

    # Make final objective function based on selected set of features
    # total_objective = ephys.objectives.WeightedSumObjective(
    #                           name    = 'optimise_all',
    #                           features= all_opt_features,
    #                           weights = all_opt_weights)
    total_objective = bpop_extensions.RootMeanSquareObjective(
                                name    = 'optimise_all',
                                features= all_opt_features)

    # Calculator maps model responses to scores
    fitcalc = ephys.objectivescalculators.ObjectivesCalculator([total_objective])

    # ALTERNATIVE: no weights
    # all_objectives = [ephys.objectives.SingletonObjective(f.name, f) for f in all_opt_features]
    # fitcalc = ephys.objectivescalculators.ObjectivesCalculator(all_objectives)

    ############################################################################
    # Evaluators & Optimization

    # Make evaluator to evaluate model using objective calculator
    opt_ephys_protos = {k.name: v.ephys_protocol for k,v in red_protos.iteritems()}
    opt_params_names = [param.name for param in free_params]
    
    # TODO: check that: synapse parameters must be on model but unactive during clamp protocols
    cell_evaluator = ephys.evaluators.CellEvaluator(
                        cell_model          = red_model,
                        param_names         = opt_params_names, # fitted parameters
                        fitness_protocols   = opt_ephys_protos,
                        fitness_calculator  = fitcalc,
                        sim                 = nrnsim,
                        isolate_protocols   = True) # SETPARAM: enable multiprocessing

    # Make optimisation using the model evaluator
    optimisation = bpop.optimisations.DEAPOptimisation(
                        evaluator       = cell_evaluator,
                        offspring_size  = 10,
                        map_function = get_map_function(parallel))

    # Save optimisation data
    opt_data = {
        'stim_protocols_features': stimprotos_feats,
        'ephys_protocols':      opt_ephys_protos,
        'frozen_params':        frozen_params,
        'free_params':          free_params,
        'objective':            total_objective,
        'fitness_calculator':           fitcalc,
        'evaluator':            cell_evaluator,
        'optimisation':         optimisation,
    }
    # opt_data = {k:v for k,v in locals().iteritems() if k in save_vars}

    if export_locals:
        globals().update(locals())

    return optimisation, opt_data


def optimize_active(**kwargs):
    """
    Make optimisation and run it.

    @param  max_ngen : int
            Maximum number of generations in the evolutionary optimization

    @param  **kwargs
            Keyword arguments passed to `make_optimisation()`
    """
    num_generations = kwargs.pop('max_ngen', 5)

    # Make deapext.optimisation object
    optimisation, opt_data = make_optimisation(**kwargs)

    # Run optimisation for fixed number of generations
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=num_generations)

    return final_pop, hall_of_fame, logs, hist


def get_map_function(parallel):
    """
    Return a map() function for evaluating individuals in the population.

    @param  parallel    bool: whether you want parallel execution of individuals.

    @see    taken from example in https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/opt_l5pc.py

    NOTES

    To use ipyparallel you must start a controller and a number of 
    engine instances before starting ipython (see 
    http://ipyparallel.readthedocs.io/en/latest/intro.html), e.g:

        `ipcluster start -n 6`
    
    Make sure that Hoc can find the .hoc model files by either executing above
    command in the directory containing those files, or adding the relevant directories
    to the environment variable `$HOC_LIBRARY_PATH` (this could also be done in 
    your protocol or cellmodel script using os.environ["HOC_LIBRARY_PATH"])
    """
    if not parallel:
        return None # DEAPOptimisation will use its default map function

    from datetime import datetime
    from ipyparallel import Client
    import socket
    
    # Create a connection to the server
    rc = Client() # if profile specified: searches JSON file in ~/.ipython/profile_name
    logger.debug('Using ipyparallel with %d engines', len(rc))

    # Create a view of all the workers (ipengines)
    lview = rc.load_balanced_view()
    host_names = lview.apply_sync(socket.gethostname) # run gethostname on all ipengines
    if isinstance(host_names, str):
        host_names = [host_names]
    print('Ready for parallel execution on folowing hosts: {}'.format(', '.join(host_names)))

    def mapper(func, it):
        start_time = datetime.now()
        ret = lview.map_sync(func, it)
        logger.debug('Generation took %s', datetime.now() - start_time)
        return ret

    return mapper


def save_fullmodel_responses(plot=True):
    """
    Save responses to all stimulation protocols for full model to file.

    @effect     responses are saved to pickle file specified in variable
                PROTO_RESPONSES_FILE
    """
    # Save full model responses
    model_type = StnModel.Gillies2005

    # Make all available protocols
    all_stim_protos = stn_protos.PROTOCOL_WRAPPERS.keys()
    full_protos = [stn_protos.BpopProtocolWrapper.make(stim_proto, model_type) for stim_proto in all_stim_protos]

    # Make cell model
    full_mechs, full_params = stn_protos.BpopProtocolWrapper.all_mechs_params(full_protos)
    full_model = stn_models.StnFullModel(
                    name        = 'StnGillies',
                    mechs       = full_mechs,
                    params      = full_params)

    # Run protocols and save responses
    ephys_protos = [proto.ephys_protocol for proto in full_protos]
    full_responses = stn_anls.run_proto_responses(full_model, ephys_protos)
    
    stn_anls.save_proto_responses(full_responses, PROTO_RESPONSES_FILE)

    if plot:
        stn_anls.plot_proto_responses(full_responses)


if __name__ == '__main__':
    # Calculate features for each optimisation protocol
    # proto_feats = get_features_targets()

    # Load and plot
    responses = stn_anls.load_proto_responses(PROTO_RESPONSES_FILE)
    stn_anls.plot_proto_responses(responses)

    # Do optimization
    optimize_active(max_ngen=50)