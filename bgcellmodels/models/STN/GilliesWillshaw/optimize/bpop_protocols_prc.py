"""
Stimulation protocols for measuring Phase Response Curves (PRC)
with BluePyOpt.

@author Lucas Koelman
@date   16/10/2019


USAGE
-----

- see comments marked 'SETPARAM' for free parameters

"""

# Third party
import bluepyopt.ephys as ephys

# Custom modules
from bgcellmodels.extensions.bluepyopt import bpop_recordings, bpop_stimuli
from bgcellmodels.extensions.bluepyopt.bpop_protocol_ext import (
    SelfContainedProtocol, PhysioProtocol, BpopProtocolWrapper,
    rng_getter, connector_getter, PROTOCOL_WRAPPERS
)
from bgcellmodels.common import stimprotocols

# Module globals
StimProtocol = stimprotocols.StimProtocol


def init_stn_physiology(sim, model):
    """
    Initialize simulator to run plateau protocol

    NOTE: function must be declared at top-level of module in order to be pickled
    """
    h = sim.neuron.h
    h.celsius = 30
    h.v_init = -60
    h.set_aCSF(4)
    h.init()


class PhaseResponseSynExcDist(BpopProtocolWrapper):
    """
    Protocol for measurin phase response curves.

    Implementation using standard BluePyOpt classes: SweepProtocol,
    NrnNetStimStimulus, NrndPoinrProcessLocation.

    Based on example BluePyOpt/examples/expsyn/expsyn.py
    """

    IMPL_PROTO = StimProtocol.PRC_SYN_EXC_DIST

    def __init__(self, **kwargs):
        """
        Initialize all protocol variables for given model type

        @post                       following attributes will be available on this object:
                                    - ephys_protocol: PhysioProtocol instance
                                    - proto_vars: dict with protocol variables
                                    - response_interval: expected time interval of response
        """

        sim_dur = 2000.0
        stim_start = 200
        stim_stop = sim_dur - 50.0

        self.response_interval = (stim_start, sim_dur)

        loc_soma_center = ephys.locations.NrnSeclistCompLocation(
                name            = 'soma_center_loc',
                seclist_name    = 'somatic',
                sec_index       = 0,
                comp_x          = 0.5)

        # Synapse parameters ---------------------------------------------------
        
        mech_syn1 = ephys.mechanisms.NrnMODPointProcessMechanism(
                        name='expsyn',
                        suffix='ExpSyn',
                        locations=[loc_soma_center])

        loc_syn1 = ephys.locations.NrnPointProcessLocation(
                        'expsyn_loc',
                        pprocess_mech=mech_syn1)

        param_syn1_tau = ephys.parameters.NrnPointProcessParameter(
                        name='expsyn_tau',
                        param_name='tau',
                        value=2,
                        frozen=True,
                        bounds=[0, 50],
                        locations=[loc_syn1])

        # TODO: set weight dynamically? Or calibrate once based on passive Ztransfer
        # NOTE: mechs and params passed to cellmodel in our code

        rate_syn1 = 50.0
        
        stim_syn_prox = ephys.stimuli.NrnNetStimStimulus(
                total_duration  = stim_stop - stim_start,
                number          = 1e9,
                interval        = 1e3 / rate_syn1,
                start           = stim_start,
                weight          = 5e-4,
                locations       = [loc_syn1])

        rec_stim_syn1 = bpop_recordings.NetStimRecording(
                        name='{}.syn1.spikes'.format(self.IMPL_PROTO.name),
                        netstim=stim_syn_prox)


        rec_soma_v = ephys.recordings.CompRecording(
                        name            = '{}.soma.v'.format(self.IMPL_PROTO.name),
                        location        = loc_soma_center,
                        variable        = 'v')

        self.ephys_protocol = PhysioProtocol(
                        name        = self.IMPL_PROTO.name, 
                        stimuli     = [stim_syn_prox],
                        recordings  = [rec_soma_v, rec_stim_syn1],
                        init_func   = init_stn_physiology)

        self.proto_vars = {
            'pp_mechs'      : [mech_syn1],
            'pp_comp_locs'  : [loc_syn1],
            'pp_target_locs': [],
            'pp_mech_params': [param_syn1_tau],
            'stims'         : [stim_syn_prox],
            'recordings'    : [rec_soma_v, rec_stim_syn1],
            # 'range_mechs' : [],
        }

        # Characterizing features and parameters for protocol
        # NOTE: these are feat_params used in bpop_features_stn/make_features
        self.characterizing_feats = {
            'PRC_traditional': {
                'weight'        : TODO,
                'norm_factor'   : TODO,
                'traces'        : {
                    loc_syn1.name : rec_stim_syn1.name,
                    loc_soma_center.name : rec_soma_v.name,
                },
            },
            TODO: feats, # other features to regularize cost (spike rate, ...)
        }


# Register protocols implemented here
PROTOCOL_WRAPPERS.update({
    StimProtocol.PRC_SYN_EXC_DIST: PhaseResponseSynExcDist,
})