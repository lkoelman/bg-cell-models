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

    def __init__(
            self,
            stim_rate=20.0,
            stim_noise=0.5,
            stim_gmax=5e-4,
            **kwargs):
        """
        Initialize all protocol variables for given model type

        @post                       following attributes will be available on this object:
                                    - ephys_protocol: PhysioProtocol instance
                                    - proto_vars: dict with protocol variables
                                    - response_interval: expected time interval of response
        """

        

        # Parameters for PRC
        cell_rate = 30.0        # (Hz)
        cell_T = 1e3 / rate     # (ms)
        prc_sampling_T = 1.0    # (ms) sampling period within ISI

        # Sample each phase X times, and randomize order
        prc_sampling_repeats = 2
        prc_delays = np.arange(0, cell_T, prc_sampling_T)
        prc_delays = np.tile(prc_delays, prc_sampling_repeats)
        prc_delays = np.random.shuffle(prc_delays)

        # Global parameters
        sim_dur = 2000.0
        stim_start = 200
        stim_stop = stim_start + (1.1 * cell_T * len(prc_delays))
        if stim_stop > 5000.0:
            raise ValueError('Long simulation time for combination of cell '
                             'firing rate, PRC sampling interval, repeats.')

        self.response_interval = (stim_start, sim_dur)

        loc_soma_center = ephys.locations.NrnSeclistCompLocation(
                name            = 'soma_center',
                seclist_name    = 'somatic',
                sec_index       = 0,
                comp_x          = 0.5)

        # Current stimulus -----------------------------------------------------
        # To get neuron firing at target rate

        stim_bias = ephys.stimuli.NrnSquarePulse(
                        step_amplitude  = 0.01, # (nA) TODO: amplitude for target rate
                        step_delay      = 0.0,
                        step_duration   = sim_dur,
                        location        = soma_center_loc,
                        total_duration  = sim_dur)

        # Synaptic stimulus ----------------------------------------------------
        
        # Synaptic mechanism
        mech_syn1 = ephys.mechanisms.NrnMODPointProcessMechanism(
                        name='syn1',
                        suffix='Exp2Syn',
                        locations=[loc_soma_center])

        loc_syn1 = ephys.locations.NrnPointProcessLocation(
                        name='loc_syn1', pprocess_mech=mech_syn1)

        param_syn1_tau1 = ephys.parameters.NrnPointProcessParameter(
                        param_name='tau1', value=1.0, frozen=True,
                        locations=[loc_syn1], name='syn_tau1')

        param_syn1_tau2 = ephys.parameters.NrnPointProcessParameter(
                        param_name='tau2', value=3.0, frozen=True,
                        locations=[loc_syn1], name='syn_tau2')

        param_syn1_erev = ephys.parameters.NrnPointProcessParameter(
                        param_name='e', value=0.0, frozen=True,
                        locations=[loc_syn1], name='syn_erev')

        # TODO: set weight dynamically? Or calibrate once based on passive Ztransfer
        # NOTE: mechs and params passed to cellmodel in our code
        
        # Stimulate using NetStim (non-adaptive, fixed rate or noisy)
        # stim_syn_prox = ephys.stimuli.NrnNetStimStimulus(
        #         total_duration  = stim_stop - stim_start,
        #         interval        = 1e3 / stim_rate,
        #         noise           = stim_noise,
        #         number          = 1e9,
        #         start           = stim_start,
        #         weight          = stim_gmax,
        #         locations       = [loc_syn1])

        # Stimulate using NetVarDelay (adaptive, feeds back spikes with delay)
        stim_syn_prox = bpop_stimuli.NetVarDelayStimulus(
                        delays=prc_delays,
                        start_time=stim_start,
                        target_locations=[loc_syn1],
                        source_location=loc_soma_center,
                        source_threshold=-20.0,
                        source_delay=0.1,
                        target_delay=0.1,
                        target_weight=stim_gmax,
                        total_duration=None)

        rec_stim_syn1 = bpop_recordings.NetStimRecording(
                        name='PRC.stim_times', # name required by feature calculator
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
                'weight'        : 1.0, # TODO: PRC weight
                'norm_factor'   : 1.0, # TODO: PRC norm factor
                'traces'        : {
                    '' : rec_soma_v.name,
                    loc_syn1.name : rec_stim_syn1.name,
                },
            },
            # TODO: other features to regularize cost (spike rate, ...)
        }


# Register protocols implemented here
PROTOCOL_WRAPPERS.update({
    StimProtocol.PRC_SYN_EXC_PROX: PhaseResponseSynExcDist,
})