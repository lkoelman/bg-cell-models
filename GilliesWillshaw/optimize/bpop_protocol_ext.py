"""
Extension of BluePyOpt protocol that allow instantiating stimuli, synapses, etc.
without specifying them in the form of ephys mechanisms & parameters (using bpop.ephys module).

@author	Lucas Koelman

@date	7/10/2017

"""

import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger('bpop_ext')


class NrnExtProtocol(ephys.protocols.SweepProtocol):
    """
    Protocol consisting of current clamps, voltage clamps,
    and changes to physiological conditions.

    NOTE: call graph for SweepProtocol is as follows:

        protocol.run(cell_model, param_values, sim) -> _run_func(...) :
            model.instantiate()
            protocol.instantiate() :
                stimulus.instantiate() :
                    location.instantiate()
                recording.instantiate()
            sim.run()
    """

    def __init__(
            self,
            name=None,
            stimuli=None,
            recordings=None,
            cvode_active=None,
            init_func=None):
        """
        Constructor
        
        Args:
            init_func:  function(sim, model) that takes Simulator and instantiated 
                        CellModel (icell) as arguments in that order
        """

        self._init_func = init_func

        super(NrnExtProtocol, self).__init__(
            name,
            stimuli=stimuli,
            recordings=recordings,
            cvode_active=cvode_active)


    def instantiate(self, sim=None, icell=None):
        """
        Instantiate

        NOTE: operations in StnModelEvaluator:

	        self.make_inputs(stim_proto)
			self.rec_traces(stim_proto, recordStep=0.05)
			self.init_sim(stim_proto)
			self.run_sim(stim_proto)

        """
        # TODO: make inputs and store

        # TODO: make custom recordings and store

        # Apply physiological conditions
        self._init_func(sim, icell)

        # Finally instantiate ephys.stimuli and ephys.recordings
        super(NrnExtProtocol, self).instantiate(
            sim=sim,
            icell=icell)


    def destroy(self, sim=None):
        """
        Destroy protocol
        """

        # Make sure stimuli are not active in next protocol if cell model reused
        # NOTE: should better be done in Stimulus objects themselves for encapsulation, but BluePyOpt built-in Stimuli don't do this
        for stim in self.stimuli:
            if hasattr(stim, 'iclamp'):
                stim.iclamp.amp = 0
                stim.iclamp.dur = 0
            elif hasattr(stim, 'seclamp'):
                for i in range(3):
                    setattr(stim.seclamp, 'amp%d' % (i+1), 0)
                    setattr(stim.seclamp, 'dur%d' % (i+1), 0)

        # Calls destroy() on each stimulus
        super(NrnExtProtocol, self).destroy(sim=sim)