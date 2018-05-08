"""
Extensions to BluePyOpt optimization-related classes

@author Lucas Koelman

@date   13/09/2017


"""


import bluepyopt.ephys as ephys

import math
import logging
logger = logging.getLogger('bpop_ext')


class NrnSpaceClamp(ephys.stimuli.Stimulus):

    """Square pulse current clamp injection"""

    def __init__(self,
                 step_amplitudes=None,
                 step_durations=None,
                 total_duration=None,
                 location=None):
        """
        Constructor
        
        Args:
            step_amplitudes (float):    amplitude (nA)
            step_durations (float):     duration (ms)
            total_duration (float):     total duration of stimulus and its effects (ms)
            location (Location):        stimulus Location
        """

        super(NrnSpaceClamp, self).__init__()
        self.step_amplitudes = step_amplitudes
        self.step_durations = step_durations
        self.location = location
        self.total_duration = total_duration
        self.seclamp = None


    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        icomp = self.location.instantiate(sim=sim, icell=icell)
        logger.debug(
            'Adding space clamp to {} with '
            'durations {}, and amplitudes {}'.format(
            str(self.location),
            self.step_durations,
            self.step_amplitudes))

        # Make SEClamp (NEURON space clamp)
        self.seclamp = sim.neuron.h.SEClamp(icomp.x, sec=icomp.sec)
        for i in range(3):
            setattr(self.seclamp, 'amp%d' % (i+1), self.step_amplitudes[i])
            setattr(self.seclamp, 'dur%d' % (i+1), self.step_durations[i])


    def destroy(self, sim=None):
        """Destroy stimulus"""

        self.seclamp = None


    def __str__(self):
        """String representation"""

        return "Square pulse amps {} durations {} totdur {} at {}".format(
            self.step_amplitudes,
            self.step_durations,
            self.total_duration,
            self.location)


class PhysioProtocol(ephys.protocols.SweepProtocol):
    """
    Protocol consisting of current clamps, voltage clamps,
    and changes to physiological conditions.
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

        super(PhysioProtocol, self).__init__(
            name,
            stimuli=stimuli,
            recordings=recordings,
            cvode_active=cvode_active)


    def instantiate(self, sim=None, icell=None):
        """
        Instantiate

        NOTE: called in self._run_func() after model.instantiate()
        """
        # First apply physiological conditions
        self._init_func(sim, icell)

        # Then instantiate stimuli and recordings
        super(PhysioProtocol, self).instantiate(
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
        super(PhysioProtocol, self).destroy(sim=sim)


class SumOfSquaresObjective(ephys.objectives.EFeatureObjective):
    """
    Objective that calculates sum of squares of EFeature scores
    """

    def __init__(self, name, features):
        """
        Constructor

        Args:
            name (str): name of this object
            features (list of EFeatures): eFeatures in the objective
        """

        super(SumOfSquaresObjective, self).__init__(name, features)


    def calculate_score(self, responses):
        """
        Objective score
        """

        feature_scores = self.calculate_feature_scores(responses)

        sumsq = sum((score**2 for score in feature_scores))
        return sumsq


class RootMeanSquareObjective(ephys.objectives.EFeatureObjective):
    """
    Objective that calculates sum of squares of EFeature scores
    """

    def __init__(self, name, features):
        """
        Constructor

        Args:
            name (str): name of this object
            features (list of EFeatures): eFeatures in the objective
        """

        super(RootMeanSquareObjective, self).__init__(name, features)


    def calculate_score(self, responses):
        """
        Objective score
        """

        feature_scores = self.calculate_feature_scores(responses)

        rms = math.sqrt(
                    sum((score**2 for score in feature_scores)) / len(feature_scores))
        return rms


class CellEvaluatorCaching(ephys.evaluators.CellEvaluator):
    """
    CellEvaluator extension that can save responses after
    distributed evaluation.
    """

    def set_responses_filename(self, folder, prefix):
        """
        Set directory to save responses to and prefix
        for responses filename.
        """
        self.responses_folder = folder
        self.responses_prefix = prefix


    def evaluate_with_dicts(self, param_dict=None, ind_suffix=None):
        """Run evaluation with dict as input and output"""

        if self.fitness_calculator is None:
            raise Exception(
                'CellEvaluator: need fitness_calculator to evaluate')

        logger.debug('Evaluating %s', self.cell_model.name)

        responses = self.run_protocols(
            self.fitness_protocols.values(),
            param_dict)

        # Save responses if folder was set
        folder = getattr(self, 'responses_folder', None)
        prefix = getattr(self, 'responses_prefix', '')

        if folder is not None and ind_suffix is not None:
            import os.path, pickle
            fname = os.path.join(folder, prefix+str(ind_suffix))
            with open(fname, 'wb') as f:
                pickle.dump(responses, f)

        return self.fitness_calculator.calculate_scores(responses)


    def evaluate_with_lists(self, param_list=None, ind_suffix=None):
        """
        Run evaluation with lists as input and outputs

        @param  param_list  list of parameter values in order corresponding to
                            self.param_names
        """

        param_dict = self.param_dict(param_list)

        obj_dict = self.evaluate_with_dicts(
                            param_dict=param_dict, 
                            ind_suffix=ind_suffix)

        return self.objective_list(obj_dict)