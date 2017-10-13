"""
Custom EFeatures to compare spike trains using different similarity mesures / 
distance metrics.

@author Lucas Koelman

@date   4/10/2017


ARCHITECTURE:

   - make custom =EFeature= class that allows to set a target (in the form of spike train or histogram, e.g. a neo spiketrain)

       - this can use eFEL to get the spike times using =efel.getFeatureValues('peak_time')=

       - then call *elephant* or other library to compute distance with target spike train

       - as long as this offers =calculate_score(responses)=, you don't need a custom Objective class

   
   - *PROS* flexible, easy to integrate and easily call into any module

"""

import bluepyopt.ephys as ephys

# Elephant library
from elephant import spike_train_dissimilarity as stds
import quantities as pq
import neo

import logging
logger = logging.getLogger('bpop_ext')


def getFeatureNames():
    """
    Return list with names of available features.
    """
    return list(SpikeTrainFeature.DISTANCE_METRICS)


class SpikeTrainFeature(ephys.efeatures.EFeature, ephys.serializer.DictMixin):

    """
    Feature representing spike times in a particular interval. The distance is
    computed usinng a given similarity metric.
    """

    # TODO: for serialization, check that only raw spike times/start/end are serialized, and spike train is rebuit after serialization.
    SERIALIZED_FIELDS = ('name', 'metric_name', 'recording_names',
                         'stim_start', 'stim_end', 'target_spike_times',
                         'threshold', 'comment')

    # List of available distance metrics
    DISTANCE_METRICS = ('victor_purpura_distance',)

    def __init__(
            self,
            name,
            metric_name=None,
            recording_names=None,
            stim_start=None,
            stim_end=None,
            target_spike_times=None,
            threshold=None,
            interp_step=None,
            comment='',
            double_settings=None,
            int_settings=None,
            force_max_score=False,
            max_score=250
    ):
        """
        Constructor

        Args:

            name (str):             name of the SpikeTrainFeature object
            
            metric_name (str):      name of similarity measure / distance metric to use
            
            recording_names (dict): eFEL features can accept several recordings
                                    as input
            
            stim_start (float):     stimulation start time (ms)
            
            stim_end (float):       stimulation end time (ms)
            
            threshold(float):       spike detection threshold (mV)
            
            comment (str):          comment
            
            interp_step(float):     interpolation step (ms)
            
            target_spike_train:     list(float) or numpy.array with spike times.
        """

        super(SpikeTrainFeature, self).__init__(name, comment)

        if metric_name.lower() not in self.DISTANCE_METRICS:
            raise NotImplementedError("Metric {} not implemented".format(metric_name))

        if double_settings is None:
            double_settings = {}
        if int_settings is None:
            int_settings = {}

        self.recording_names = recording_names
        self.metric_name = metric_name

        self.stim_start = stim_start
        self.stim_end = stim_end
        
        self.threshold = threshold
        self.interp_step = interp_step
        self.double_settings = double_settings
        self.int_settings = int_settings
        
        self.force_max_score = force_max_score
        self.max_score = max_score
        self._target_spike_times = target_spike_times # also builds spike train
        self.exp_std = 1.0
        self.exp_mean = None


    def _construct_neo_spiketrain(self, spike_times):
        """
        Construct Neo spike train for use with Elephant library.
        """
        if len(spike_times)>0 and (spike_times[0] < self.stim_start or spike_times[-1] > self.stim_end):
            spike_times = [t for t in spike_times if 
                                        (t >= self.stim_start and t <= self.stim_end)]
        
        spike_train = neo.SpikeTrain(
                            spike_times,
                            units ='ms',
                            t_start = self.stim_start,
                            t_stop = self.stim_end)

        return spike_train


    def get_target_spike_times(self):
        """
        Return target spike times as list.
        """
        return self._target_spike_times


    def set_target_spike_times(self, value):
        """
        Set target spike times.
        """
        self._target_spike_times = value
        self.target_spike_train = self._construct_neo_spiketrain(value)

    # Make property explicitly
    target_spike_times = property(get_target_spike_times, set_target_spike_times)


    def _construct_efel_trace(self, responses):
        """
        Construct trace that can be passed to eFEL
        """

        trace = {}
        if '' not in self.recording_names:
            raise Exception(
                'SpikeTrainFeature: \'\' needs to be in recording_names')
        
        for location_name, recording_name in self.recording_names.items():
            if location_name == '':
                postfix = ''
            else:
                postfix = ';%s' % location_name

            if recording_name not in responses:
                logger.debug(
                    "Recording named %s not found in responses %s",
                    recording_name,
                    str(responses))
                return None

            if responses[self.recording_names['']] is None or \
                    responses[recording_name] is None:
                return None
            trace['T%s' % postfix] = \
                responses[self.recording_names['']]['time']
            trace['V%s' % postfix] = responses[recording_name]['voltage']
            trace['stim_start%s' % postfix] = [self.stim_start]
            trace['stim_end%s' % postfix] = [self.stim_end]

        return trace

    def _setup_efel(self):
        """
        Set up efel before extracting the feature
        """

        import efel
        efel.reset()

        if self.threshold is not None:
            efel.setThreshold(self.threshold)

        if self.interp_step is not None:
            efel.setDoubleSetting('interp_step', self.interp_step)


    def calculate_feature(self, responses, raise_warnings=True):
        """
        Calculate feature value: for this EFeature it is just the spike train
        as a sequence of spike times.

        """

        efel_trace = self._construct_efel_trace(responses)

        if efel_trace is None:
            feature_value = None
        else:

            self._setup_efel()
            import efel

            # Calculate spike times from response
            values = efel.getFeatureValues(
                [efel_trace],
                ['peak_time'],
                raise_warnings=raise_warnings
            )
            feature_value = values[0]['peak_time']

            efel.reset()

        logger.debug(
            'Calculated value for %s: %s',
            self.name,
            str(feature_value))

        return feature_value


    def calculate_score(self, responses, trace_check=False):
        """
        Calculate score: spike train distance between given response and target
        spike train.
        """

        efel_trace = self._construct_efel_trace(responses)

        if efel_trace is None:
            score = self.max_score
        else:
            
            self._setup_efel()
            import efel

            # Calculate spike times from response
            feat_vals = efel.getFeatureValues(
                [efel_trace],
                ['peak_time'],
                raise_warnings = True
            )
            resp_spike_times = feat_vals[0]['peak_time']
            resp_spike_train = self._construct_neo_spiketrain(resp_spike_times)


            # Compute spike train dissimilarity
            # TODO: read up and decide on q factor
            if self.metric_name.lower() == 'victor_purpura_distance':

                spike_shift_cost = self.double_settings.get('spike_shift_cost', 1.0)
                
                dist_mat = stds.victor_purpura_dist(
                                [self.target_spike_train, resp_spike_train], 
                                q = spike_shift_cost * pq.Hz, 
                                algorithm = 'fast')

                score = dist_mat[0, 1]
            else:
                raise NotImplementedError("Metric {} not implemented".format(self.metric_name))

            # Need to take into account std (can be used as weight too)
            score /= self.exp_std
            
            if self.force_max_score:
                score = max(score, self.max_score)

        logger.debug('Calculated score for %s: %f', self.name, score)

        return score


    def __str__(self):
        """
        String representation
        """

        return "%s for %s with stim start %s and end %s, " \
            "target_spike_times and AP threshold override %s" % \
            (self.metric_name,
             self.recording_names,
             self.stim_start,
             self.stim_end,
             self.target_spike_train,
             self.threshold)