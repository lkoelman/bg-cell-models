"""
Extensions to bluepyopt.ephys.stimuli
"""

import bluepyopt.ephys as ephys

from bgcellmodels.mechanisms import stimuli # loads VecStim.mod

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


class NrnVecStimStimulus(ephys.stimuli.Stimulus):

    """Current stimulus based on current amplitude and time series"""

    def __init__(self,
                 locations=None,
                 total_duration=None,
                 times=None,
                 weight=1):
        """Constructor
        Args:
            location: synapse point process location to connect to
            times (list[float]) : stimulus times
        """

        super(NrnVecStimStimulus, self).__init__()
        if total_duration is None:
            raise ValueError(
                'NrnNetStimStimulus: Need to specify a total duration')
        else:
            self.total_duration = total_duration

        self.locations = locations
        self.times = times
        self.times_vec = None
        self.weight = weight
        self.connections = {}

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        self.times_vec = sim.neuron.h.Vector(self.times)

        for location in self.locations:
            self.connections[location.name] = []
            for synapse in location.instantiate(sim=sim, icell=icell):
                netstim = sim.neuron.h.VecStim()
                netstim.play(self.times_vec)
                netcon = sim.neuron.h.NetCon(netstim, synapse)
                netcon.weight[0] = self.weight

                self.connections[location.name].append((netcon, netstim))

    def destroy(self, sim=None):
        """Destroy stimulus"""

        self.connections = None
        self.times = None
        self.times_vec = None

    def __str__(self):
        """String representation"""

        return "VecStim at %s" % ','.join(
            location
            for location in self.locations) \
            if self.locations is not None else "Netstim"