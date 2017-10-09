"""
Extensions to BluePyOpt optimization-related classes

@author Lucas Koelman

@date   13/09/2017


"""


import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger('bpop_ext')


class NrnScaleRangeParameter(ephys.parameters.NrnParameter, ephys.serializer.DictMixin):
    """
    Parameter that scales a NEURON RANGE parameter (pre-existing spatial distribution) in target region.
    """

    SERIALIZED_FIELDS = ('name', 'value', 'frozen', 'bounds', 'param_name',
                         'value_scaler', 'locations', )

    def __init__(
            self,
            name,
            value=None,
            frozen=False,
            bounds=None,
            param_name=None,
            locations=None):
        """
        Contructor

        Args:
            name (str): name of the Parameter
            
            value (float): Value for the parameter, required if Frozen=True
            
            frozen (bool): Whether the parameter can be varied, or its values
            is permently set
            
            bounds (indexable): two elements; the lower and upper bounds
                (Optional)
            
            param_name (str): name used within NEURON
            
            locations (list of ephys.locations.Location): locations on which
                to instantiate the parameter
        """

        super(NrnScaleRangeParameter, self).__init__(
            name,
            value=value,
            frozen=frozen,
            bounds=bounds)

        self.locations = locations
        self.param_name = param_name


    def instantiate(self, sim=None, icell=None):
        """
        Instantiate (i.e. apply the parameter)
        """

        if self.value is None:
            raise Exception(
                'NrnRangeParameter: impossible to instantiate parameter "%s" '
                'without value' % self.name)

        for location in self.locations:
            for isection in location.instantiate(sim=sim, icell=icell):
                for seg in isection:
                    # Scale the parameter
                    old_val = getattr(seg, '%s' % self.param_name)
                    new_val = old_val * self.value
                    setattr(seg, '%s' % self.param_name, new_val)
        
        logger.debug(
            'Scaled %s in %s by factor %s', self.param_name,
            [str(location)
             for location in self.locations],
            self.value)


    def __str__(self):
        """String representation"""
        return '%s: %s %s = %s' % (self.name,
                                   [str(location)
                                    for location in self.locations],
                                   self.param_name,
                                   self.value if self.frozen else self.bounds)


class NrnOffsetRangeParameter(ephys.parameters.NrnParameter, ephys.serializer.DictMixin):
    """
    Parameter that offsets a NEURON RANGE parameter (pre-existing spatial distribution) in target region.
    """

    SERIALIZED_FIELDS = ('name', 'value', 'frozen', 'bounds', 'param_name',
                         'value_scaler', 'locations', )

    def __init__(
            self,
            name,
            value=None,
            frozen=False,
            bounds=None,
            param_name=None,
            locations=None,
            threshold=None):
        """
        Contructor

        Args:
            name (str):         name of the Parameter
            
            value (float):      Value for the parameter, required if Frozen=True
            
            frozen (bool):      Whether the parameter can be varied, or its values
            is permently set
            
            bounds (indexable): two elements; the lower and upper bounds
                                (Optional)
            
            param_name (str):   name used within NEURON
            
            locations (list of ephys.locations.Location):
                                locations on which to instantiate the parameter

            threshold:          threshold on parameter value: only offset if parameter
                                is larger than this value.
        """

        super(NrnOffsetRangeParameter, self).__init__(
            name,
            value=value,
            frozen=frozen,
            bounds=bounds)

        self.locations = locations
        self.param_name = param_name
        self.threshold = threshold


    def instantiate(self, sim=None, icell=None):
        """
        Instantiate (i.e. apply the parameter)
        """

        if self.value is None:
            raise Exception(
                'NrnRangeParameter: impossible to instantiate parameter "%s" '
                'without value' % self.name)

        for location in self.locations:
            for isection in location.instantiate(sim=sim, icell=icell):
                for seg in isection:
                    # Scale the parameter
                    old_val = getattr(seg, '%s' % self.param_name)
                    if old_val > self.threshold:
                        new_val = old_val + self.value
                        setattr(seg, '%s' % self.param_name, new_val)
        
        logger.debug(
            'Offset %s in %s by value %s', self.param_name,
            [str(location)
             for location in self.locations],
            self.value)


    def __str__(self):
        """String representation"""
        return '%s: %s %s = %s' % (self.name,
                                   [str(location)
                                    for location in self.locations],
                                   self.param_name,
                                   self.value if self.frozen else self.bounds)


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


class NrnNamedSecLocation(ephys.locations.Location, ephys.serializer.DictMixin):
    """
    Location in a specific section, identified by name.
    """

    SERIALIZED_FIELDS = (
        'name',
        'comment',
        'seclist_name',
        'sec_index',
        'comp_x',
    )

    def __init__(
            self,
            name,
            sec_name=None,
            comp_x=None,
            comment=''):
        """
        Constructor
        
        Args:
            name (str): name of the object
            seclist_name (str): name of Neuron section list (ex: 'somatic')
            sec_index (int): index of the section in the section list
            comp_x (float): segx (0..1) of segment inside section
        """

        super(NrnNamedSecLocation, self).__init__(name, comment)
        self.sec_name = sec_name
        self.comp_x = comp_x

    def instantiate(self, sim=None, icell=None):  # pylint: disable=W0613
        """
        Find the instantiate compartment
        """
        iseclist = icell.all
        
        try:
            isection = next((sec for sec in iseclist if sec.name()==self.sec_name))
        except StopIteration:
            raise ValueError("Section with name {} not found on this cell".format(self.sec_name))
        icomp = isection(self.comp_x)
        return icomp

    def __str__(self):
        """String representation"""
        return '%s(%s)' % (self.sec_name, self.comp_x)