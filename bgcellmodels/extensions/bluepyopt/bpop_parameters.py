"""
Extensions to Ephys.parameters module

@author Lucas Koelman

@date   8/05/2018
"""

import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger('bpop_ext')


class NrnSegmentParameter(ephys.parameters.NrnParameter, ephys.serializer.DictMixin):
    """
    Parameter of a section
    """
    SERIALIZED_FIELDS = ('name', 'value', 'frozen', 'locations', )

    def __init__(
            self,
            name,
            value=None,
            frozen=False,
            param_name=None,
            locations=None):
        """
        Contructor
        Args:
            name (str): name of the Parameter
            value (float): Value for the parameter, required if Frozen=True
            frozen (bool): Whether the parameter can be varied, or its values
            is permently set
            param_name (str): name used within NEURON
            locations (list of ephys.locations.Location): locations on which
                to instantiate the parameter
        """

        super(NrnSegmentParameter, self).__init__(
            name,
            value=value,
            frozen=frozen)

        self.locations = locations
        self.param_name = param_name


    def instantiate(self, sim=None, icell=None):
        """
        Instantiate
        """
        if self.value is None:
            raise Exception(
                'NrnSegmentParameter: impossible to instantiate parameter "%s" '
                'without value' % self.name)

        for location in self.locations:
            isegments = location.instantiate(sim=sim, icell=icell)
            for segment in isegments:
                setattr(segment, self.param_name, self.value)
            logger.debug(
                'Set %s in %s to %s',
                self.param_name,
                location,
                self.value)

    def __str__(self):
        """String representation"""
        return '%s: %s %s = %s' % (self.name,
                                   [str(location)
                                    for location in self.locations],
                                   self.param_name,
                                   self.value if self.frozen else self.bounds)

class NrnScaleRangeParameter(ephys.parameters.NrnParameter, ephys.serializer.DictMixin):
    """
    Parameter that scales a NEURON RANGE parameter 
    (pre-existing spatial distribution) in target region.
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
            segment_filter=None):
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
        self.segment_filter = segment_filter


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
                    # Skip segment if doesn't match filter
                    if (self.segment_filter is not None) and (not self.segment_filter(seg)):
                        continue
                    # Scale the parameter
                    old_val = getattr(seg, '%s' % self.param_name)
                    new_val = old_val * self.value
                    setattr(seg, '%s' % self.param_name, new_val)
        
        logger.debug(
                'Scaled %s in %s by factor %s', self.param_name,
                [str(location) for location in self.locations],
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
    Parameter that offsets a NEURON RANGE parameter
    (pre-existing spatial distribution) in target region.
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
