"""
Extensions to BluePyOpt optimization-related classes

@author Lucas Koelman

@date   13/09/2017


"""


import bluepyopt.ephys as ephys

import logging
logger = logging.getLogger(__name__)


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

        super(NrnOffsetRangeParameter, self).__init__(
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
