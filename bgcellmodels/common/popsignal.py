"""
Population signal analysis tools.

@author     Lucas Koelman
@date       17/07/2019

For use with Jupyter notebooks, making use of the following globar variables:

@global     pops_segments : dict[str, neo.Segment]
            Mapping of recorded population labels to data segments


JUPYTER EXAMPLE
---------------

>>> %load_ext autoreload
>>> %autoreload 1
>>> %aimport bgcellmodels.common.popsignal
>>> popsig = bgcellmodels.common.popsignal

"""

import numpy as np
import numba

@numba.jit("UniTuple(f8[:], 2)(f8[:],f8[:],f8,f8)", nopython=True)
def find_stimlocked_spikes(pulse_times, spike_times, window_lo, window_hi):
    """
    Find pulse-locked spikes in time window after each pulse.

    @param  window_lo : float
            Minimum time after pulse (ms) to look for phase-locked spike

    @param  window_hi : float
            Maximum time after pulse (ms) to look for phase-locked spike

    @return (indices, spike_times) : tuple(numpy.array[int], numpy.array[float])
            Indices of all pulses that have stimulus-spikes, and the spike
            times that are stimulus-locked
    """
    locked_indices = []
    locked_spike_times = []
    for i_pulse, t_pulse in enumerate(pulse_times):
        mask_following = (
            (spike_times > (t_pulse + window_lo)) & 
            (spike_times <= (t_pulse + window_hi))
        )
        spikes_following = spike_times[mask_following]
        if spikes_following.size > 0:
            locked_indices.append(i_pulse)
            locked_spike_times.append(spikes_following)
    
    return np.array(locked_indices), np.array(locked_spike_times)


@numba.jit("f8[:](f8[:],f8[:],f8,f8)", nopython=True)
def find_stimlocked_indices(pulse_times, spike_times, window_lo, window_hi):
    """
    Find pulse-locked spikes in time window after each pulse.

    Set window_lo = 0.0 and window_hi = inter-pulse-interval to find all pulses
    that have stimulus-locked spikes.

    @param  window_lo : float
            Minimum time after pulse (ms) to look for phase-locked spike

    @param  window_hi : float
            Maximum time after pulse (ms) to look for phase-locked spike
    """
    locked_indices = []
    for i_pulse, t_pulse in enumerate(pulse_times):
        mask_following = (
            (spike_times > (t_pulse + window_lo)) & 
            (spike_times <= (t_pulse + window_hi))
        )
        spikes_following = spike_times[mask_following]
        if spikes_following.size > 0:
            locked_indices.append(i_pulse)
    
    return np.array(locked_indices)

