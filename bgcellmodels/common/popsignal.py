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

@numba.jit("f8[:](f8[:],f8[:],f8,f8)", nopython=True)
def find_pulse_locked_spikes(pulse_times, spike_times, window_lo, window_hi):
    """
    Find pulse-locked spikes in time window after each pulse.

    @param  window_lo : float
            Minimum time after pulse (ms) to look for phase-locked spike

    @param  window_hi : float
            Maximum time after pulse (ms) to look for phase-locked spike
    """
    locked_spike_times = []
    for t_pulse in pulse_times:
        mask_following = (
            ((t_pulse + window_lo) < spike_times) & 
            (spike_times <= (t_pulse + window_hi))
        )
        spikes_following = spike_times[mask_following]
        if spikes_following.size > 0:
            locked_spike_times.append(spikes_following[0])
    
    return np.array(locked_spike_times)