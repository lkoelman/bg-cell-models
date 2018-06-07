"""
Analysis and generation of spike trains.

"""

import numpy as np

def make_oscillatory_bursts(
    T_burst, dur_burst, f_intra, f_inter, 
    max_dur=10e3, rng=None):
    """
    Make oscillatory bursting spike times with given burst period, 
    burst duration, and intra- and inter-burst firing rates.

    The inter-burst interval (IBI) and burst duration are fixed so bursts
    will occur regularly with fixed intervals.

    @param      T_burst : float
                Burst period (ms)

    @param      dur_burst : float
                Burst duration (ms)

    @param      f_intra : float
                Intra-burst firing rate (Hz)

    @param      f_inter : float
                Inter-burst firing rate (Hz)

    @param      rng : numpy.Random
                Random object (optional)

    @return     Generator object that generates spike times


    EXAMPLES
    --------

    >>> import numpy
    >>> burst_gen = make_oscillatory_bursts(3500.0, 545.0, 50.0, 5.0)
    >>> bursty_spikes = numpy.fromiter(burst_gen, float)


    EXPERIMENTAL DATA
    -----------------

    Li 2012 - "Therapeutic DBS...":

        - T_burst = 3500 ms (f_burst = 0.286 Hz)
        - dur_burst = 545 ms
        - f_intra = 50 Hz
        - f_inter = 5 hz
    """
    if rng is None:
        rng = np.random
    # taken from example specific_network.py
    # Blend ISIs from two negexp distributions centered at intra- and
    # inter-burst firing rate, respectively
    T_intra, T_inter = 1e3/f_intra, 1e3/f_inter
    intra_ISIs = rng.exponential(T_intra, size=max(10, int(5*max_dur/T_intra)))
    inter_ISIs = rng.exponential(T_inter, size=max(10, int(5*max_dur/T_inter)))
    t = 0.0
    i_inter, i_intra = 0, 0
    while t < max_dur:
        j = t // T_burst # we are in the j-th cycle
        if j*T_burst <= t < (j*T_burst) + dur_burst:
            t += intra_ISIs[i_intra]
            i_intra += 1
            yield t
        else:
            t += inter_ISIs[i_inter]
            i_inter += 1
            yield t


def make_variable_bursts(
    T_burst, dur_burst, f_intra, f_inter, 
    max_dur=10e3, rng=None):
    """
    Make bursting spike times with given burst period, 
    burst duration, and intra- and inter-burst firing rates.

    The inter-burst interval (IBI) and burs duration are variable and drawn
    from a negative exponential distribution so the burst rate is 
    Poisson-distributed.

    @param      T_burst : float
                Burst period (ms), mean of negexp distribution

    @param      dur_burst : float
                Burst duration (ms), mean of negexp distribution

    @param      f_intra : float
                Intra-burst firing rate (Hz)

    @param      f_inter : float
                Inter-burst firing rate (Hz)

    @param      rng : numpy.Random
                Random object (optional)

    @return     Generator object that generates spike times


    EXAMPLES
    --------

    >>> import numpy
    >>> burst_gen = make_variable_bursts(3500.0, 545.0, 50.0, 5.0)
    >>> bursty_spikes = numpy.fromiter(burst_gen, float)

    """
    if rng is None:
        rng = np.random
    # Blend ISIs from two negexp distributions centered at intra- and
    # inter-burst firing rate, respectively
    T_intra, T_inter = 1e3/f_intra, 1e3/f_inter
    
    # Pre-sample with safety margin of 5 (times faster than mean)
    max_num_burst = int(5*max_dur/T_burst)
    burst_IBIs = rng.exponential(T_burst, size=max(10, max_num_burst))
    burst_durs = rng.exponential(dur_burst, size=max(10, max_num_burst))
    max_num_intra = int(5 * np.sum(burst_durs) / T_intra)
    max_num_inter = int(5 * np.sum(burst_IBIs) / T_inter)
    intra_ISIs = rng.exponential(T_intra, size=max(10, max_num_intra))
    inter_ISIs = rng.exponential(T_inter, size=max(10, max_num_inter))
    
    t_start_burst = burst_IBIs[0]
    t_end_burst = t_start_burst + burst_durs[0]

    # Indices into samples of interval distributions
    i_IBI, i_dur = 1, 1
    i_inter, i_intra = 0, 0

    in_burst = False
    t = 0.0    
    while t < max_dur:
        if t < t_start_burst:
            if in_burst and t < t_end_burst:
                t += intra_ISIs[i_intra]
                i_intra += 1
                yield t
            elif in_burst and t >= t_end_burst:
                in_burst = False
                i_dur += 1
                t_end_burst = t_start_burst + burst_durs[i_IBI]
                continue
            else:
                t += inter_ISIs[i_inter]
                i_inter += 1
                yield t
        elif t >= t_start_burst:
            # only reached once per burst
            in_burst = True
            i_IBI += 1
            t_start_burst += burst_IBIs[i_IBI]
            continue
            
            