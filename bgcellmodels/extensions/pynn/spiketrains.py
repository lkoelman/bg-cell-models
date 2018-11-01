"""
Spike train generation for PyNN

Notes
-----

The 'spike_times' argument of a PyNN SpikeSourceArray accepts
- a list of Sequence or numpy array
- a function returning such a list
"""

import numpy as np
from pyNN.parameters import Sequence
from bgcellmodels.common import spikelib


def make_bursty_spike_generator(bursting_fraction, synchronous, rng,
                                T_burst, dur_burst, f_intra, f_inter,
                                f_background, duration):
    """
    Make generator function that returns bursty spike sequences.
    """
    if synchronous:
        make_bursts = spikelib.make_oscillatory_bursts
    else:
        make_bursts = spikelib.make_variable_bursts

    def spike_seq_gen(cell_indices):
        """
        Spike sequence generator

        @param  cell_indices : list(int)
                Local indices of cells (NOT index in entire population)

        @return spiketimes_for_cell : list(Sequence)
                Sequencie of spike times for each cell index
        """
        # Choose cell indices that will emit bursty spike trains
        num_bursting = int(bursting_fraction * len(cell_indices))
        bursting_cells = rng.choice(cell_indices, 
                                    num_bursting, replace=False)

        spiketimes_for_index = []
        for i in cell_indices:
            if i in bursting_cells:
                # Spiketimes for bursting cells
                burst_gen = make_bursts(T_burst, dur_burst, f_intra, f_inter,
                                        rng=rng, max_dur=duration)
                spiketimes = Sequence(np.fromiter(burst_gen, float))
            else:
                # Spiketimes for background activity
                number = int(2 * duration * f_background / 1e3)
                if number == 0:
                    spiketimes = Sequence([])
                else:
                    spiketimes = Sequence(np.add.accumulate(
                        rng.exponential(1e3/f_background, size=number)))
            spiketimes_for_index.append(spiketimes)
        return spiketimes_for_index

    return spike_seq_gen


def bursty_spiketrains_during(intervals, bursting_fraction, 
                          T_burst, dur_burst, f_intra, f_inter, f_background, 
                          duration, randomize_bursting, rng):
    """
    Make spiketrains where a given fraction fires synchronized bursts during
    each interval.

    @param      intervals : iterable(tuple[float, float])
                Sequence of time intervals in which cells should burst.

    @return     function(iterable(index)) -> iterable(Sequence)
                Function that returns a sequence of spike times for each
                cell index.
    """

    def spike_seq_gen(cell_indices):
        """
        Spike sequence generator
        """
        # First pick bursting cells during each bursty interval
        num_bursting = int(bursting_fraction * len(cell_indices))
        num_intervals = len(intervals)
        if randomize_bursting:
            # pick new bursting cells in each interval
            bursting_ids = [rng.choice(cell_indices, num_bursting, replace=False) 
                                for i in range(num_intervals)]
        else:
            bursting_ids = [rng.choice(cell_indices, num_bursting, replace=False)] * num_intervals

        spiketimes_for_index = []
        for i in cell_indices:
            # Get intervals where this cell is bursting
            burst_intervals = [intervals[j] for j in range(num_intervals) if i in bursting_ids[j]]
            
            if len(burst_intervals) > 0:
                # Spiketimes for bursting cells
                spikegen = spikelib.generate_bursts_during(burst_intervals, 
                                T_burst, dur_burst, f_intra, f_inter, 
                                f_background, duration, max_overshoot=0.25, rng=rng)
                spiketimes = Sequence(np.fromiter(spikegen, float))
            else:
                # Spiketimes for background activity
                number = int(2 * duration * f_background / 1e3)
                if number == 0:
                    spiketimes = Sequence([])
                else:
                    spiketimes = Sequence(np.add.accumulate(
                        rng.exponential(1e3/f_background, size=number)))
            spiketimes_for_index.append(spiketimes)
        return spiketimes_for_index
    return spike_seq_gen


def bursty_permuted_spiketrains(intervals, bursting_fraction, 
                          T_burst, dur_burst, f_intra, f_inter, f_background, 
                          duration, randomize_bursting, rng):
    """
    Make spiketrains that burst semi-synchronously in each cycle of a
    reference sinusoid. In each cycle only a given fraction of spiketrains has
    a burst and the indices of bursting spiketrains are permuted in each cycle.

    @param      intervals : iterable(tuple[float, float])
                Sequence of time intervals in which cells should burst.

    @return     function(iterable(index)) -> iterable(Sequence)
                Function that returns a sequence of spike times for each
                cell index.
    """
    # Make a mask that has the bursting cell indices for each period in its rows
    # i.e. element mask[i][j] is the j-th bursting cell index in period i

    # For each spiketrains, find periods that it burst and make burst.
    # Take into acoount the intervals.
    pass