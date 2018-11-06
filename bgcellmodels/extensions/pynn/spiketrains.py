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
    Make generator for continuous regularly bursting spike trains.
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
    Generator for a given fraction of spiketrains firing synchronized bursts
    during given time intervals.

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


def bursty_permuted_spiketrains(
        T_burst         = 20.0,
        phi_burst       = 0.0,
        num_spk_burst   = 4,
        f_intra         = 180.0,
        f_background    = 5.0,
        max_dt_spk      = 1.0,
        t_refrac_pre    = 5.0,
        t_refrac_post   = 10.0,
        bursting_fraction = 0.25,
        intervals       = None,
        duration        = 10e3,
        rng             = None):
    """
    Make spiketrains that burst semi-synchronously in each cycle of a
    reference sinusoid. In each cycle only a given fraction of spiketrains has
    a burst and the indices of bursting spiketrains are permuted in each cycle.

    Arguments
    ---------

    @param      T_burst : float
                Burst period (ms)

    @param      phi_burst : float
                Phase angle (degrees) where burst starts

    @param      num_spk_burst : tuple[int, int]
                Number of spikes per burst, lower and upper bound

    @param      f_intra : float
                Intra-burst firing rate (Hz)

    @param      f_background : float
                Firing rate when cell is not bursting.

    @param      rng : numpy.Random
                Random generator (optional)

    @param      intervals : iterable(tuple[float, float])
                Sequence of time intervals in which cells should burst.

    @param      max_dt_spk : float
                Max dt (ms) added to spiketimes inside a burst. The interval
                between spikes in a burst is 1/f_intra + uniformly sampled
                value in (0, max_dt_spk). I.e. f_intra is the max firing rate.

    @return     function(iterable(index)) -> iterable(Sequence)
                Function that returns a sequence of spike times for each
                cell index.
    """
    # Make a mask that has the bursting cell indices for each cycle in its rows
    # i.e. element mask[i][j] is the j-th bursting cell index in cycle i
    def spike_seq_gen(cell_indices):
        """
        Spike sequence generator
        """
        # First pick bursting cells during each bursty interval
        num_bursting = int(bursting_fraction * len(cell_indices))
        num_cycles = int(duration / T_burst) + 1
        # element i are cell ids that burst during cycle i
        cycle_burst_ids = np.array(
            [rng.choice(cell_indices, num_bursting, replace=False) 
                for i in range(num_cycles)])

        T_intra = 1e3 / f_intra
        spk_burst_centered = np.arange(0, num_spk_burst[1]*T_intra, T_intra)

        # For each spiketrains
        # - Generate bursty spikes in cycles that it's active and overlap with interval
        # - Generate background spikes in remaining cycles.
        spiketimes_for_index = []
        for cell_idx in cell_indices:
            # Spiketimes for background activity
            number = int(2 * duration * f_background / 1e3)
            spk_bg = np.add.accumulate(
                rng.exponential(1e3/f_background, size=number))
            bg_del_mask = spk_bg > duration # np.zeros_like(spk_bg, dtype=bool)

            # Generate background spikes until t > next active period
            bursting_cycles = np.where(cycle_burst_ids == cell_idx)[0]
            if len(bursting_cycles) == 0:
                spiketimes_for_index.append(Sequence(spk_bg))
                continue

            # Add bursty spikes in burst cycles
            all_spk = []
            for i_cycle in bursting_cycles:
                t0_cycle = i_cycle * T_burst
                t1_cycle = t0_cycle + T_burst
                t0_burst = t0_cycle + (phi_burst / 360.0 * T_burst)
                if intervals is not None:
                    # only add burst in cycle if cycle falls in bursty interval
                    if not any((((ival[0] <= t0_cycle) and (t1_cycle <= ival[1])) for ival in intervals)):
                        continue
                num_spk = rng.randint(num_spk_burst[0], num_spk_burst[1]+1)
                spk_var_dt = rng.uniform(0.0, max_dt_spk, num_spk)
                spk_burst = t0_burst + spk_burst_centered[0:num_spk] + spk_var_dt
                all_spk.append(spk_burst)

                # Delete background spikes in tspk0-refrac, tspk[-1]+refrac
                # - build a deletion mask and apply at end
                cycle_mask = ((spk_bg >= (spk_burst[0] - t_refrac_pre)) & 
                              (spk_bg <= (spk_burst[-1] + t_refrac_post)))
                bg_del_mask = bg_del_mask | cycle_mask

            # Delete background spikes that fall in refractory period around burst
            all_spk.append(spk_bg[~bg_del_mask])

            # Sort the spikes
            spk_merged = np.concatenate(all_spk)
            spk_merged.sort()

            spiketimes = Sequence(spk_merged)
            spiketimes_for_index.append(spiketimes)
        return spiketimes_for_index
    return spike_seq_gen


def test_spiketime_generator(gen_func, num_spiketrains, *args, **kwargs):
    """
    Test case for generate_modulated_spiketimes()
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    generator = gen_func(*args, **kwargs)
    cells_spiketimes = generator(np.arange(num_spiketrains))

    for i, sequence in enumerate(cells_spiketimes):
        spiketimes = sequence.value
        y_vec = np.ones_like(spiketimes) * i
        ax.plot(spiketimes, y_vec, marker='|', linestyle='', color='red', snap=True)

    ax.set_title("Result of '{}'".format(gen_func.__name__))
    ax.grid(True)
    plt.show(block=False)


if __name__ == '__main__':
    T_burst, dur_burst, f_intra, f_background = 50.0, 10.0, 180.0, 5.0
    rng = np.random
    duration = 2e3
    intervals = [(500.0, 20e3)]
    num_spiketrains = 20
    num_spk_burst = (3, 5)

    # Test for 'bursty_permuted_spiketrains'
    max_dt_spk = 1.0
    refrac = 20.0, 20.0
    frac_bursting = 0.2
    test_spiketime_generator(
        bursty_permuted_spiketrains, num_spiketrains, 
        # args and kwargs below:
        T_burst, num_spk_burst, f_intra, f_background, max_dt_spk,
        refrac[0], refrac[1], frac_bursting, intervals, duration, np.random)
