"""
Signal analysis tools for electrophysiology data.

@author     Lucas Koelman

Notes
-----

See features in PyElectro: https://pyelectro.readthedocs.io/en/latest/pyelectro.html

See features in eFEL: https://efel.readthedocs.io/en/latest/eFeatures.html
"""

import neuron
h = neuron.h
import numpy as np
import scipy.signal


def spike_indices(vrec, thresh, loc='onset'):
    """
    Get spike indices in voltage trace.
    
    @param  vrec : np.array
            membrane voltage
    
    @param threshold : float
            spike threshold

    @param  loc : str
            'onset' : time of spike onset, i.e. where threshold is first reached
            'offset' : time of spike offset, i.e. where v dips below threshold
    """
    thresholded = np.array((vrec - thresh) >= 0, dtype=float)
    i_onset = np.where(np.diff(thresholded) == 1)[0] + 1
    i_offset = np.where(np.diff(thresholded) == -1)[0] + 1
    if loc == 'onset':
        return i_onset
    elif loc == 'offset':
        return i_offset
    else:
        raise ValueError("Unrecognized value for argument 'loc' : {}".format(loc))


def spike_times(v_rec, t_rec, v_th, loc='onset'):
    """
    Get spike times in voltage trace.
    
    @param  vrec : np.array
            membrane voltage
    
    @param threshold : float
            spike threshold

    @param  loc : str
            'onset' : time of spike onset, i.e. where threshold is first reached
            'offset' : time of spike offset, i.e. where v dips below threshold
    """
    thresholded = np.array((v_rec - v_th) >= 0, dtype=float)
    i_onset = np.where(np.diff(thresholded) == 1)[0] + 1
    i_offset = np.where(np.diff(thresholded) == -1)[0] + 1
    if loc == 'onset':
        return t_rec[i_onset]
    elif loc == 'offset':
        return t_rec[i_offset]
    else:
        raise ValueError("Unrecognized value for argument 'loc' : {}".format(loc))


def coefficient_of_variation(v_rec, t_rec, v_th):
    """
    Calculate coefficient of variation of spikes.
    This is the ratio of the standard deviation and the mean of the ISIs.
    """
    t_spikes = spike_times(v_rec, t_rec, v_th, loc='onset')
    isi_vals = np.diff(t_spikes)
    return np.std(isi_vals) / np.mean(isi_vals)


def burst_metrics(
        v_rec, t_rec, threshold=-20.0, 
        onset_isi=20.0, offset_isi=20.0, min_spk=4):
    """
    Calculate burst rate, inter-burst intervals, intra-burst rates,
    number of spikes per burst.
    """
    # Using eFEL: use peak indices
    # FIXME: indices are half the values of our own function ???
    # features = ['ISI_values', 'peak_indices']
    # feat_vals = get_efel_features(v_rec, t_rec, interval, features, threshold=threshold)
    # isi_vals = feat_vals['ISI_values']
    # peak_idx = feat_vals['peak_indices']

    # Using threshold crossing indices
    peak_idx = spike_indices(v_rec, thresh=threshold, loc='onset')
    peak_times = t_rec[peak_idx]
    isi_vals = np.diff(peak_times)

    burst_lengths = []
    burst_intra_rates = []
    onset_times = []
    offset_times = []
    inter_burst_intervals = []

    # Find bursts according to absolute criteria 
    num_isi = len(isi_vals)
    isi_burst = np.zeros_like(isi_vals, dtype=bool) # flag which ISIs are in a burst
    isi_below_offset = np.array(isi_vals <= offset_isi, dtype=int)
    candidate_offsets = np.diff(isi_below_offset) == -1
    i = 0
    while True:
        if i > num_isi-1:
            break
        isi = isi_vals[i]
        if isi <= onset_isi:
            # i_offset == 0 if i itself is the last ISI of a burst
            offset_dists, = np.where(candidate_offsets[i:])
            if len(offset_dists) == 0:
                offset_dist = num_isi - i - 1
            else:
                offset_dist = offset_dists[0]

            if offset_dist+2 < min_spk:
                i += 1
                continue

            # Now still have to check if all ISI until candidate offset are < offset_ISI
            num_to_offset = np.sum(isi_below_offset[i:i+offset_dist+1])
            if (num_to_offset+1 < min_spk) or (num_to_offset != offset_dist+1):
                i += 1
                continue
            num_follow = num_to_offset
            
            # Burst properties
            burst_lengths.append(num_follow+1)
            f_intra = 1e3/np.mean(isi_vals[i:i+num_follow])
            if np.isnan(f_intra):
                raise Exception('NaN!')
            burst_intra_rates.append(f_intra)
            t_onset = t_rec[peak_idx[i]]
            t_offset = t_rec[peak_idx[min(i+num_follow, num_isi-1)]]
            # print("Burst at t={} with ISIs: {}".format(t_onset, isi_vals[i:i+num_follow]))
            if len(offset_times) > 0:
                inter_burst_intervals.append(t_onset - offset_times[-1])
            onset_times.append(t_onset)
            offset_times.append(t_offset)
            # Skip to end of burst
            isi_burst[i:i+num_follow+1] = True
            i += num_follow + 1
        else:
            i += 1

    burst_rate = len(burst_lengths) * 1e3 / (t_rec[-1] - t_rec[0])
    metrics = {
        'spikes_per_burst': burst_lengths,
        'intra_burst_rates': burst_intra_rates,
        'inter_burst_intervals': inter_burst_intervals,
        'burst_rate': burst_rate,
    }
    return metrics



def numpy_sum_psth(spiketrains, tstart, tstop, binwidth=10.0, average=False):
    """ 
    Sum peri-stimulus spike histograms (PSTH) of spike trains

    @param spiketimes   list of Hoc.Vector() objects or other iterables,
                        each containing spike times of one cell.
    
    @param tstart       stimulus time/start time for histogram
    
    @param tstop        stop time for histogram
    
    @param binwidth     bin width (ms)
    
    @return             hoc.Vector() containing binned spikes
    """
    # Compute bin edges
    num_bins = int((tstop-tstart)/binwidth)
    edges = np.linspace(tstart, tstop, num_bins+1)

    # Concatenate spike trains and compute histogram
    sts_concat = np.concatenate(spiketrains)
    bin_vals, bin_edges = np.histogram(sts_concat, edges, density=False)

    if average:
        bin_vals /= len(spiketrains)

    return bin_vals


def numpy_avg_rate_simple(spiketrains, tstart, tstop, binwidth):
    """
    Simple algorithm for calculating the running firing rate
    (average firing rate in each bin of the psth, averaged over spike trains)
    """
    # first compute the histogram
    avghist = numpy_sum_psth(spiketrains, tstart, tstop, binwidth)

    # divide by nb. of cells/trials and by binwidth in ms to get rate
    return avghist / (binwidth*1e-3*len(spiketrains))


def nrn_sum_psth(spiketrains, tstart, tstop, binwidth=10.0, average=False):
    """ 
    Sum peri-stimulus spike histograms (PSTH) of spike trains

    @param spiketimes   list of Hoc.Vector() objects or other iterables,
                        each containing spike times of one cell.
    
    @param tstart       stimulus time/start time for histogram
    
    @param tstop        stop time for histogram
    
    @param binwidth     bin width (ms)
    
    @return             hoc.Vector() containing binned spikes
    """

    # Create histogram with empty bins
    # avghist = h.Vector(int((tstop-tstart)/binwidth) + 2, 0.0)
    st1 = h.Vector(spiketrains[0])
    avghist = st1.histogram(tstart, tstop, binwidth)
    
    # add histogram for each cell (add bins element-wise)
    for st in spiketrains[1:-1]:
        if not isinstance(st, neuron.hoc.HocObject):
            st = h.Vector(st)
        
        spkhist = st.histogram(tstart, tstop, binwidth)
        avghist.add(spkhist) # add element-wise
    
    # divide by nb. of cells/trials and by binwidth in ms to get rate
    if average:
        avghist.div(len(spiketrains))
    return avghist


def nrn_avg_rate_simple(spiketrains, tstart, tstop, binwidth):
    """
    Simple algorithm for calculating the running firing rate
    (average firing rate in each bin of the psth, averaged over spike trains)

    @param      binwidth: float
                Bin width in (ms)

    @return     h.Vector() containing, the firing rate in each bin.
                The firing rate is just the number of spikes in a bin
                divided by the bin width in seconds.
    """
    # first compute the histogram
    avghist = nrn_sum_psth(spiketrains, tstart, tstop, binwidth)

    # divide by nb. of cells/trials and by binwidth in ms to get rate
    avghist.div(len(spiketrains)*binwidth*1e-3)
    return avghist


def nrn_avg_rate_adaptive(spiketrains, tstart, tstop, binwidth=10.0, minsum=15):
    """
    Compute running average firing rate of population.
    
    @param spiketrains  list of Hoc.Vector() objects or other iterables,

    @param minsum       minimum number of spikes required in adaptive
                        window used to compute per-bin firing rate

    @return             h.Vector() meanfreq, computed as follows:

                        For bin i, the corresponding mean frequency f_mean[i] is determined by centering an adaptive square window on i and widening the window until the number of spikes under the window equals <minsum>. Then f_mean[i] is calculated as:

                            f_mean[i] = N[i] / (m * binwidth * trials)
                        
                        where m is the number of 
                        bins included after widening the adaptive window.

    @see                https://neuron.yale.edu/neuron/static/new_doc/programming/math/vector.html#Vector.psth
    """
    # first compute the histogram
    psth = nrn_sum_psth(spiketrains, tstart, tstop, binwidth)

    # convert to firing rates
    ntrials = len(spiketrains)
    minsum = min(minsum, psth.sum())
    vmeanfreq = h.Vector()
    vmeanfreq.psth(psth, binwidth, ntrials, minsum)
    return vmeanfreq


def composite_spiketrain(spike_times, duration, dt_out=1.0, select=None):
    """
    Construct composite spiketrain for population.
    Baseline is subtracted.

    @param  spike_times : list(numpy.array)
            List of spike time vectors.
    
    @param  duration : float
            Total duration of simulation (ms).

    @param  select : int / enumerable[int]
            Number of spike trains to select or their indices.
    """
    num_samples = int(np.round(duration, 3)) / dt_out
    # Select subset of spike trains
    if select is None:
        selected = range(len(spike_times))
    elif isinstance(select, int):
        selected = np.random.choice(len(spike_times), select, replace=False)
    else:
        selected = select # assume list/tuple/numpy array
    # Insert ones at spike indices
    binary_trains = np.zeros((num_samples, len(selected)))
    for i_select, i_cell in enumerate(selected):
        st = spike_times[i_cell]
        spike_indices = (st / dt_out).astype(int)
        binary_trains[spike_indices, i_select] = 1.0
    return binary_trains.mean(axis=1)
    

def composite_spiketrain_coherence(trains_a, trains_b, duration, freq_res=1.0, overlap=0.75):
    """
    Coherence between composite (pooled) spike trains as binary signals, averaged.

    See Terry and Griffin 2008 : https://doi.org/10.1016/j.jneumeth.2007.09.014
    """
    Ts = 1.0 # ms
    nperseg = int(1e3 / Ts / freq_res) # fs / dF
    noverlap = int(nperseg * overlap)
    comp_trains = [composite_spiketrain(trains, duration, dt_out=Ts)
                    for trains in trains_a, trains_b]
    f, Cxy = scipy.signal.coherence(*comp_trains, fs=1e3/Ts, nperseg=nperseg, noverlap=noverlap)
    return f, Cxy


def combinatorial_spiketrain_coherence(
        spike_trains, duration,
        freq_res=1.0, overlap=0.75, max_comb=100):
    """
    Spiketrain coherence as average over all combinations of two groups.
    
    As described in McManus et al 2016, https://doi.org/10.1152/jn.00097.2016
    
    WARNING: scales badly with population size, e.g. for a population of 50
    spike trains, there are 126e12 ways of dividing them in two groups.
    """
    import scipy.special
    import itertools
    
    Ts = 1.0 # ms
    nperseg = int(1e3 / Ts / freq_res) # fs / dF
    noverlap = int(nperseg * overlap)
    
    N = len(spike_trains)
    k = N / 2
    inds_N = set(range(N))
    ncomb_tot = scipy.special.comb(N, k) # binomial coefficient
    if ncomb_tot > max_comb:
        grp_a_inds = [set(np.random.choice(N, k, replace=False)) for i in range(max_comb)]
    else:
        # Generate all possible combinations of choosing k out of N
        grp_a_inds = [set(inds) for inds in itertools.combinations(inds_N, k)]
    Cxy_sum = 0.0
    for grp_a in grp_a_inds:
        grp_b = inds_N - grp_a
        comp_spk_a = composite_spiketrain(spike_trains, dt_out=Ts, select=grp_a)
        comp_spk_b = composite_spiketrain(spike_trains, dt_out=Ts, select=grp_b)
        f, Cxy = scipy.signal.coherence(comp_spk_a, comp_spk_b, 
                                        fs=1e3/Ts, nperseg=nperseg, noverlap=noverlap)
        Cxy_sum += Cxy
    return f, Cxy_sum / len(grp_a_inds)


def morgera_covariance_complexity(
    v_rec, t_rec, interval, 
    t_window=1000.0, t_overlap=800.0):
    """
    Morgera covariance complexity ('synchronization index').

    Works on the raw signal so also measures synchronization of sub-threshold
    voltage oscillations. I.e. not only spike synchronization.

    Explained succinctly in this review article: https://arxiv.org/abs/q-bio/0603035

    Arguments
    ---------

    @param  v_rec : numpy.array
            Array with signals as columns (along first axis).

    References
    ----------

    S. S. Morgera, "Information Theoretic Complexity and Relation to Pattern
    Recognition", IEEE Transactions on Systems, Man, and Cybernetics, 
    15 (1985) 608-619
    """
    M_values = []
    delta_t = t_window - t_overlap
    Ts = t_rec[1] - t_rec[0]
    t_start, t_stop = interval
    t0_values = np.arange(t_start, t_stop, delta_t)
    t1_values = []
    for t0 in t0_values:
        window = [t0, t0+t_window]
        if window[1] > t_stop:
            break
        irange = [int((t - t_rec[0])/Ts) for t in window]
        islice = np.s_[irange[0]:irange[1]] # slice object
        
        # SVD and singular values
        u, s, vh = np.linalg.svd(v_rec[islice, :], full_matrices=True)
        lambas = s**2
        sigmas = lambas / lambas.sum()
        C = - 1./np.log(len(sigmas)) * np.sum(sigmas * np.log(sigmas))
        
        t1_values.append(window[1])
        M_values.append(1 - C)

    return zip(t0_values, t1_values), M_values


def get_efel_features(
        vm, t, interval, features, 
        threshold=None, raise_warnings=True, **kwargs):
    """
    Calculate electrophysiology features using eFEL.

    https://efel.readthedocs.io/en/latest/eFeatures.html
    http://bluebrain.github.io/eFEL/efeature-documentation.pdf

    @param  features : list[str]
            List of eFeature names described in eFEL documentation.

    @param  kwargs : dict[str, double/int]
            Extra eFEL parameters like 'interp_step' or required parameters
            listed under a feature.

    Example
    -------

    >>> from bgcellmodels.common import signal
    >>> features = ['ISI_values', 'burst_ISI_indices', 'burst_mean_freq', 'burst_number']
    >>> feat_vals = signal.get_efel_features(vm, t, [0.5e3, 3e3], features)
    >>> print("\n".join(("{} : {}".format(name, val) for name,val in feat_vals)))
    
    """
    
    import efel
    efel.reset()

    if threshold is not None:
        efel.setThreshold(threshold)

    if kwargs is not None:
        for pname, pval in kwargs.iteritems():
            if isinstance(pval, float):
                efel.setDoubleSetting(pname, pval)
            elif isinstance(pval, int):
                efel.setIntSetting(pname, pval)
            else:
                raise ValueError("Unknown eFEL parameter or unexpected type:"
                                 "{} : {}".format(pname, pval))

    # Put trace in eFEL compatible format
    efel_trace = {
        'T': t,
        'V': vm,
        'stim_start': [interval[0]],
        'stim_end': [interval[1]],
    }

    # Calculate spike times from response
    values = efel.getFeatureValues(
        [efel_trace],
        features,
        raise_warnings=raise_warnings
    )
    efeat_values = {
        feat_name: values[0][feat_name] for feat_name in features
    }

    return efeat_values


def get_all_pyelectro_features(
        vm, t, interval, analysis_params=None):
    """
    Calculate electrophysiology features using PyElectro.

    https://pyelectro.readthedocs.io/en/latest/pyelectro.html

    @param  analysis_params : dict[str, float]
            Parameters for calculation of voltage trace metrics.

    @return features : dict[str, float]
            Dictionary containing all PyElectro features.

    Notes
    -----

    See IClampAnalysis.analyse() method for calculating individual features:
    https://github.com/lkoelman/pyelectro/blob/master/pyelectro/analysis.py#L1204
    """

    from pyelectro import analysis

    if analysis_params is None:
        analysis_params = {
            'peak_delta':       1e-4, # the value by which a peak or trough has to exceed its neighbours to be considered outside of the noise
            'baseline':         -10., # voltage at which AP width is measured
            'dvdt_threshold':   0, # used in PPTD method described by Van Geit 2007
        }

    trace_analysis = analysis.IClampAnalysis(
                            vm,
                            t,
                            analysis_params,
                            start_analysis = interval[0],
                            end_analysis = interval[1],
                            show_smoothed_data = False
                        ) 

    trace_analysis.analyse()

    return trace_analysis.analysis_results