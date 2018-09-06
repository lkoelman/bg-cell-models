"""
Signal analysis tools for electrophysiology data.

@author     Lucas Koelman
"""

import neuron
h = neuron.h
import numpy as np


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