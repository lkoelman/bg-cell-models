"""
Signal manipulation tools for Neo

@author     Lucas Koelman
@date       22/11/2018
"""

import numpy as np
import scipy.signal


def make_slice(signal, interval):
    """
    Make numpy slice object for indexing Neo.AnalogSignal
    """
    Ts = signal.sampling_period.magnitude
    tstart = signal.t_start.magnitude
    irange = [int((t-tstart)/Ts) for t in interval]
    return np.s_[irange[0]:irange[1]] # slice object


def butterworth_filter(
    signal, cutoff, order, btype, 
    filter_format='sos', plot_response=False):
    """
    Butterworth lowpass/highpass/bandpass filtering.

    @param  btype : str
            See 'btype' argument of scipy.signal.butter.
            https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.signal.butter.html
    """
    Fs = signal.sampling_rate.rescale('Hz').magnitude
    Fn = Fs / 2. # Nyquist frequency

    # WARNING: phase of filter must be NEUTRAL at target frequency!!! -> CENTER on target
    if btype in ('lowpass', 'highpass'):
        Wn = cutoff / Fn
    else:
        Wn = [f/Fn for f in cutoff]

    # Note: filter in (numerator, denominator) format can be unstable
    #       => use 'sos' representation
    filter_repr = scipy.signal.butter(order, Wn, btype=btype, analog=False, output=filter_format)

    if filter_format == 'ba': # b=numerator, a=denominator
        # Check filter stability if in (num, denom) format. Otherwise -> NaN values
        b, a = filter_repr
        if not np.all(np.abs(np.roots(a))<1):
            raise Exception("Unstable filter!")
        signal_bp = signal.duplicate_with_new_array(
            scipy.signal.filtfilt(b, a, signal.as_array(), axis=0)) # can also use 'lfilter'

    elif filter_format == 'sos':
        sos = filter_repr
        signal_bp = signal.duplicate_with_new_array(
            scipy.signal.sosfiltfilt(sos, signal.as_array(), axis=0))

    if plot_response:
        import matplotlib.pyplot as plt

        if filter_format == 'ba':
            w, h = scipy.signal.freqz(b, a, np.linspace(0, np.pi, 2**np.ceil(np.log2(Fn))))
        elif filter_format == 'sos':
            w, h = scipy.signal.sosfreqz(sos, np.linspace(0, np.pi, 2**np.ceil(np.log2(Fn))))

        angles = np.unwrap(np.angle(h))
        fax = w * Fn / (np.pi)

        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle("Filter response (2*pi = {})".format(Fs))
        ax = axes[0]
        ax.plot(fax, abs(h), 'b') # 20 * np.log10(abs(h))
        ax.set_ylabel('Amplitude [dB]', color='b')

        ax = axes[1] # ax2 = ax.twinx()
        ax.plot(fax, angles, 'g')
        ax.set_ylabel('Angle (radians)', color='g')

        # plt.axis('tight')
        ax.set_xlim((0, 50))
        ax.set_xlabel('Frequency [Hz]')
        ax.grid(True)

    return signal_bp
    

def subsample(signal, fmax=None, factor=None, max_factor=20):
    """
    Subsample a Neo.AnalogSignal.
    """
    if not ((fmax is None) ^ (factor is None)): # XOR
        raise ValueError("Specify either a subsampling factor or maximum frequency to preserive!")
    elif factor is None:
        fs_old = signal.sampling_rate.rescale('Hz').magnitude 
        subsample_factor = int(fs_old / (2 * fmax))
    elif fmax is None:
        subsample_factor = factor
    subsample_factor = min(max_factor, subsample_factor)

    # Indexing automatically adjusts 'sampling period' attribute
    return signal[::subsample_factor, :]