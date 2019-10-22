
# Third party libraries
import numpy as np
import scipy.signal
from scipy.ndimage.filters import gaussian_filter


def curve_distance(xy1, xy2, **kwargs):
    """
    Distance betweeen two curves

    @param  smooth_method : str
            - 'polyfit'
            - 'convolve'
            - 'savgol'

    @param  distance_metric : str
            - 'sumsquared'
            - 'RMS' / 'norm'

    @param  reference : int
            Reference for distance measurement. The x-values of reference
            dataset will be used, and the other dataset will be interpolated

    @param  (smooth_method == 'polyfit')
            poly_order : int

    @param  (smooth_method == 'convolve')
            window_length : int
            window_function : str
            window_sigma : double (window_function == 'gauss')

    @param  (smooth_method == 'savgol')
            window_length : int
            poly_order : int
    """
    data = [{'x': xy1[0], 'y': xy1[1]},
            {'x': xy2[0], 'y': xy2[1]}]

    # Smooth both series
    # - Farries 2012a uses Gaussian filter with sigma=0.05 phse units
    # - filtering approaches: 
    #     - numpy.convolve : https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    #     - scipy.signal.savgol_filter(x, window_length, polyorde)
    #     - scipy.ndimage.filters.gaussian_filter
    smooth_method = kwargs['smooth_method']

    if smooth_method == 'polyfit':
        for i in 0, 1:
            coeff = np.polyfit(data[i]['x'], data[i]['y'], kwargs['poly_order'])
            data[i]['poly'] = poly = np.poly1d(coeff)
            data[i]['y_smooth'] = poly(data[i]['x'])
            data[i]['y_interp'] = poly(data[1-i]['x'])

    elif smooth_method == 'convolve' and kwargs['window_function'] == 'gauss':
        for i in 0, 1:
            data[i]['y_smooth'] = gaussian_filter(data[i]['y'],
                                    sigma=kwargs['window_sigma'])

    elif smooth_method == 'convolve':
        for i in 0, 1:
            data[i]['y_smooth'] = smooth(data[i]['y'],
                window_len=kwargs['window_length'],
                window=kwargs['window_function'])

    elif smooth_method == 'savgol':
        for i in 0, 1:
            data[i]['y_smooth'] = scipy.signal.savgol_filter(data[i]['y'],
                kwargs['window_length'], kwargs['poly_order'])

    else:
        raise ValueError(smooth_method)

    # Interpolate data (y-values at x-values of other dataset)
    for i in 0, 1:
        if 'y_interp' not in data[i]:
            data[i]['y_interp'] = np.interp(data[1-i]['x'],
                                            data[i]['x'], data[i]['y'])

    reference = kwargs['reference']
    if reference != 0 and reference != 1:
        raise ValueError(reference)
        
    y_ref = data[reference]['y_smooth']
    y_cmp = data[1-reference]['y_interp']
    dy = np.asarray(y_ref - y_cmp)

    # Distance between smoothed series
    distance_metric = kwargs['distance_metric']
    if distance_metric == 'sumsquared':
        distance = np.sum(dy**2)

    elif distance_metric in ('RMS', 'norm'):
        distance = np.linalg.norm(dy)

    else:
        raise ValueError(distance_metric)

    return distance


def smooth(x, window_len=11, window='hanning', pad_method='boundary'):
    """
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    Author: anonymous @ https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    ARGUMENTS
    ---------

    @param  x: the input signal 

    @param  window_len: the dimension of the smoothing window; should be an odd integer
    
    @param  window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'. Flat window will produce a moving average
            smoothing.

    OUTPUTS
    -------

        the smoothed signal
        
    USAGE
    -----

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    if not isinstance(pad_method, tuple) and pad_method not in (
        'mirror', 'boundary'):
        raise ValueError("Pad method must be 'mirror', 'boundary', or a tuple of floats.")

    # Pad with mirrored copy
    # - by using convolve mode 'valid', you remove window_len // 2 samples
    #   on each side of the longest series. This is because the 'sliding' starts
    #   with the shortest series (window) centered on the first sample, and
    #   all non-fully overlappling convolution results will be removed
    pad_size = int(window_len // 2)
    if pad_method == 'mirror':
        s = np.r_[x[pad_size:0:-1],x,x[-2:-2-pad_size:-1]]
    elif pad_method == 'boundary':
        s = np.r_[pad_size*[x[0]], x, pad_size*[x[-1]]]
    else:
        s = np.r_[pad_size*[pad_method[0]], x, pad_size*[pad_method[1]]]
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        wfun = getattr(np, window)
        w = wfun(window_len)

    # Convolution is integral, so scale by window sum
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y