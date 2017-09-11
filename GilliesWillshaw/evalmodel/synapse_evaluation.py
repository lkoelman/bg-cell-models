"""
Analyze PSP frequency contents for accurate synaptic mapping.

@author		Lucas Koelman

@date		11/09/2017
"""

from cellpopdata import StnModel
from proto_common import StimProtocol
from stn_model_evaluation import run_protocol

# see https://docs.scipy.org/doc/scipy/reference/signal.html
# see https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
from scipy import signal
from scipy import integrate
from scipy.fftpack import fft, fftfreq
import numpy as np
from matplotlib import pyplot as plt

def median_frequency(ft_freqs, ft_amps, f_upper):
	"""
	Compute median frequency similar to MATLAB medfreq.

	See https://uk.mathworks.com/matlabcentral/answers/271071-what-is-the-median-frequency-matlab-code
	"""
	# Find median
	i_upper = np.argmin(np.abs(ft_freqs-f_upper))

	# Integrate to f_upper (cumulative sum)
	cum_amp = integrate.cumtrapz(ft_amps[0:i_upper], ft_freqs[0:i_upper]) 
	
	# Find point where area/2 is reached
	half_area = cum_amp[-1]/2
	f_med = np.interp(half_area, ft_freqs[1:i_upper], cum_amp) 

	return f_med


def analyze_AMPA_spectrum(export_locals=True):
	"""
	Analyze frequency contents of AMPA EPSPs.

	RESULTS:

	median frequency between 72-106 Hz depending on which method used

	median frequency of EPSP triggered by single spike:
		- FFT method:		102.9 Hz
		- Welch method:		66.2 Hz
	"""

	# Run stimulation protocol to trigger an EPSP
	model = StnModel.Gillies2005
	protocol = StimProtocol.SINGLE_SYN_GLU

	evaluator = run_protocol(protocol, model)
	rec_data = evaluator.model_data[model]['rec_data'][protocol]
	trace_data = rec_data['trace_data']
	rec_dt = rec_data['rec_dt']

	# signal parameters
	dt = rec_dt		# milliseconds
	T = rec_dt*1e-3	# seconds
	fs = 1.0 / (T)	# Hz
	t_AP = 850.0	# ms (SETPARAM: time of incoming spike)

	# extract PSP waveform
	Vpost = trace_data['V_GLUseg0'].as_numpy()
	psp_interval = [t_AP-250, t_AP+250]
	sig = Vpost[int(psp_interval[0]/dt):int(psp_interval[1]/dt)]
	N = len(sig)
	sig -= np.mean(sig)

	################################################
	# FFT

	# window it
	window = signal.blackmanharris(len(sig))
	sig_tapered = window * sig

	# FFT of signal
	ywf = fft(sig_tapered)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	# Median frequency using FFT method
	ft_amps = 2.0/N * np.abs(ywf[1:N//2])
	ft_freqs = xf[1:N//2]
	f_upper = 250
	f_med = median_frequency(ft_freqs, ft_amps, f_upper)

	print("Median frequency using FFT method is {} Hz".format(f_med))

	################################################
	# Periodogram (PSD)

	# Welch PSD 
	# window = signal.get_window('hann', len(sig)/4)
	# window_width = len(window)
	# nfft = window_width
	# nfft = 2**int(np.log2(len(sig))) # high resolution
	# fx, Pxx = signal.welch(sig, fs=fs, window=window, nperseg=window_width, 
	# 			nfft=nfft, noverlap=window_width/2)

	# NFFT equal to segment length, overlap = 50%
	segment_length = 2**int(np.log2(len(sig)/4))
	fx, Pxx = signal.welch(sig, fs=fs, nperseg=segment_length)

	# plt.figure()
	# plt.subplot(2,1,1)
	# plt.plot(fx, Pxx)
	# plt.subplot(2,1,2)
	# plt.semilogy(fx, Pxx)
	# plt.show(block=False)

	f_med = median_frequency(fx, Pxx, f_upper)

	print("Median frequency using Welch method is {} Hz".format(f_med))

	# # Plot integral of FFT amplitude and 50% percentile
	# plt.figure()
	# plt.plot(ft_freqs[1:i_upper], cum_amp)
	# plt.vlines(f_med, 0, max(cum_amp), 'r')
	# plt.show(block=False)

	# # Plot spectrum
	# fig, ax = plt.subplots(3,1)
	# ax[0].plot(sig_tapered)
	# ax[1].plot(ft_freqs, ft_amps)
	# ax[1].set_xlim([0, 250])
	# ax[2].semilogy(ft_freqs, ft_amps)
	# ax[2].set_xlim([0,250])
	# plt.show(block=False)

	if export_locals:
		globals().update(locals())


def analyze_PSP_spectrum(export_locals=True):
	"""
	Analyze frequency contents of EPSPs.


	AMPA:

		median frequency of EPSP triggered by burst of 10 spikes at 100 Hz
			- FFT method:		27.68 Hz
			- Welch method:		88.97 Hz

		median frequency of EPSP triggered by single spike:
			- FFT method:		27.30 Hz
			- Welch method:		83.24 Hz

	
	NMDA:

		median frequency of EPSP triggered by burst of 10 spikes at 100 Hz
			- FFT method:		2.3 Hz
			- Welch method:		2.3 Hz

		median frequency of EPSP triggered by single spike:
			- FFT method:		0.42 Hz
			- Welch method:		0.37 Hz

	GABA-A:

		median frequency of EPSP triggered by burst of 5 spikes at 100 Hz
			- FFT method:		1.24 Hz
			- Welch method:		17.68 Hz

		median frequency of EPSP triggered by single spike:
			- FFT method:		0.07 Hz
			- Welch method:		1.10 Hz

	GABA-B:

		median frequency of EPSP triggered by burst of 5 spikes at 100 Hz
			- FFT method:		2.05 Hz
			- Welch method:		4.08 Hz

		median frequency of EPSP triggered by single spike:
			- FFT method:		1.66 Hz
			- Welch method:		2.57 Hz
	"""

	# Run stimulation protocol to trigger an EPSP
	model = StnModel.Gillies2005
	protocol = StimProtocol.SINGLE_SYN_GLU # AMPA & NMDA
	# protocol = StimProtocol.SINGLE_SYN_GABA # GABA-A & GABA-B

	evaluator = run_protocol(protocol, model)
	rec_data = evaluator.model_data[model]['rec_data'][protocol]
	trace_data = rec_data['trace_data']
	rec_dt = rec_data['rec_dt']

	# signal parameters
	dt = rec_dt		# milliseconds
	T = rec_dt*1e-3	# seconds
	fs = 1.0 / (T)	# Hz
	t_AP = 850.0	# ms (SETPARAM: time of incoming spike)

	# extract PSP waveform
	if 'V_GLUseg0' in trace_data:
		Vpost = np.array(trace_data['V_GLUseg0'])
	else:
		Vpost = np.array(trace_data['V_GABAseg0'])

	psp_interval = [t_AP-250, t_AP+650]
	sig = Vpost[int(psp_interval[0]/dt):int(psp_interval[1]/dt)]
	N = len(sig)
	sig -= np.mean(sig)

	################################################
	# FFT method

	# window it
	# window = signal.blackmanharris(len(sig)) # see https://en.wikipedia.org/wiki/Window_function
	window = signal.kaiser(len(sig), beta=14) # window with good side-lobe characteristics
	sig_tapered = window * sig

	# FFT of signal
	ywf = fft(sig_tapered)
	xf = fftfreq(len(sig_tapered), T)
	# xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	# Median frequency using FFT method
	ft_amps = 2.0/N * np.abs(ywf[1:N//2])
	ft_freqs = xf[1:N//2]
	f_upper = 250
	f_med = median_frequency(ft_freqs, ft_amps, f_upper)

	print("Median frequency using FFT method is {} Hz".format(f_med))


	################################################
	# Periodogram method

	# Welch method with NFFT equal to segment length, overlap = 50%
	segment_length = 2**int(np.log2(len(sig)/4))
	window = signal.kaiser(segment_length, beta=14)
	fx, Pxx = signal.welch(sig, fs=fs, window=window)

	f_med = median_frequency(fx, Pxx, f_upper)

	print("Median frequency using Welch method is {} Hz".format(f_med))

	if export_locals:
		globals().update(locals())


if __name__ == '__main__':
	analyze_PSP_spectrum()