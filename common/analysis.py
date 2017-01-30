"""
Common plotting/recording functions for cell experiments.

@author Lucas Koelman
@date 26/10/2016
"""

from neuron import h
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections

def recordTraces(secs, traceSpecs, recordStep, duration=None):
	""" Record the given traces from section

	EXAMPLE USE:

	secs = {'soma': soma, 'dend', dends[1], 'izhpp', izh}

	traceSpecs = {
		'V_izhi':{'pointp': 'izh'}
		'V_soma':{'sec':'soma','loc':0.5,'var':'v'},
		'GP_RT_cai':{'sec':'soma','loc':0.5,'var':'cai'},
		'GP_RT_ainf':{'sec':'soma','loc':0.5,'mech':'gpRT','var':'a_inf'}, 
		'GP_RT_r':{'sec':'soma','loc':0.5,'mech':'gpRT','var':'r'},
		'STN_r':{'sec':'soma','loc':0.5,'mech':'stn','var':'r'},
		'STN_p':{'sec':'soma','loc':0.5,'mech':'stn','var':'p'},
		'STN_q':{'sec':'soma','loc':0.5,'mech':'stn','var':'q'},
	}

	"""
	recData = collections.OrderedDict() # empty dict for storing recording vectors

	for trace, spec in traceSpecs.iteritems():
		ptr = None
		if 'loc' in spec:
			sec = secs[spec['sec']]
			if 'mech' in spec:  # eg. soma(0.5).hh._ref_gna
				ptr = sec(spec['loc']).__getattribute__(spec['mech']).__getattribute__('_ref_'+spec['var'])
			else:  # eg. soma(0.5)._ref_v
				ptr = sec(spec['loc']).__getattribute__('_ref_'+spec['var'])
		else:
			if 'pointp' in spec: # eg. soma.izh._ref_u
				if spec['pointp'] in secs:
					pp = secs[spec['pointp']]
					ptr = pp.__getattribute__('_ref_'+spec['var'])

		if ptr:  # if pointer has been created, then setup recording
			if duration is not None:
				recData[trace] = h.Vector(duration/recordStep+1).resize(0)
			else:
				recData[trace] = h.Vector()
			recData[trace].record(ptr, recordStep)

	return recData


def plotTraces(traceData, recordStep, timeRange=None, oneFigPer='cell', 
				includeTraces=None, excludeTraces=None, labelTime=False,
				showFig=True, colorList=None, lineList=None, yRange=None,
				traceSharex=False, showGrid=True):
	""" Plot previously recorded traces

		- timeRange ([start:stop]): Time range of spikes shown; if None shows all (default: None)
		- oneFigPer ('cell'|'trace'): Whether to plot one figure per cell (showing multiple traces) 
			or per trace (showing multiple cells) (default: 'cell')
		- showFig (True|False): Whether to show the figure or not (default: True)
		- labelTime (True|False): whether to show the time axis label (default:True)
		- includeTraces (['V_soma', ...]): traces to include in this plot
		- excludeTraces (['V_soma', ...]): traces to exclude in this plot
		- yRange: y limit for range, e.g. (0, 60). Can be a tuple, list or dict with traces as keys
		- traceSharex: if True, all x-axes will be shared (maintained during zooming/panning), else
			if an axes object is provided, share x axis with that axes
		
		Returns figure handles
	"""

	tracesList = traceData.keys()
	if includeTraces is not None:
		tracesList = [trace for trace in tracesList if trace in includeTraces]
	if excludeTraces is not None:
		tracesList = [trace for trace in tracesList if trace not in excludeTraces]

	# time range
	if timeRange is None:
		timeRange = [0, traceData[tracesList[0]].size()*recordStep]

	if colorList is None:
		colorList = [[0.42,0.67,0.84], [0.90,0.76,0.00], [0.42,0.83,0.59], [0.90,0.32,0.00],
				[0.34,0.67,0.67], [0.90,0.59,0.00], [0.42,0.82,0.83], [1.00,0.85,0.00],
				[0.33,0.67,0.47], [1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
				[0.71,0.82,0.41], [0.0,0.2,0.5]]

	# Sharing axes for comparing signals
	shared_ax = None
	if isinstance(traceSharex, matplotlib.axes.Axes):
		shared_ax = traceSharex

	figs = []
	fontsiz=12
	for itrace, trace in enumerate(tracesList):
		if oneFigPer == 'cell':
			if itrace == 0:
				figs.append(plt.figure())
				ax = plt.subplot(len(tracesList), 1, itrace+1, sharex=shared_ax)
				if traceSharex == True:
					shared_ax = ax
			else:
				ax = plt.subplot(len(tracesList), 1, itrace+1, sharex=shared_ax)
		else: # one separate figure per trace
			fig = plt.figure()
			figs.append(fig)
			ax = fig.axes[0]

		# Get data to plot
		tracevec = traceData[trace].as_numpy()
		data = tracevec[int(timeRange[0]/recordStep):int(timeRange[1]/recordStep)]
		t = np.arange(timeRange[0], timeRange[1]+recordStep, recordStep)

		plt.plot(t[:len(data)], data, linewidth=1.0, color=colorList[itrace%len(colorList)])

		# Axes ranges/labels
		plt.ylabel(trace, fontsize=fontsiz)
		if labelTime:
			plt.xlabel('Time (ms)', fontsize=fontsiz)
		if isinstance(yRange, dict):
			plt.ylim(yRange[trace])
		elif yRange is not None:
			plt.ylim(yRange)
		plt.xlim(timeRange)

		# Customize the grid
		ax.grid(showGrid)

	if showFig:
		plt.show(block=False)
	return figs

def cumulPlotTraces(traceData, recordStep, timeRange=None, cumulate=False,
					includeTraces=None, excludeTraces=None,
					showFig=True, colorList=None, lineList=None, 
					traceSharex=None, parentFig=None, showGrid=True,
					yRange=None):
	""" Cumulative plot of traces """

	tracesList = traceData.keys()
	if includeTraces is not None:
		tracesList = [trace for trace in tracesList if trace in includeTraces]
	if excludeTraces is not None:
		tracesList = [trace for trace in tracesList if trace not in excludeTraces]

	if timeRange is None:
		timeRange = [0, traceData[tracesList[0]].size()*recordStep]

	if colorList is None:
		colorList = ['#7742f4', # Dark purple
					[0.90,0.76,0.00], # Ochre
					[0.42,0.83,0.59], 
					[0.90,0.32,0.00],
					[0.90,0.59,0.00], # OrangeBrown
					'#f442c5', # Pink
					'#c2f442', # Lime
					[1.00,0.85,0.00],
					[0.33,0.67,0.47], 
					[1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
					[0.71,0.82,0.41], [0.0,0.2,0.5]]

	if lineList is None:
		lineList = ['-', '--', '-.', ':']

	fontsiz=12
	if parentFig:
		fig = parentFig
		nax = len(fig.axes)
		for i, ax in enumerate(fig.axes):
			ax.change_geometry(nax+1, 1, i+1) # change grid and position in grid
		ax1 = fig.add_subplot(nax+1, 1, nax+1, sharex=traceSharex)
	else:
		fig = plt.figure()
		ax1 = fig.add_subplot(111, sharex=traceSharex)

	t = np.arange(timeRange[0], timeRange[1]+recordStep, recordStep)
	cumulTrace = np.zeros(int((timeRange[1]-timeRange[0])/recordStep))

	for itrace, trace in enumerate(tracesList):
		tracevec = traceData[trace].as_numpy()
		data = tracevec[int(timeRange[0]/recordStep):int(timeRange[1]/recordStep)]
		plt.plot(t[:len(data)], data+cumulTrace, label=trace, 
			color=colorList[itrace%len(colorList)], linestyle=lineList[itrace%len(lineList)])
		if cumulate: cumulTrace += data

	# Axes ranges/labels
	plt.xlim(timeRange)
	if yRange is not None:
		plt.ylim(yRange)
	plt.ylabel('I', fontsize=fontsiz)
	plt.xlabel(' + '.join(tracesList), fontsize=fontsiz)
	plt.legend()

	# Customize the grid
	ax1.grid(showGrid)

	if showFig:
		plt.show(block=False)
	return fig

				
