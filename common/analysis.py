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
import re

################################################################################
# Recording & Analysis functions
################################################################################

def rec_currents_activations(traceSpecs, sec_tag, sec_loc, ion_species=None, 
								currents=True, chan_states=True, ion_conc=True):
	"""
	Specify trace recordings for all ionic currents
	and activation variables 

	@param traceSpecs	collections.OrderedDict of trace specifications

	@param	sec_tag		tag for the section in the section dictionary passed
						to analysis.recordTraces. Note that the first two
						characters must be unique

	@param	sec_loc		location in the section to record the variable
						(location maps to a segment where the var is recorded)

	@effect				for each ionic current, insert a trace specification
						for the current, open fractions, activation, 
						and inactivation variables


	EXAMPLE USAGE

		secs = {'soma': soma, 'dend': dendsec}

		traceSpecs = collections.OrderedDict()
		traceSpecs['V_soma'] = {'sec':'soma', 'loc':0.5, 'var':'v'}
		traceSpecs['t_global'] = {'var':'t'}

		rec_currents_activations(traceSpecs, 'soma', 0.5)
		rec_currents_activations(traceSpecs, 'dend', 0.9, ion_species=['ca','k'])

		recordStep = 0.025
		recData = analysis.recordTraces(secs, traceSpecs, recordStep)
		h.init()
		h.run()

		figs_soma, cursors_soma = plot_currents_activations(recData, recordStep, sec_tag='soma')
		figs_dend, cursors_dend = plot_currents_activations(recData, recordStep, sec_tag='dend')

	"""
	if ion_species is None:
		ion_species = ['na', 'k', 'ca', 'nonspecific']

	# Derive suffix for traces from section tag
	suf = '_' + sec_tag[0:2]
	ts = traceSpecs

	if 'na' in ion_species:
		if currents:
			# Na currents
			ts['I_NaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Na','var':'ina'} # transient sodium
			ts['I_NaP'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'NaL','var':'inaL'} # persistent sodium
		if chan_states:
			# Na Channel open fractions
			ts['O_NaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Na','var':'o'}
			# NOTE: leak currents such as I_NaL and gpas_STh are always open
			# Na channel activated fractions
			ts['A_NaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Na','var':'m'}
			# Na channel inactivated fractions
			ts['B_NaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Na','var':'h'}

	if 'k' in ion_species:
		if currents:
			# K currents
			ts['I_KDR'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'KDR','var':'ik'}
			ts['I_Kv3'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Kv31','var':'ik'}
			ts['I_KCa'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'sKCa','var':'isKCa'}
		if chan_states:
			# K channel open fractions
			ts['O_KDR'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'KDR','var':'n'}
			ts['O_Kv3'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Kv31','var':'p'}
			ts['O_KCa'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'sKCa','var':'w'}
			# K channels activated fractions - same as open fractions (single state variable)
			ts['A_KDR'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'KDR','var':'n'}
			ts['A_Kv3'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Kv31','var':'p'}
			ts['A_KCa'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'sKCa','var':'w'}
			# K channels inactivated fractions - not present (single state variable)

	if 'ca' in ion_species:
		# Ions
		ts['C_CaL_cai'+suf] = {'sec':sec_tag,'loc':sec_loc,'var':'cai'} # intracellular calcium concentration
		ts['C_CaT_cai'+suf] = {'sec':sec_tag,'loc':sec_loc,'var':'cai'} # intracellular calcium concentration
		if currents:
			# Ca currents
			ts['I_CaL'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'iLCa'}
			ts['I_CaN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'iNCa'}
			ts['I_CaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'CaT','var':'iCaT'}
		if chan_states:
			# Ca channel open fractions
			ts['O_CaL'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'o_L'}
			ts['O_CaN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'o_N'}
			ts['O_CaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'CaT','var':'o'}
			# Ca channels activated fractions
			ts['A_CaL'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'q'} # shared activation var for L/N
			ts['A_CaN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'q'} # shared activation var for L/N
			ts['A_CaT'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'CaT','var':'r'}
			# Ca channels inactivated fractions
			ts['B_CaL'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'h'} # Ca-depentdent inactivation
			ts['B_CaN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'HVA','var':'u'} # V-dependent inactivation
			ts['B_CaTf'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'CaT','var':'s'} # fast inactivation
			ts['B_CaTs'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'CaT','var':'d'} # slow inactivation

	if 'nonspecific' in ion_species:
		if currents:
			# Nonspecific currents
			ts['I_HCN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Ih','var':'ih'}
		if chan_states:
			# Nonspecific channel open fractions
			ts['O_HCN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Ih','var':'f'}
			# Nonspecific channels activated fractions - same as open fractions (single state variable)
			ts['A_HCN'+suf] = {'sec':sec_tag,'loc':sec_loc,'mech':'Ih','var':'f'}
			# Nonspecific channels activated fractions - not present (single state variable)

def plot_currents_activations(recData, recordStep, timeRange=None, sec_tag=None):
	"""
	Plot currents and (in)activation variable for each ionic
	current in the same axis. Ionic currents are grouped per ion
	in one figure, and the x-axes are synchronized for zooming
	and panning.

	@param recData		traces recorded using a trace specification provided
						by rec_currents_activations()

	@return				tuple figs, cursors where figs is a list of figures
						that were created and cursors a list of cursors

	EXAMPLE USAGE:		see function rec_currents_activations()
	"""
	figs = []
	cursors = []

	# Plot activations-currents on same axis per current
	ions_chans = [('NaT', 'NaP', 'HCN'), ('KDR', 'Kv3', 'KCa'), ('CaL', 'CaN', 'CaT')]
	for iontraces in ions_chans: # one figure for each ion
		fig, axrows = plt.subplots(len(iontraces), 1, sharex=True) # one plot for each channel
		for i, trace_abbr in enumerate(iontraces):
			# Find traces that are marked with the tracee abbreviation (e.g. 'CaT')
			if sec_tag is None:
				pat = re.compile(r'^[A-Z]_' +trace_abbr) # find 'char+_+abbr' at beginning of word
			else:
				sec_suffix = sec_tag[0:2]
				pat = re.compile(r'^[A-Z]_' + trace_abbr + r'\w+' + sec_suffix + r'$')
			chanFilter = lambda x: re.search(pat, x) # variables plotted on left axis match this filter
			twinFilter = lambda	x: x.startswith('I_') # vars plotted on right axis match this filter
			# Plot traces that match pattern
			cumulPlotTraces(recData, recordStep, showFig=False, 
								fig=None, ax1=axrows[i], yRange=(-0.1,1.1), timeRange=timeRange,
								includeFilter=chanFilter, twinFilter=twinFilter)
			# Add figure interaction
			cursor = matplotlib.widgets.MultiCursor(fig.canvas, fig.axes, 
						color='r', lw=1, horizOn=False, vertOn=True)
		figs.append(fig)
		cursors.append(cursor)

	return figs, cursors

def recordTraces(secs, traceSpecs, recordStep, duration=None):
	""" Record the given traces from section

	EXAMPLE USE:

	secs = {'soma': soma, 'dend', dends[1], 'izhpp', izh, 'synpp':syn}

	traceSpecs = {
		'V_izhi': {'pointp': 'izh'},
		'g_syn': {'pointp': 'synpp', 'var': 'g'}
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
			else:
				ptr = h.__getattribute__('_ref_'+spec['var'])

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
				traceSharex=False, showGrid=True, title=None, traceXforms=None):
	"""
	Plot previously recorded traces

	- traceData
		dict(trace_name -> h.Vector()) containing recorded traces

	- traceXforms
		dict(trace_name -> function) containg transformation to apply
		to trace before plotting

	- timeRange ([start:stop])
		Time range of spikes shown; if None shows all (default: None)
	
	- oneFigPer ('cell'|'trace')
		Whether to plot one figure per cell (showing multiple traces) 
		or per trace (showing multiple cells) (default: 'cell')
	
	- showFig (True|False)
		Whether to show the figure or not (default: True)
	
	- labelTime (True|False)
		whether to show the time axis label (default:True)
	
	- includeTraces (['V_soma', ...])
		traces to include in this plot
	
	- excludeTraces (['V_soma', ...])
		traces to exclude in this plot
	
	- yRange
		y limit for range, e.g. (0, 60). Can be a tuple, list or dict with traces as keys
	
	- traceSharex
		if True, all x-axes will be shared (maintained during zooming/panning), else
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
			ax = plt.subplot(111, sharex=shared_ax)
			if itrace==0 and traceSharex==True:
				shared_ax = ax

		# Get data to plot
		if (traceXforms is not None) and (trace in traceXforms):
			xform = traceXforms[trace]
			tracevec = xform(traceData[trace].as_numpy())
		else:
			tracevec = traceData[trace].as_numpy()
		
		data = tracevec[int(timeRange[0]/recordStep):int(timeRange[1]/recordStep)]
		t = np.arange(timeRange[0], timeRange[1]+recordStep, recordStep)

		plt.plot(t[:len(data)], data, linewidth=1.0, color=colorList[itrace%len(colorList)])

		# Axes ranges/labels
		plt.ylabel(trace, fontsize=fontsiz)
		if labelTime:
			plt.xlabel('Time (ms)', fontsize=fontsiz)
		if isinstance(yRange, dict):
			if trace in yRange:
				plt.ylim(yRange[trace])
		elif yRange is not None:
			plt.ylim(yRange)
		plt.xlim(timeRange)

		# Customize the grid
		ax.grid(showGrid)

	if title:
		plt.suptitle(title) # suptitle() is Fig title, title() is ax title

	if showFig:
		plt.show(block=False)
	
	return figs

allcolors = [
	'#7742f4', # Dark purple
	[0.90,0.76,0.00], # Ochre
	[0.42,0.83,0.59], # soft pastel green
	[0.90,0.32,0.00], # pastel red brick
	[0.90,0.59,0.00], # OrangeBrown
	'#f442c5', # Pink
	'#c2f442', # Lime
	[1.00,0.85,0.00], # hard yellow
	[0.33,0.67,0.47], # dark pastel green
	[1.00,0.38,0.60], [0.57,0.67,0.33], [0.5,0.2,0.0],
	[0.71,0.82,0.41], [0.0,0.2,0.5],
]
greenish = [
	[0.42,0.83,0.59], # soft pastel green
	'#c2f442', # Lime
	[0.33,0.67,0.47], # dark pastel green
]
redish = [
	[0.90,0.32,0.00], # pastel red brick
	'#f442c5', # Bright pink
	[0.90,0.59,0.00], # OrangeBrown
]
blueish = [
	'#7742f4', # Dark purple
	'#c2f442', # Soft cyan blue
	'#0066FF', # pastel blue
]
solid_styles = ['-']
broken_styles = ['--', '-.', ':']

def pick_line(trace_name, trace_index, solid_only=False):
	""" Pick a line style and color based on the trace name """
	style_map = {
		'I': (allcolors, solid_styles),
		'V': (allcolors, solid_styles),
		'C': (allcolors, solid_styles),
		'A': (greenish, broken_styles),
		'B': (redish, broken_styles),
		'O': (blueish, broken_styles),
	}
	if solid_only:
		style_map['A'] = (greenish, solid_styles)
		style_map['B'] = (redish, solid_styles)
		style_map['O'] = (blueish, solid_styles)
	default_style = (allcolors, broken_styles)

	# pick a line style
	match_prefix = re.search(r'^[a-zA-Z]', trace_name) # first letter
	if match_prefix:
		prefix = match_prefix.group()
		colors, styles = style_map.get(prefix, default_style)
	else:
		colors, styles = default_style
	return colors[trace_index%len(colors)], styles[trace_index%len(styles)]


def match_traces(recData, matchfun):
	"""
	Get ordered dictionary with matching traces.

	@param matchfun		lambda string -> bool
	"""
	return collections.OrderedDict([(trace_name,v) for trace_name,v in recData.iteritems() if matchfun(trace_name)])

def cumulPlotTraces(traceData, recordStep, timeRange=None, cumulate=False,
					includeTraces=None, excludeTraces=None,
					showFig=True, colorList=None, lineList=None, 
					traceSharex=None, fig=None, showGrid=True,
					yRange=None, twinFilter=None, includeFilter=None,
					yRangeR=None, ax1=None, solid_only=False):
	""" Cumulative plot of traces

	@param fig 				if provided, add axs as subplot to this fig

	@param traceSharex		a matplotlib.axes.Axes object to share the
							x-axis with (x-axis locked while zooming/panning)

	@param twinFilter		function that takes as argument a trace name, 
							returns True if trace should be plotted on right
							axis (using ax.twinx()), False if on left
	"""

	tracesList = traceData.keys()
	if includeFilter is not None:
		tracesList = [trace for trace in tracesList	if includeFilter(trace)]
	if includeTraces is not None:
		tracesList = [trace for trace in tracesList if trace in includeTraces]
	if excludeTraces is not None:
		tracesList = [trace for trace in tracesList if trace not in excludeTraces]
	if len(tracesList) == 0:
		print('WARNING: No traces left for plotting after applying filter! Empty figure will be returned.')
		return fig

	if timeRange is None:
		timeRange = [0, traceData[tracesList[0]].size()*recordStep]

	fontsiz=12

	# Get the axes to draw on
	if not fig and not ax1: # create fig and axis
		fig = plt.figure()
		ax1 = fig.add_subplot(111, sharex=traceSharex)
	elif not fig and ax1: # get fig from given axis
		fig = ax1.figure
	elif fig and not ax1: # add new axis to figure
		nax = len(fig.axes)
		for i, ax in enumerate(fig.axes):
			ax.change_geometry(nax+1, 1, i+1) # change grid and position in grid
		ax1 = fig.add_subplot(nax+1, 1, nax+1, sharex=traceSharex)
	if twinFilter is not None:
		ax2 = ax1.twinx()
	else:
		ax2 = None
		twinFilter = lambda x: False

	t = np.arange(timeRange[0], timeRange[1]+recordStep, recordStep)
	cumulTrace = np.zeros(int((timeRange[1]-timeRange[0])/recordStep))

	# plot each trace
	lines = []
	for itrace, tracename in enumerate(tracesList):
		tracevec = traceData[tracename].as_numpy()
		data = tracevec[int(timeRange[0]/recordStep):int(timeRange[1]/recordStep)]
		if twinFilter(tracename):
			pax = ax2
		else:
			pax = ax1
		line_color, line_style = pick_line(tracename, itrace, solid_only=solid_only)
		li, = pax.plot(t[:len(data)], data+cumulTrace, label=tracename, 
						color=line_color, linestyle=line_style)
		lines.append(li)
		if cumulate: cumulTrace += data

	# Axes ranges/labels
	plt.xlim(timeRange)
	if yRange is not None:
		ax1.set_ylim(yRange)
	if yRangeR is not None:
		ax2.set_ylim(yRangeR)

	# Set labels
	L_prefixes = []
	R_prefixes = []
	for tracename in tracesList:
		match = re.search(r'^._', tracename)
		if match and twinFilter(tracename):
			R_prefixes.append(match.group()[:-1])
		elif match:
			L_prefixes.append(match.group()[:-1])
	ax1.set_ylabel(', '.join(set(L_prefixes)), fontsize=fontsiz)
	if ax2:
		ax2.set_ylabel(', '.join(set(R_prefixes)), fontsize=fontsiz)

	# legend for lines plotted in all axes
	ax1.legend(lines, [l.get_label() for l in lines])

	# Customize the grid
	ax1.grid(showGrid)

	if showFig:
		plt.show(block=False)
	return fig

				
