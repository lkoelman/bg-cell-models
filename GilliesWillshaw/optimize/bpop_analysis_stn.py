"""
Analysis of STN model optimization using BluePyOpt.

@author	Lucas Koelman

@date	6/10/2017

@see	based on scripts:
			https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/l5pc_analysis.py
			https://github.com/BlueBrain/BluePyOpt/blob/master/examples/l5pc/opt_l5pc.py
"""

import pickle

# Scipy
import numpy as np
from matplotlib import pyplot as plt

import bluepyopt.ephys as ephys

def plot_proto_responses(proto_responses):
	"""
	Plot responses stored in a dictionary.

	@param	proto_responses		dict {protocol_name: responses_dict}
	"""
	for proto_name, responses in proto_responses.iteritems():
		
		fig, axes = plt.subplots(len(responses))
		try:
			iter(axes)
		except TypeError:
			axes = [axes]

		for index, (resp_name, response) in enumerate(sorted(responses.items())):
			axes[index].plot(response['time'], response['voltage'], label=resp_name)
			axes[index].set_title(resp_name)
		
		fig.tight_layout()
	
	plt.show(block=False)


def plot_responses(responses):
	"""
	Plot response dictionary for a protocol

	@param	proto_responses		dict {str: responses.TimeVoltageResponse}
	"""
		
	fig, axes = plt.subplots(len(responses))
	try:
		iter(axes)
	except TypeError:
		axes = [axes]

	for index, (resp_name, response) in enumerate(sorted(responses.items())):
		axes[index].plot(response['time'], response['voltage'], label=resp_name)
		axes[index].set_title(resp_name)
	
	fig.tight_layout()
	
	plt.show(block=False)


def save_proto_responses(responses, filepath):
	"""
	Save protocol responses to file.
	"""

	# Save to file
	with open(filepath, 'w') as recfile:
		pickle.dump(responses, recfile)

	print("Saved responses to file {}".format(filepath))


def load_proto_responses(filepath):
	"""
	Load protocol responses from pickle file.

	@return		dictionary {protocol: responses}
	"""
	with open(filepath, 'r') as recfile:
		responses = pickle.load(recfile)
		return responses


def run_proto_responses(cell_model, ephys_protocols):
	"""
	Run protocols using given cell model and return responses,
	indexed by protocol.name.
	"""
	nrnsim = ephys.simulators.NrnSimulator(dt=0.025, cvode_active=False)

	# Run each protocol and get its responses
	all_responses = {}
	for e_proto in ephys_protocols:

		response = e_proto.run(
						cell_model		= cell_model, 
						param_values	= {},
						sim				= nrnsim,
						isolate			= False)

		all_responses[e_proto.name] = response

	return all_responses

def plot_log(log):
	"""
	Plot logbook
	"""

	fig, axes = plt.subplots(facecolor='white')

	gen_numbers = log.select('gen')
	mean = np.array(log.select('avg'))
	std = np.array(log.select('std'))
	minimum = log.select('min')
	maximum = log.select('max')

	stdminus = mean - std
	stdplus = mean + std
	axes.plot(
		gen_numbers,
		mean,
		color='black',
		linewidth=2,
		label='population average')

	axes.fill_between(
		gen_numbers,
		stdminus,
		stdplus,
		color='lightgray',
		linewidth=2,
		label=r'population standard deviation')

	axes.plot(
		gen_numbers,
		minimum,
		color='green',
		linewidth=2,
		label='population minimum')

	axes.plot(
		gen_numbers,
		maximum,
		color='red',
		linewidth=2,
		label='population maximum')

	axes.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
	axes.set_xlabel('Generation #')
	axes.set_ylabel('Sum of objectives')
	axes.set_ylim([0, max(maximum)])
	axes.legend()

	fig.tight_layout()


def plot_fitness_scores(objectives, box=None):
	"""
	Plot objectives of the cell model

	@param objectives	dict { (str) objective_name: (float) objective_score}

	USAGE
		reponses = plot_responses(protocols)
		objectives = optimisation.evaluator.fitness_calculator.calculate_scores(responses)
		plot_fitness_scores(objectives)
	"""

	import collections
	objectives = collections.OrderedDict(sorted(objectives.iteritems()))

	fig, axes = plt.subplots(facecolor='white')
	
	# left_margin = box['width'] * 0.4
	# right_margin = box['width'] * 0.05
	# top_margin = box['height'] * 0.05
	# bottom_margin = box['height'] * 0.1

	# axes = fig.add_axes(
	# 	(box['left'] + left_margin,
	# 	 box['bottom'] + bottom_margin,
	# 	 box['width'] - left_margin - right_margin,
	# 	 box['height'] - bottom_margin - top_margin))

	ytick_pos = [x + 0.5 for x in range(len(objectives.keys()))]

	axes.barh(ytick_pos,
			  objectives.values(),
			  height=0.5,
			  align='center',
			  color='#779ECB')
	axes.set_yticks(ytick_pos)
	axes.set_yticklabels(objectives.keys(), size='x-small')
	axes.set_ylim(-0.5, len(objectives.values()) + 0.5)
	axes.set_xlabel('Objective value (# std)')
	axes.set_ylabel('Objectives')


def plot_history(history):
	"""
	Plot the history of the individuals
	"""

	import networkx
	import matplotlib.pyplot as plt

	plt.figure()

	graph = networkx.DiGraph(history.genealogy_tree)
	graph = graph.reverse()     # Make the grah top-down
	# colors = [\
	#        toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
	positions = networkx.graphviz_layout(graph, prog="dot")
	networkx.draw(graph, positions)


def plot_individual_params(
		opt,
		ax,
		params,
		marker,
		color,
		markersize=40,
		plot_bounds=False,
		fitness_cut_off=None): 
	'''
	plot the individual parameter values
	'''
	observations_count = len(params)
	param_count = len(params[0])

	results = np.zeros((observations_count, param_count))
	good_fitness = 0
	for i, param in enumerate(params):
		if fitness_cut_off < max(param.fitness.values):
			continue
		results[good_fitness] = param
		good_fitness += 1

	results = results

	for c in range(good_fitness):
		x = np.arange(param_count)
		y = results[c, :]
		ax.scatter(x=x, y=y, s=float(markersize), marker=marker, color=color)

	if plot_bounds:
		def plot_tick(column, y):
			col_width = 0.25
			x = [column - col_width,
				 column + col_width]
			y = [y, y]
			ax.plot(x, y, color='black')

		# plot min and max
		for i, parameter in enumerate(opt.evaluator.params):
			min_value = parameter.lower_bound
			max_value = parameter.upper_bound
			plot_tick(i, min_value)
			plot_tick(i, max_value)


def plot_diversity(opt, checkpoint_file, param_names):
	'''
	plot the whole history, the hall of fame, and the best individual
	from a unpickled checkpoint
	'''
	checkpoint = pickle.load(open(checkpoint_file, "r"))

	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(1, 1, 1)

	import copy
	best_individual = copy.deepcopy(checkpoint['halloffame'][0])

	# for index, param_name in enumerate(opt.evaluator.param_names):
	# 	best_individual[index] = release_params[param_name]
	
	plot_individual_params(
		opt,
		ax,
		checkpoint['history'].genealogy_history.values(),
		marker='.',
		color='grey',
		plot_bounds=True) 
	
	plot_individual_params(opt, ax, checkpoint['halloffame'],
						   marker='o', color='black')
	
	plot_individual_params(opt,
						   ax,
						   [checkpoint['halloffame'][0]],
						   markersize=150,
						   marker='x',
						   color='blue') 
	
	plot_individual_params(opt, ax, [best_individual], markersize=150,
						   marker='x', color='red')

	labels = [name.replace('.', '\n') for name in param_names]

	param_count = len(checkpoint['halloffame'][0])
	x = range(param_count)
	for xline in x:
		ax.axvline(xline, linewidth=1, color='grey', linestyle=':')

	plt.xticks(x, labels, rotation=80, ha='center', size='small')
	ax.set_xlabel('Parameter names')
	ax.set_ylabel('Parameter values')
	ax.set_yscale('log')
	ax.set_ylim(bottom=1e-7)

	plt.tight_layout()
	plt.plot()
	ax.set_autoscalex_on(True)