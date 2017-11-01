"""
Utilities for connecting neurons.

@author Lucas Koelman
@date	30-10-2017
"""

import re

def interpretParamSpec(spec):
	"""
	Extract <mechanism_type>, <parameter_name>, <index> from parameter specification in format 'mechanism:parameter[index]'

	@return		tuple (mechanism_type, parameter_name, index)

					mechanism_type: <str> 'syn', 'netcon'
					parameter_name: <str> attribute name on Hoc object
					index:			<int> index of attribute on Hoc object
	"""
	# Regular expression with ?P<groupname> to mark named groups
	matches = re.search(r'^(?P<mech>\w+):(?P<parname>\w+)(\[(?P<idx>\d+)\])?', spec)
	
	mech_type = matches.group('mech')
	mech_param = matches.group('parname')
	param_index = matches.group('idx')
	
	return mech_type, mech_param, param_index