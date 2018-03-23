"""
Utilities for connecting neurons.

@author Lucas Koelman
@date	30-10-2017
"""

import re

def interpretParamSpec(spec):
	"""
	Extract <mechanism_type>, <parameter_name>, <index> from parameter specification in format 'mechanism:parameter[index]'

	@param		spec : str

				A string in the format "mechtype.paramname[index]" where the
				index part is optional. The only requirement is to have
				two substring separated by "." and possibly followed by the
				index in brackets.


	@return		tuple(mechanism_type: str, parameter_name: str, index: int)
				
				Tuple corresponding to the three parts of the parameter spec,
				with the index equal to None of not specified.
	"""
	# Regular expression with ?P<groupname> to mark named groups
	matches = re.search(r'^(?P<mech>\w+):(?P<parname>\w+)(\[(?P<idx>\d+)\])?', spec)
	
	mech_type = matches.group('mech')
	mech_param = matches.group('parname')
	param_index = matches.group('idx')
	
	return mech_type, mech_param, param_index