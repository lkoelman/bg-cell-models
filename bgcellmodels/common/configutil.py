"""
Utilities for connecting neurons.

@author Lucas Koelman
@date   30-10-2017
"""

import re
from bgcellmodels.common.stdutil import eval_context


def destructure_param_spec(spec):
    """
    Extract <mechanism_type>, <parameter_name>, <index> 
    from parameter specification in format 'mechanism:parameter[index]'

    @param      spec : str

                A string in the format "mechtype.paramname[index]" where the
                index part is optional. The only requirement is to have
                two substring separated by "." and possibly followed by the
                index in brackets.


    @return     tuple(mechanism_type: str, parameter_name: str, index: int)
                
                Tuple corresponding to the three parts of the parameter spec,
                with the index equal to None of not specified.
    """
    # Regular expression with ?P<groupname> to mark named groups
    matches = re.search(r'^(?P<mech>\w+):(?P<parname>\w+)(\[(?P<idx>\d+)\])?', spec)
    
    mech_type = matches.group('mech')
    mech_param = matches.group('parname')
    param_index = matches.group('idx')
    
    return mech_type, mech_param, param_index

# deprecated name
interpretParamSpec = destructure_param_spec


def eval_params(param_dict, caller_globals, caller_locals):
    """
    Evaluate parameters in dictionary.

    @param      caller_globals : dict
                Gobals dictionary that will be used if a parameter specification
                has its "globals" entry set to None.

    @param      caller_locals : dict
                Locals dictionary that will be used if a parameter specification
                has its "locals" entry set to None.

    @return     evaluated_params : dict
                Same as param_dict but each parameter specification is evaluated.
    """
    params_evaluated = {}
    for param_name, param_spec in param_dict.iteritems():

        if isinstance(param_spec, dict):
            # Parameter specification is a statement to be avaluated in
            # a context of local and global variables
            if "statement" in param_spec:
                params_evaluated[param_name] = eval_context(
                    caller_globals=caller_globals,
                    caller_locals=caller_locals,
                    **param_spec)
            else:
                params_evaluated[param_name] = param_spec

        elif isinstance(param_spec, (float, int, bool)) or param_spec is None:
            # Numerical values are passed as is
            params_evaluated[param_name] = param_spec

        elif isinstance(param_spec, (str, unicode)):
            # String values are passed as is
            params_evaluated[param_name] = param_spec

        elif isinstance(param_spec, list):
            # List values are passed as is
            params_evaluated[param_name] = param_spec
        
        else:
            raise ValueError("Unexpected parameter specification {} "
                             "for parameter '{}'".format(param_spec, param_name))
    return params_evaluated