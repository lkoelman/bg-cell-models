"""
Subthalamic nucleus (STN) cell models.

@author     miscellaneous, see individual models
"""
# override modules to export
# __all__ = ["module_a", "module_b", "module_c"]

# make classes available at package level
# from . import submodule as submodule_alias
# from .submodule import myclass

import os
from neuron import h

pkg_dir = os.path.abspath(os.path.dirname(__file__))

templates = {
    "STN_morph_arcdist": "stn_proto_arcdist.hoc",
    "STN_morph_cartdist": "stn_proto_arcdist.hoc",
    "STN_morph_type1RD": "stn_proto_type1RD.hoc",
}

def load_template(template_name):
    """
    Load Hoc code with template definition.
    """
    if template_name not in templates:
        raise ValueError(
            "Unknown template: {}. Available templates are: {}".format(
                template_name, ", ".join(templates.keys())))

    # Load Hoc template
    prev_wd = os.getcwd()
    os.chdir(pkg_dir)
    h.xopen(templates[template_name])
    os.chdir(prev_wd)
