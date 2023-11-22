"""
Generate simulation configuration files from a template.

USAGE
-----

The script does not work with input arguments. Just modify the parameters
directly in the script, by looking for lines marked with 'SETPARAM'.

>>> python generate_configs.py

To generate a 2D sweep of two independent variables:
- generate configs for sweep of one variable
- use these as inputs for the next sweep: paste in 'template_paths'
"""
import json, os.path
from bgcellmodels.common import fileutils
import numpy as np

# SETPARAM: template file and output directory
template_paths = """
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-3.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-6.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-9.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-12.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-15.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-18.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-21.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-24.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-27.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-30.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-33.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-36.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-39.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-42.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-45.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-48.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-51.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-54.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-57.0-Hz.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweeps_f-burst-input/syn-V18_f-burst-60.0-Hz.json
""".strip().split()

# SETPARAM: output dir
outdir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetSgRedux/configs/sweeps_f-burst-ctx"

for template_path in template_paths:

    template_dir, template_name = os.path.split(template_path)
    config = fileutils.parse_json_file(template_path, nonstrict=True, ordered=True)


    # Replace all occurrences of format keywords
    # SETPARAM: substitutions as multiplication/addition/replacement
    substitutions = {
    }

    # SETPARAM: format string for json filename
    suffix_replaced = '.json'
    suffix_format = '.json'

    removals = [
        ('STN', 'traces', 0, 'specs', 'v_dend1_dist{:d}'),
        ('STN', 'traces', 0, 'specs', 'v_dend0_dist{:d}'),
        ('STN', 'traces', 0, 'specs', 'STN_cai'),
        ('STN', 'traces', 0, 'specs', 'STN_CaT_inact_fast'),
        ('STN', 'traces', 0, 'specs', 'STN_CaT_inact_slow'),
        ('STN', 'traces', 0, 'specs', 'STN_CaT_open'),
        ('STN', 'traces', 0, 'specs', 'STN_CaL_inact'),
        ('STN', 'traces', 0, 'specs', 'STN_CaL_open'),
    ]

    # remove config entries
    for nested_keys in removals:
        parent_dict = config
        for k in nested_keys[:-1]:
            parent_dict = parent_dict[k]

        removed_key = nested_keys[-1]
        if removed_key in parent_dict:
            del parent_dict[removed_key]

    # If no substitutions: only write one copy
    if len(substitutions) == 0:
        # SETPARAM: config filename substitution
        outname = template_name.replace(suffix_replaced, suffix_format)
        outfile = os.path.join(outdir, outname)
        
        with open(outfile, 'w') as f:
            json.dump(config, f, indent=4)

        print("Wrote config file {}".format(outfile))
        sweep_length = 0

    else:
        suffix_substitutions = substitutions.values()[0] # SETPARAM: file name substitutions
        sweep_length = len(suffix_substitutions)

    # Do substitutions
    for i in range(sweep_length):

        # Update each config parameter (nested dictionary) with sweep variable
        for nested_keys, sweep_vals in substitutions.items():
            assert len(sweep_vals) == sweep_length
            parent_dict = config
            for k in nested_keys[:-1]:
                if k not in parent_dict:
                    parent_dict[k] = {}
                parent_dict = parent_dict[k]
            sweep_param = nested_keys[-1]

            # SETPARAM: uncomment line if you want to check the modified entry exists
            if sweep_param not in parent_dict:
                raise ValueError("Key {} not present at nesting level {}".format(
                    sweep_param, nested_keys))
            parent_dict[sweep_param] = sweep_vals[i]
            print("Updated key {} for sweep {}".format(sweep_param, i))

        # Write config after doing all substitutions for current sweep value
        # SETPARAM: config filename substitution
        outname = template_name.replace(suffix_replaced,
                        suffix_format.format(suffix_substitutions[i]))
        outfile = os.path.join(outdir, outname)
        
        with open(outfile, 'w') as f:
            json.dump(config, f, indent=4)

        # Write unicode (encoding='utf-8')
        # import io
        # with io.open(filename, 'w', encoding=encoding) as f:
        #     f.write(json.dumps(morph_dicts, ensure_ascii=False))

        print("Wrote config file {}".format(outfile))
