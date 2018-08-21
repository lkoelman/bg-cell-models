"""
Generate simulation configuration files from a template.

USAGE
-----

The script does not work with input arguments. Just modify the parameters
directly in the script, by looking for lines marked with 'SETPARAM'.

>>> python generate_configs.py
"""
import json, os.path
from bgcellmodels.common import fileutils

# SETPARAM: template file and output directory
template_paths = [
    "../configs/q13_sweep-str-gpe-gsyn/DA-depleted-v3_CTX-f0_STR-GPE-gsyn-x2.0.json",
]

for template_path in template_paths:

    template_dir, template_name = os.path.split(template_path)
    outdir = "../configs/q13_sweep-str-gpe-gsyn" # SETPARAM: output dir
    config = fileutils.parse_json_file(template_path, nonstrict=True, ordered=True)

    # SETPARAM: substitutions
    factors = [0.1, 0.333, 0.5, 1.5, 2.0, 3.0, 5.0]
    # ampa_base = config['STN']['STN']['synapse']['parameters']['gmax_AMPA']
    # nmda_base = config['STN']['STN']['synapse']['parameters']['gmax_NMDA']

    # Replace all occurrences of format keywords
    substitutions = {
        # ('STN', 'PyNN_cell_parameters', 'tau_m_scale'): factors,
        # ('GPE', 'PyNN_cell_parameters', 'tau_m_scale'): factors,
        ('GPE', 'STR', 'synapse', 'parameters', 'gmax_GABAA'): [1e-3*f for f in factors],
    }
    suffix_format = '-{}'
    suffix_substitutions = factors
    sweep_length = len(suffix_substitutions)

    for i in range(sweep_length):

        # Update each config parameter (nested dictionary) with sweep variable
        for nested_keys, sweep_vals in substitutions.items():
            assert len(sweep_vals) == sweep_length
            parent_dict = config
            for k in nested_keys[:-1]:
                if k not in parent_dict:
                    parent_dict[k] = {}
                parent_dict =  parent_dict[k]
            sweep_param = nested_keys[-1]

            # SETPARAM: uncomment line if you want to check the modified entry exists
            # if sweep_param not in parent_dict:
            #     raise ValueError("Key {} not present at nesting level {}".format(
            #         sweep_param, nested_keys))
            parent_dict[sweep_param] = sweep_vals[i]
            print("Updated key {} for sweep {}".format(sweep_param, i))

        # Write config after doing all substitutions for current sweep value
        # SETPARAM: config filename substitution
        # outname = template_name.replace('template',
        #                             suffix_format.format(suffix_substitutions[i]))
        outname = template_name.replace('.json',
                                    suffix_format.format(suffix_substitutions[i]) + '.json')
        outfile = os.path.join(outdir, outname)
        
        with open(outfile, 'w') as f:
            json.dump(config, f, indent=4)

        # Write unicode (encoding='utf-8')
        # import io
        # with io.open(filename, 'w', encoding=encoding) as f:
        #     f.write(json.dumps(morph_dicts, ensure_ascii=False))

        print("Wrote config file {}".format(outfile))
