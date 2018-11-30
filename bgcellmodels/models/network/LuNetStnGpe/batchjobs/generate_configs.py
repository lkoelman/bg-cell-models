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
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-0.33.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-0.67.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-1.00.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-1.33.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-1.67.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-2.00.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-2.33.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-2.67.json
/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/sweep_gpe-stn_weight/StnGpe_template_syn-V18_gpe-stn_x-3.00.json
""".strip().split()

for template_path in template_paths:

    template_dir, template_name = os.path.split(template_path)
    outdir = "../configs/sweep_gpe-stn_weight" # SETPARAM: output dir
    config = fileutils.parse_json_file(template_path, nonstrict=True, ordered=True)

    # SETPARAM: substitutions
    factors = np.arange(1./3, 3.0+1./3, 1./3)
    # gs_gabaa = config['STN']['GPE.all']['synapse']['parameters'][
    #                   'gmax_GABAA']['locals']['gmax_base']
    # gs_gabab = config['STN']['GPE.all']['synapse']['parameters'][
    #                   'gmax_GABAB']['locals']['gmax_base']
    # cs_ampa = config['STN']['CTX']['synapse']['parameters'][
    #                   'GLUsyn_gmax_AMPA']['locals']['gmax_base']
    # cs_nmda_dend = config['STN']['CTX']['synapse']['parameters'][
    #                   'GLUsyn_gmax_NMDA']['locals']['gmax_base']
    # cs_nmda_soma = config['STN']['CTX']['synapse']['parameters'][
    #                   'NMDAsynTM_gmax_NMDA']['locals']['gmax_base']
    gg_gabaa = config['GPE.proto']['GPE.all']['synapse']['parameters'][
                      'gmax_GABAA']['locals']['gmax_base']
    gg_gabab = config['GPE.proto']['GPE.all']['synapse']['parameters'][
                      'gmax_GABAB']['locals']['gmax_base']

    # Replace all occurrences of format keywords
    substitutions = {
        # ('STN', 'GPE.all', 'synapse', 'parameters', 'gmax_GABAA', 'locals', 
        #     'gmax_base'): [f*gs_gabaa for f in factors],
        # ('STN', 'GPE.all', 'synapse', 'parameters', 'gmax_GABAB', 'locals', 
        #     'gmax_base'): [f*gs_gabab for f in factors],
        # ('STN', 'CTX', 'synapse', 'parameters', 'GLUsyn_gmax_AMPA', 'locals', 
        #     'gmax_base'): [f*cs_ampa for f in factors],
        # ('STN', 'CTX', 'synapse', 'parameters', 'GLUsyn_gmax_NMDA', 'locals', 
        #     'gmax_base'): [f*cs_nmda_dend for f in factors],
        # ('STN', 'CTX', 'synapse', 'parameters', 'NMDAsynTM_gmax_NMDA', 'locals', 
        #     'gmax_base'): [f*cs_nmda_soma for f in factors],
        ('GPE.proto', 'GPE.all', 'synapse', 'parameters', 'gmax_GABAA', 'locals', 
            'gmax_base'): [f*gg_gabaa for f in factors],
        ('GPE.proto', 'GPE.all', 'synapse', 'parameters', 'gmax_GABAB', 'locals', 
            'gmax_base'): [f*gg_gabab for f in factors],
    }
    suffix_format = '_gpe-gpe_x-{:.2f}' # SETPARAM: format string for json filename
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
                parent_dict = parent_dict[k]
            sweep_param = nested_keys[-1]

            # SETPARAM: uncomment line if you want to check the modified entry exists
            if sweep_param not in parent_dict:
                raise ValueError("Key {} not present at nesting level {}".format(
                    sweep_param, nested_keys))
            # SETPARAM: substitution or multiplication of target value
            parent_dict[sweep_param] = sweep_vals[i]
            # parent_dict[sweep_param] *= sweep_vals[i]
            print("Updated key {} for sweep {}".format(sweep_param, i))

        # Write config after doing all substitutions for current sweep value
        # SETPARAM: config filename substitution
        outname = template_name.replace('template_', '').replace('.json',
                        suffix_format.format(suffix_substitutions[i]) + '.json')
        outfile = os.path.join(outdir, outname)
        
        with open(outfile, 'w') as f:
            json.dump(config, f, indent=4)

        # Write unicode (encoding='utf-8')
        # import io
        # with io.open(filename, 'w', encoding=encoding) as f:
        #     f.write(json.dumps(morph_dicts, ensure_ascii=False))

        print("Wrote config file {}".format(outfile))
