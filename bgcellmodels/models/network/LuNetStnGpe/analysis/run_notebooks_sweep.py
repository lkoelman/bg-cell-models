#!/usr/bin/env python

"""
Automatically run a notebook with each simulation output.
"""

import os
import subprocess
import re

nb_dir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/analysis"
nb_infile = "lunet_stn-gpe_analysis.ipynb"
nb_path = os.path.join(nb_dir, nb_infile)
log_path = os.path.join(nb_dir, "nb_exec_list.log") # change for copies of this script
conf_path = os.path.join(nb_dir, "nb_exec_conf.py") # change for copies of this script


# List simulation output directories to analyze
outputs_clipboard = """
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_22.48.44_job-1184523.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-1.33
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_22.48.44_job-1184524.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-1.67
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_22.51.17_job-1184520.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-0.33
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_22.51.18_job-1184521.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-0.67
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_23.07.00_job-1184522.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-1.00
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_23.07.02_job-1184525.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-2.00
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_23.07.04_job-1184526.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-2.33
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_23.09.54_job-1184527.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-2.67
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q3_sweep_gmax-ctx-stn/LuNetStnGpe_2018.11.19_23.18.14_job-1184528.sonic-head_StnGpe_template_syn-V18_ctx-stn_x-3.00
"""

output_dirs = outputs_clipboard.strip().split()

sweep_name = "gmax_ctx_stn"

for sim_outdir in output_dirs:

    # Variables for the target notebook
    match = re.search(r"x-([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", sim_outdir)
    sweep_val = match.groups()[0]
    nb_pyvars = {
        'outputs': sim_outdir,
        'ROI_INTERVAL': (6e3, 10e3),
        'reference_phase': {'method': 'from_ctx', 'passband': (17.0, 23.0)},
        'sweep_var_name': sweep_name,
        'sweep_var_value': sweep_val,
        'automatic_execution': True,
    }
    # use property eval(repr(object)) == object.
    nb_pyscript= "\n".join(("{} = {}".format(k, repr(v)) for k,v in nb_pyvars.items()))

    with open(conf_path, 'w') as conf_file:
        conf_file.write(nb_pyscript)

    # Add some environment variables to be read from notebook
    env = dict(os.environ)
    env["NB_CONF_FILE"] = conf_path

    # Specify outputs
    job_id = re.search(r'job-(\w+)', sim_outdir).groups()[0]
    ival = nb_pyvars['ROI_INTERVAL']
    out_suffix = '{:.1f}s-{:.1f}s_AUTO'.format(ival[0]/1e3, ival[1]/1e3)
    nb_outfile = 'sweep-{}_job-{}_t-{}.ipynb'.format(sweep_val, job_id, out_suffix)
    nb_out_path = os.path.join(sim_outdir, nb_outfile)

    print("\n{rule}\nPROCESSING: {outdir}\n{rule}\n".format(
          outdir=sim_outdir, rule='-'*80))

    # Execute notebook (optional: --allow-errors)
    nb_exec_cmd = ("jupyter nbconvert --ExecutePreprocessor.timeout=None "
                   "--execute --to notebook {infile} "
                   "--output={outname} "
                   "--output-dir={outdir}").format(infile=nb_path,
                                                   outname=nb_outfile,
                                                   outdir=sim_outdir)

    nb_save_cmd = ("jupyter nbconvert {outfile} --to html --template=toc2 "
                   "--output-dir={outdir}").format(outfile=nb_out_path,
                                                   outdir=sim_outdir)

    # Use subprocess.check_output to raise Exception when command fails
    print("Executing notebook for output: " + sim_outdir)
    exe_status = subprocess.call(nb_exec_cmd, shell=True, env=env)

    print("Finished! Saving notebook to output directory ...")
    sav_status = subprocess.call(nb_save_cmd, shell=True)

    if not (exe_status == sav_status == 0):
        raise Exception('Notebook execution or saving failed for output {}'.format(
                         sim_outdir))


print("Finished executing notebooks.")