#!/usr/bin/env python

"""
Automatically run a notebook with each simulation output.
"""

import os
import subprocess
import re

nb_dir = "/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/analysis"
nb_filename = "lunet_stn-gpe_analysis.ipynb"
nb_path = os.path.join(nb_dir, nb_filename)
log_path = os.path.join(nb_dir, "nb_exec_list.log") # change for copies of this script
conf_path = os.path.join(nb_dir, "nb_exec_conf.py") # change for copies of this script


# List simulation output directories to analyze
outputs_clipboard="""
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/LuNetStnGpe/q1_sweep_gmax-gpe-gpe/LuNetStnGpe_2018.11.19_20.31.44_job-1184502.sonic-head_StnGpe_template_syn-V18_gpe-gpe_x-0.33
"""

output_dirs = outputs_clipboard.strip().split()

sweep_name = "gmax_gpe_gpe"

for sim_outdir in output_dirs:

    # Variables for the target notebook
    match = re.search(r"x-([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", sim_outdir)
    nb_pyvars = {
        'outputs': sim_outdir,
        'ROI_INTERVAL': (1e3, 5e3),
        'reference_phase': {'method': 'from_gpe', 'passband': (6.0, 20.0)},
        'sweep_var_name': sweep_name,
        'sweep_var_value': match.groups()[0],
        'automatic_execution': True,
    }
    # use property eval(repr(object)) == object.
    nb_pyscript= "\n".join(("{} = {}".format(k, repr(v)) for k,v in nb_pyvars.items()))

    with open(conf_path, 'w') as conf_file:
        conf_file.write(nb_pyscript)

    # Add some environment variables to be read from notebook
    env = dict(os.environ)
    env["NB_CONF_FILE"] = conf_path

    # Execute notebook
    nb_out_path = os.path.join(sim_outdir, nb_filename)
    nb_exec_cmd = ("jupyter nbconvert --ExecutePreprocessor.timeout=None "
                   "--execute --to notebook {notebook} "
                   "--output-dir={outdir}").format(notebook=nb_path,
                                                   outdir=sim_outdir)

    nb_save_cmd = ("jupyter nbconvert {outfile} --to html --template=toc2 "
                   "--output-dir={outdir}").format(outfile=nb_out_path,
                                                   outdir=sim_outdir)

    print("Executing notebook for output: " + sim_outdir)
    exe_status = subprocess.call(nb_exec_cmd, shell=True, env=env)

    print("Finished! Saving notebook to output directory ...")
    sav_status = subprocess.call(nb_save_cmd, shell=True)


print("Finished executing notebooks.")