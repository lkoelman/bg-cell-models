#!/bin/bash -l

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho"
nb_filename="synchrony_analysis_auto.ipynb"
nb_dir="${model_dir}/analysis"
notebook="${nb_dir}/${nb_filename}"
logfile="${nb_dir}/nb_exec_list.log"
conffile="${nb_dir}/nb_exec_conf.py"
export NB_CONF_FILE=${conffile} # for subprocesses

# List simulation output directories to analyze
outputs_clipboard="/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780651.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x0.2
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780652.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x0.4
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780653.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x0.6
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780654.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x0.8
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780655.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x1.2
/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q5_sweep-gpe-gpe-ipsp/2018.08.02_job-780656.sonic-head_DA-depleted-v3_CTX-f0_GpeGpe-inh-x1.4"
readarray -t output_dirs <<< "${outputs_clipboard}"
# output_dirs=(${outputs_clipboard//$'n'// }) # substitute newlines and make array
# output_dirs=( \
#     "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q7_sweep-transmission-delay/2018.08.03_job-780788.sonic-head_DA-depleted-v3_CTX-f0_StnGpe-d4.0_GpeStn-d6.0" \
# )

# Generate output dirs from parent directory
# parent_dir="/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac"
# dir_contents=$(ls -d -1 ${parent_dir}/**)
# output_dirs=($(echo ${dir_contents})) # echo without quote turns newlines into spaces

num_outputs=${#output_dirs[@]}
# hpfreqs=(2  5  15  20  20  20)
hpfreqs=($(for ((i=0;i<num_outputs;i++)); do echo 5; done))
# lpfreqs=(15 15 30  30  30  30)
lpfreqs=($(for ((i=0;i<num_outputs;i++)); do echo 20; done))

# Parameter sweep
sweep_name="gsyn_gpe_gpe"
sweep=(0.2 0.4 0.6 0.8 1.2 1.4)

cd $nb_dir

# for outdir in "${output_dirs[@]}"; do
for ((i=0;i<num_outputs;i++)); do
    outdir=${output_dirs[i]}
    hpfreq=${hpfreqs[i]}
    lpfreq=${lpfreqs[i]}
    outfile="${outdir}/${nb_filename}"

    # Don't run notebook if exists
    # if [ -e ${outfile} ]; then
    #     echo -e "Skipping ${outdir}"
    #     continue
    # fi

    # notebook will read variables from Python configuration script
    echo $outdir >> $logfile
    pyscript="outputs = \"${outdir}\"
hpfreq = ${hpfreq}
lpfreq = ${lpfreq}
sweep_var_name = \"${sweep_name}\"
sweep_var_value = ${sweep[i]}"
    echo -e "${pyscript}" > ${conffile} # (quote preserves newlines)

    # Execute notebook
    nb_exec="jupyter nbconvert --execute --to notebook ${notebook} --output-dir=${outdir}"
    echo -e "Executing notebook for output: ${outdir}"
    eval $nb_exec

    echo -e "Finished! Saving notebook to output directory ..."
    nb_save="jupyter nbconvert ${outfile} --template=toc2 --output-dir=${outdir}"
    eval $nb_save
done

echo "Finished executing notebooks!"