#!/bin/bash -l

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho"
nb_filename="synchrony_analysis_auto.ipynb"
nb_dir="${model_dir}/analysis"
notebook="${nb_dir}/${nb_filename}"
logfile="${nb_dir}/nb_exec_list.log"
conffile="${nb_dir}/nb_exec_conf.py"

# List simulation output directories to analyze
output_dirs=( \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780629.sonic-head_DA-depleted-v3_CTX-favg14_fburst5" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780630.sonic-head_DA-depleted-v3_CTX-favg14_fburst7" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780631.sonic-head_DA-depleted-v3_CTX-favg14_fburst9" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780632.sonic-head_DA-depleted-v3_CTX-favg14_fburst11" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780633.sonic-head_DA-depleted-v3_CTX-favg14_fburst13" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780634.sonic-head_DA-depleted-v3_CTX-favg14_fburst15" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780635.sonic-head_DA-depleted-v3_CTX-favg14_fburst17" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780636.sonic-head_DA-depleted-v3_CTX-favg14_fburst19" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780637.sonic-head_DA-depleted-v3_CTX-favg14_fburst21" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780638.sonic-head_DA-depleted-v3_CTX-favg14_fburst23" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780639.sonic-head_DA-depleted-v3_CTX-favg14_fburst25" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780640.sonic-head_DA-depleted-v3_CTX-favg14_fburst27" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780641.sonic-head_DA-depleted-v3_CTX-favg14_fburst29" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780642.sonic-head_DA-depleted-v3_CTX-favg14_fburst31" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q1_const-rate_vary-freq/2018.08.02_job-780643.sonic-head_DA-depleted-v3_CTX-favg14_fburst50" \
)

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
sweep_name="f_burst_ctx"
sweep=(5 7 9 11 13 15 17 19 21 23 25 27 29 31 50)

cd $nb_dir

# for outdir in "${output_dirs[@]}"; do
for ((i=0;i<num_outputs;i++)); do
    outdir=${output_dirs[i]}
    hpfreq=${hpfreqs[i]}
    lpfreq=${lpfreqs[i]}

    # Don't run notebook if exists
    outfile="${outdir}/${nb_filename}"
    if [ -e ${outfile} ]; then
        echo -e "Skipping ${outdir}"
        continue
    fi

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
    nb_save="jupyter nbconvert ${outfile} --template=toc2 --output-dir=$outdir"
    eval $nb_save
done

echo "Finished executing notebooks!"