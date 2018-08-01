#!/bin/bash -l

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho"
nb_filename="synchrony_analysis_auto.ipynb"
nb_dir="${model_dir}/analysis"
notebook="${nb_dir}/${nb_filename}"
logfile="${nb_dir}/nb_exec_list.log"
conffile="${nb_dir}/nb_exec_conf.py"

# Config files you want to repeat with different seeds
output_dirs=( \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779545.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779546.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779547.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779548.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779549.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
    "/run/media/luye/Windows7_OS/Users/lkoelman/simdata-win/syn-v2_q3_const-freq_vary-burst-frac/2018.07.24_job-779550.sonic-head_DA-depleted-v3_CTX-burst-frac02" \
)
num_outputs=${#output_dirs[@]}
hpfreqs=(2  5  15  20  20  20)
lpfreqs=(15 15 30  30  30  30)

cd $nb_dir

# for outdir in "${output_dirs[@]}"; do
for ((i=0;i<num_outputs;i++)); do
    outdir=${output_dirs[i]}
    hpfreq=${hpfreqs[i]}
    lpfreq=${lpfreqs[i]}

    # notebook will read variables from configuration script
    echo $outdir >> $logfile
    pyscript="outputs = \"${outdir}\"
hpfreq = ${hpfreq}
lpfreq = ${lpfreq}"
    echo -e "${pyscript}" > ${conffile} # (quote preserves newlines)

    # Execute notebook
    nb_exec="jupyter nbconvert --execute --to notebook ${notebook} --output-dir=${outdir}"
    echo -e "Executing notebook for output: ${outdir}"
    eval $nb_exec

    outfile="${outdir}/${nb_filename}"
    echo -e "Finished! Saving notebook to output directory ..."
    nb_save="jupyter nbconvert ${outfile} --template=toc2 --output-dir=$outdir"
    eval $nb_save
done

echo "Finished executing notebooks!"