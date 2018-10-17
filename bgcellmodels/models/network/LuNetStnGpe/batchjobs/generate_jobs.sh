#!/bin/bash -l

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe"
job_script="${model_dir}/batchjobs/runjob_bgnetmodel.sh"

# Config files you want to repeat with different seeds
# configs=( \
#     "config.json" \
# )
outputs_clipboard="/home/luye/workspace/bgcellmodels/bgcellmodels/models/network/LuNetStnGpe/configs/template_net-full_syn-V8.json"
readarray -t configs <<< "${outputs_clipboard}"

start_seed=777

for conf in "${configs[@]}"; do
    for seed in {0..0}; do
        qsub_command="qsub ${job_script} \
-l walltime=1:00:00 \
-v seed=$((start_seed+seed)),config=${conf},\
dur=10e3,transient-period=0.0,write-interval=10e3"

        echo -e "Submitting qsub command:\n> $qsub_command"
        eval $qsub_command
    done
# for fburst in {5,10,20,25,50,100}; do
#         qsub_command="qsub ${job_script} \
# -l walltime=02:30:00 \
# -v ncell=100,dur=26000,burst=${fburst},config=${conf}"

#         echo -e "Submitting qsub command:\n> $qsub_command"
#         eval $qsub_command
#     done
done

echo -n "Waiting for 5 seconds to check job status"
for i in {1..5}; do
    echo -n "."
    sleep 1
done
echo ""

qstat
