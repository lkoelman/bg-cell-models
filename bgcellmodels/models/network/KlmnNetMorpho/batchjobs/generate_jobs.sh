#!/bin/bash -l

model_dir="${HOME}/workspace/bgcellmodels/bgcellmodels/models/network/KlmnNetMorpho"
job_script="${model_dir}/batchjobs/runjob_bgnetmodel.sh"

# Config files you want to repeat with different seeds
# configs=( \
#     "config.json" \
# )
outputs_clipboard="q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x0.0.json
q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x0.5.json
q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x0.25.json
q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x0.75.json
q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x1.5.json
q6_sweep_stn-stn-strength/DA-depleted-v3_CTX-f0__STN-STN-gsyn-x2.0.json"
readarray -t configs <<< "${outputs_clipboard}"

start_seed=888

for conf in "${configs[@]}"; do
    for seed in {0..0}; do
        qsub_command="qsub ${job_script} \
-l walltime=2:30:00 \
-v ncell=100,dur=26000,seed=$((start_seed+seed)),config=${conf}"

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
