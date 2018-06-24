#!/bin/bash -l


model_dir="${HOME}/workspace/bgcellmodels/models/KlmnNetMorpho"
job_script="${model_dir}/batchjobs/runjob_bgnetmodel.sh"

# Config files you want to repeat with different seeds
configs=( \
    "DA-depleted_CTX-beta-sync05_STN-lateral02.json" \
)

for conf in "${configs[@]}"; do
    for seed in {1..5}; do
        qsub_command="qsub ${job_script} \
-l walltime=02:30:00 \
-v ncell=100,dur=22000,seed=${seed},config=${conf}"
        
        echo -e "Submitting qsub command:\n> $qsub_command"
        eval $qsub_command
    done
done