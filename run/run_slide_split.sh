##!/usr/bin/env bash
#
CONFIG=c1_red
GRID=split_grid_c1
REPEAT=4
MAX_JOBS=8
SLEEP=1
IMC_PATH=C:\/Users\/shayshi\/Documents\/IMCcode
# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python -m GraphGym.run.configs_gen --config ${IMC_PATH}/GraphGym/run/configs/${CONFIG}.yaml \
  --config_budget ${IMC_PATH}/GraphGym/run/configs/${CONFIG}.yaml \
  --grid ${IMC_PATH}/GraphGym/run/grids/${GRID}.txt \
  --out_dir ${IMC_PATH}/GraphGym/run/configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash ${IMC_PATH}/GraphGym/run/parallel.sh ${IMC_PATH}/GraphGym/run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash ${IMC_PATH}/GraphGym/run/parallel.sh ${IMC_PATH}/GraphGym/run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
# rerun missed / stopped experiments
bash ${IMC_PATH}/GraphGym/run/parallel.sh ${IMC_PATH}/GraphGym/run/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP

# aggregate results for the batch
python -m GraphGym.run.agg_batch --dir ${IMC_PATH}/results/${CONFIG}_grid_${GRID}
