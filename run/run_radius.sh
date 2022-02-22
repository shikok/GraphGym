##!/usr/bin/env bash
#
REPEAT=4
MAX_JOBS=8
SLEEP=1
IMC_PATH=C:\/Users\/shayshi\/Documents\/IMCcode
for CNUM in c1 c2
do
  CONFIG=$CNUM'_radius'
  for RADIUS in 2 5 10 20 35 50 100
  do
    GRID=$CNUM'_radius_'$RADIUS
    # generate configs (after controlling computational budget)
    # please remove --config_budget, if don't control computational budget
    python -m GraphGym.run.configs_gen --config ${IMC_PATH}/GraphGym/run/configs/${CONFIG}.yaml \
      --config_budget ${IMC_PATH}/GraphGym/run/configs/${CONFIG}.yaml \
      --grid ${IMC_PATH}/GraphGym/run/grids/radius/${GRID}.txt \
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
  done
done