#!/bin/bash
EPOCHS=1000
BATCH_SIZE=16
LR=(0.01 0.005 0.001)
WD=0.0001
SCHEDULER_STEP_SIZE=100
SCHEDULER_GAMMA=0.8
MODEL_TYPE="fno"
N_TRAIN=700
N_TESTS="100"  # Space-separated values for list arguments
LOSSES="l2h1"

# Iterate over the LR array
for lr in "${LR[@]}"; do
  echo "Running training with learning rate: $lr"
  python <train_script_path> \
    --epochs $EPOCHS \
    -b $BATCH_SIZE \
    --lr $lr \
    --wd $WD \
    --scheduler-step-size $SCHEDULER_STEP_SIZE \
    --scheduler-gamma $SCHEDULER_GAMMA \
    --model-type "$MODEL_TYPE" \
    --n-train $N_TRAIN \
    --n-tests $N_TESTS \
    --losses "$LOSSES"
done