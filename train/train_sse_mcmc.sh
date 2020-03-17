DATASET=$1
ARCH=$2
ITER=$3
BASE_DIR=$4
DATA_PATH=$5
METHOD=$6

if [ $3 == "SSE" ];
then
  INJECT_NOISE_OR_NOT=""
  MAX_LR=0.2
  CYCLE_EPOCHS=40
  CYCLE_SAVES=1
  # save more snapshots per cycle for possible analysis
#   CYCLE_SAVES=5
  NOISE_EPOCHS=0
  CYCLES=100
else
  INJECT_NOISE_OR_NOT="--inject_noise"
  MAX_LR=0.5
  CYCLE_EPOCHS=50
  CYCLE_SAVES=3
  CYCLES=34
  NOISE_EPOCHS=3
fi

if [ $2 == "PreResNet110" ] || [ "$2" == "PreResNet164" ];
then
  WD=3e-4
else
  WD=5e-4
fi

python train/cifar/sse_mcmc_train.py $INJECT_NOISE_OR_NOT \
    --dir="${BASE_DIR}/${METHOD,,}/${DATASET}_${ARCH}_it_${ITER}" \
    --model="$ARCH" --dataset="$DATASET" --noise_epochs=$NOISE_EPOCHS --data_path="$DATA_PATH" \
    --alpha=1 --cycles=$CYCLES --iter="$ITER" \
    --cycle_epochs=$CYCLE_EPOCHS --cycle_saves=$CYCLE_SAVES --max_lr=$MAX_LR --wd=$WD
