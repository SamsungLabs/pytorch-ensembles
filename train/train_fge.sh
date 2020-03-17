DATASET=$1
ARCH=$2
ITER=$3
BASE_DIR=$4
DATA_PATH=$5

if [ $ARCH == "VGG16BN" ]; then
  LR_INIT=0.05
  LR_1=1e-2
  LR_2=5e-4
  WD=5e-4
  CYCLE=2
elif [ $ARCH == "WideResNet28x10" ]; then
  LR_INIT=0.1
  LR_1=5e-2
  LR_2=5e-4
  WD=5e-4
  CYCLE=4
else
  LR_INIT=0.1
  LR_1=5e-2
  LR_2=5e-4
  WD=3e-4
  CYCLE=4
fi

EPOCHS=160

python train/cifar/fge_pretrain.py --dir="${BASE_DIR}/fge/${DATASET}_${ARCH}_it_${ITER}" --data_path="$DATA_PATH" --dataset="$DATASET" \
    --transform=VGG --model="$ARCH" --epochs=$EPOCHS --wd=$WD --lr=$LR_INIT --save_freq=40
python train/cifar/fge_train.py --dir="${BASE_DIR}/fge/${DATASET}_${ARCH}_it_${ITER}" --ckpt="${BASE_DIR}/fge/${DATASET}_${ARCH}_it_${ITER}/pretraining-${EPOCHS}.pt" \
    --data_path="$DATA_PATH" --dataset=$DATASET --iter=$ITER \
    --transform=VGG --model="$ARCH" --epochs=$((100*CYCLE+10)) --wd=$WD --lr_1=$LR_1 --lr_2=$LR_2 --cycle=$CYCLE

