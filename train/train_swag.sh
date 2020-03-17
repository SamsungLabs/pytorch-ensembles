DATASET=$1
ARCH=$2
ITER=$3
BASE_DIR=$4
DATA_PATH=$5

if [ $2 == "VGG16BN" ];
then
  LR_INIT=0.05
  LR_SWA=0.01
  WD=5e-4
  TRANSFORM=VGG
elif [ $2 == "WideResNet28x10" ]; then
  LR_INIT=0.1
  LR_SWA=0.01
  WD=5e-4
  TRANSFORM=ResNet
else
  LR_INIT=0.1
  if [ $1 == "CIFAR100" ];
  then
    LR_SWA=0.05
  else
    LR_SWA=0.01
  fi
  WD=3e-4
  TRANSFORM=ResNet
fi

python train/cifar/swag_train.py --data_path="$DATA_PATH" --epochs=300 --dataset="$DATASET" --save_freq=25 --model="$ARCH" \
--lr_init=$LR_INIT --wd=$WD --swa_start=160 --swa_lr=$LR_SWA --cov_mat --dir="${BASE_DIR}/swag/${DATASET}_${ARCH}_it_${ITER}" --max_num_models=20 \
--transform=$TRANSFORM
python train/cifar/swag_sample.py --data_path="$DATA_PATH" --dataset="$DATASET" --model="$ARCH" --cov_mat \
--method=SWAG --scale=0.5 --file="${BASE_DIR}/swag/${DATASET}_${ARCH}_it_${ITER}/swag_ensembled-300.pt" \
--N=100 --max_num_models=20 --all_sample_nums --iter=$ITER --transform=$TRANSFORM
