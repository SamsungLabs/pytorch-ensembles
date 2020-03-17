#!/usr/bin/env bash

START=1
END=1
ARCH=VGG16BN
DATASET=CIFAR10
DATAPATH=../data
DIR=../megares
WORKER=$(hostname)$CUDA_VISIBLE_DEVICES
TIME=$(date +%s)
EPOCHS=90
BETA=0.0001
METHOD=regular

while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
	--start)
		START="$2"
		shift
		shift
		;;
	--end)
		END="$2"
		shift
		shift
		;;
	--arch)
		ARCH="$2"
		shift
		shift
		;;
	--dataset)
		DATASET="$2"
		shift
		shift
		;;
	--datapath)
		DATAPATH="$2"
		shift
		shift
		;;
	--dir)
		DIR="$2"
		shift
		shift
		;;
  --epochs)
		EPOCHS="$2"
		shift
		shift
		;;
  --beta)
		BETA="$2"
		shift
		shift
		;;
	--worker)
		WORKER="$2"
		shift
		shift
		;;
  --method)
      METHOD="$2"
      shift
      shift
      ;;
	*)
		echo "Unrecognized argument $1"
		exit 1
		;;
	esac
done

if [[ $ARCH == *VGG* ]]
then
	LR=0.05
	EPOCHS=400
	WD=5e-4
elif [[ $ARCH == *WideResNet28x10do* ]]
then
	LR=0.1
	EPOCHS=400
	WD=5e-4
elif [[ $ARCH == *Wide* ]]
then
	LR=0.1
	EPOCHS=300
	WD=5e-4
	BETA=0.00001
else
	LR=0.1
	EPOCHS=300
	WD=3e-4
fi

if [[ $METHOD == vi ]]
then
    ARGS="train/cifar/train_vi.py --dir=$DIR --data_path=$DATAPATH --eval_freq 10 --dataset=$DATASET --model=Bayes$ARCH --epochs 100 --wd=$WD --beta $BETA --fname=$DATASET-$ARCH-$WORKER-$TIME"

    for i in $(seq $START $END)
    do
        echo ipython -- $ARGS-$i
        ipython -- $ARGS-$i
    done
elif [[ $METHOD == regular ]]
then
    ARGS="./train/cifar/train.py --dir=$DIR --dataset=$DATASET --data_path=$DATAPATH --epochs=$EPOCHS --model=$ARCH --lr_init=$LR --wd=$WD --fname=$DATASET-$ARCH-$WORKER-$TIME"

    for i in $(seq $START $END)
    do
        echo ipython --  $ARGS-$TIME-$i
        ipython -- $ARGS-$TIME-$i
    done
else
  echo "unknown method $METHOD, avalible options are regular and vi"
fi