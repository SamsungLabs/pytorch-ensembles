#!/usr/bin/env bash

START=1
END=1
METHOD=regular
DATAPATH=../imagenet/raw-data
DIR=../megares
WORKER=$(hostname)$CUDA_VISIBLE_DEVICES
TIME=$(date +%s)
DATASET=ImageNet

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

if [[ $METHOD == vi ]]
then
    ARGS="train/imagenet/train_vi_imagenet.py --dir=$DIR --data $DATAPATH --eval_freq 1 --workers=16 --dataset=ImageNet --model=BayesResNet50 --epochs 45 --wd=1e-4 --beta 0.001 --lr_init 0.01 --lr_vars 0.001 --lv_init -6 --fname=$DATASET-BayesResNet50-$WORKER-$TIME"

    for i in $(seq $START $END)
    do
        echo ipython -- $ARGS-$i
        ipython -- $ARGS-$i
    done
elif [[ $METHOD == regular ]]
then
    ARGS="train/imagenet/train_imagenet.py -a resnet50 --data $DATAPATH --print-freq 400 --workers 32 --dir=$DIR --epochs 130 --fname=$DATASET-ResNet50-$WORKER-$TIME"
    for i in $(seq $START $END)
    do
        echo ipython -- $ARGS
        ipython -- $ARGS
    done
elif [[ $METHOD == sse ]]
then
   echo
   echo "SSE assumes a multi-GPU training, in our case 1 sample tooks 15-17 hours (45 epoch) on 4 v100."
   echo
   ARGS="./train/imagenet/train_imagenet_sse.py -a resnet50 --data $DATAPATH --print-freq 400 --workers 64 --cycle_epochs 45 --max_lr 0.1 --dist-url tcp://127.0.0.1:5552 --multiprocessing-distributed --world-size 1 --rank 0 "
   echo ipython -- $ARGS
   ipython -- $ARGS
elif [[ $METHOD == fge ]]
then
  echo
  echo "FGE assumes a multi-GPU training."
  echo "We hardcode a start model (--resume) for this script."
  echo

  ARGS="train/imagenet/train_imagenet_fge.py -a resnet50 --data $DATAPATH --print-freq 400 --workers 64 --cycle_epochs 2 --dist-url tcp://127.0.0.1:5551 --multiprocessing-distributed --world-size 1 --rank 0 --resume /home/aashukha/megares/ImageNet-ResNet50-cn-008--1564934041-1.pth.tar"
   echo ipython -- $ARGS
   ipython -- $ARGS
else
  echo wrong method
fi