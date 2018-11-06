#!/bin/bash
set -e

python prepare.py
cd detector
# eps=100
set eps=001
# TODO: -b 32
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 
# python main.py --model res18 -b 1 --epochs $eps --save-dir res18 
# Train N-net, can run on CPU/GPU.
python main.py --model res18 -b 1 --epochs %eps% --save-dir res18 -j 1 --save-freq 1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 
# Test N-net, need 12G GPU to run.
# TODO: run this commented one instead when 12G+ GPU available.
# python main.py --model res18 -b 1 --resume results/res18/%eps%.ckpt --test 1 -j 1 --save-freq 1
python main.py --model res18 -b 1 --epochs 002 --save-dir res18 -j 1 --save-freq 1 --resume results/res18/001.ckpt --gpu all
cp results/res18/%eps%.ckpt ../../model/detector.ckpt

# TODO: -b2 12
cd ../classifier
python adapt_ckpt.py --model1  net_detector_3 --model2  net_classifier_3  --resume ../detector/results/res18/%eps%.ckpt 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 
# epochs 130
python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 1 -b2 1 --save-dir net3 --resume ./results/start.ckpt --start-epoch 30 --epochs 1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 
# 160.ckpt
python main.py --model1  net_detector_3 --model2  net_classifier_4 -b 1 -b2 1 --save-dir net4 --resume ./results/net3/130.ckpt --freeze_batchnorm 1 --start-epoch 121
cp results/net4/31.ckpt ../../model/classifier.ckpt
