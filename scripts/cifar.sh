#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=0
python prune_resnet.py --config config/cifar100.cfg --gpu ${GPU}
