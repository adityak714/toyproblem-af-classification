#!/bin/bash
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 centralized.py --epochs 20 --batch_size 256 --learning_rate 1e-5 --weight_decay 0.01
