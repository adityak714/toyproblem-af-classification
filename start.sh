#!/bin/bash
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 centralized.py --epochs 18 --batch_size 256 --learning_rate 1e-4 --weight_decay 0.01
