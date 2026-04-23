#!/bin/bash
OMP_NUM_THREADS=16 torchrun --standalone --nproc_per_node=2 centralized.py --epochs 20 --batch_size 256 --learning_rate 1e-4 --weight_decay 0.01
