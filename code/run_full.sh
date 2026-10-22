#!/bin/bash
# Stage 1: native OSTIA pretraining (1993-2014), Stage 2: JAXA adaptation. 4 GPUs DDP (A100 0-3).
cd /mnt/disk_e/lz/fno_diffusion_v1/code
export CUDA_VISIBLE_DEVICES=0,1,2,3 FNO_CACHE=/mnt/disk_e/lz/fno_diffusion_v1/cache OMP_NUM_THREADS=4
R=/mnt/disk_e/lz/fno_diffusion_v1/runs
TR=/home/lz/miniconda3/envs/pytorch/bin/torchrun
$TR --nproc_per_node 4 --master_port 29533 train.py --mode ostia_full --out $R/ostia_full --steps 60000 --bs 32 --lr 2e-4 --scale 1.6 --val-every 2000 --ostia-years 1993 2014 > $R/ostia_full.stdout 2>&1 || exit 1
$TR --nproc_per_node 4 --master_port 29534 train.py --mode jaxa --init $R/ostia_full/ckpt.pt --out $R/jaxa_adapt --steps 20000 --bs 16 --lr 1e-4 --scale 1.6 --val-every 1000 --ostia-years 1993 2014 > $R/jaxa_adapt.stdout 2>&1
