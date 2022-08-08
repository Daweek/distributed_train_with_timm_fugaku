#!/bin/bash
#PJM -L  "node=1024"
#PJM -L  "rscgrp=large"
#PJM -L  "elapse=10:00:00"
#PJM --mpi  "rank-map-bynode"
#PJM --name "Imnet-deit-n1024"
#PJM -o ./outs/%n.%j.stdout
#PJM -e ./outs/%n.%j.err
#PJM -s --spath ./outs/%n.%j.stats

source /home/hp190122/u01961/working/pytorch/pytorch/scripts/fujitsu/env.src
source /home/hp190122/u01961/venv/bin/activate

# Do not create a file if there is not output
export PLE_MPI_STD_EMPTYFILE=off
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

nodos=1024

output_path="output/${PJM_JOBID}_n${nodos}"

mkdir $output_path

export MODEL=tiny
export LR=6.0e-4
export CLS=1
export EPOCHS=300
export BATCH_SIZE=65536
export OUT_DIR=/home/hp190122/u01961/working/deepL/transformer/timm_main_sora/checkpoint/${MODEL}/imnet${CLS}k/pre_training

# export NGPUS=8
# export NUM_PROC=8
export LOCAL_BATCH_SIZE=64

## For performance tuning.
export OMP_NUM_THREADS=48
export KMP_AFFINITY=granularity=fine,compact,1,0


LD_PRELOAD=$PREFIX/lib/libtcmalloc.so mpirun -np ${nodos} -std-proc ${output_path}/stdproc \
python pretrain.py /NOT/WORKING \
    -w --trainshards "/home/hp190122/u01961/data/ILSVRC2012_5MB_shards/train/imagenet-train-{000000..028606}.tar" \
    --model vit_deit_${MODEL}_patch16_224 --dataset_size 1281167 --experiment pretrain_deit_${MODEL}_imnet${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_cpu${nodos} \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw --num-classes ${CLS}000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    --eval-metric loss --no-prefetcher \
    --interval-saved-epochs 5 --checkpoint-hist 20 --output ${OUT_DIR} \
    --log-wandb

#mpirun -np ${nodos} -stdout ${output_path}/stdout -std-proc ${output_path}/stdproc python train_benchmark.py --bs ${bs} --epochs 1 --n ${nodos}

