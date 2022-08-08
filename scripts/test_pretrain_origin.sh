#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=2:00:00
#$ -j y
#$ -o output/test_pretrain_original_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1 cudnn/8.3/8.3.2 openmpi/4.0.5 nccl/2.8/2.8.4-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=base
export LR=1.0e-3
export CLS=1
export EPOCHS=2
export BATCH_SIZE=1024
export OUT_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/i${CLS}k/test

export NGPUS=16
export NUM_PROC=4
export LOCAL_BATCH_SIZE=64

mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /groups/gcd50691/datasets/ImageNet \
    --model deit_${MODEL}_patch16_224 --experiment test_deit_${MODEL}_i${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_noshard_original \
    --input-size 3 224 224 --num-classes ${CLS}000 \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 \
    -j 16 --eval-metric loss \
    --warmup-lr 1e-6 --min-lr 1e-6 \
    --output ${OUT_DIR}

    # --fsdp
    # --resume ${OUT_DIR}/test_deit_${MODEL}_i${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_noshard_original/last.pth.tar
