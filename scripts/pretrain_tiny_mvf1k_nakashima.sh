#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_deit_tiny_mvf1k_lr6e-4_300epochs_bs1024_nakashima_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=tiny
export LR=6.0e-4
export CLS=1
export EPOCHS=300
export BATCH_SIZE=1024
export OUT_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/mvf${CLS}k/pre_training

export NGPUS=8
export NUM_PROC=4
export LOCAL_BATCH_SIZE=128

mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /groups/gcd50691/datasets/MV-FractalDB/cat1000_ins145 \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_MVfractal${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_nakashima \
    --hflip 0.0 --aa rand-m9-mstd0.5-inc1 --interpolation bicubic \
    --mean 0.5 0.5 0.5 \
    --std 0.5 0.5 0.5 \
    --reprob 0.25 --remode pixel \
    --batch-size ${LOCAL_BATCH_SIZE} -j 10 --pin-mem \
    --seed 0 \
    --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 \
    --opt adamw --lr ${LR} --weight-decay 0.05 \
    --epochs ${EPOCHS} --sched cosine_iter --min-lr 1.0e-5 \
    --warmup-lr 1.0e-6 --warmup-epochs 5 --cooldown-epochs 10 \
    --num-classes ${CLS}000 --eval-metric loss \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --resume ${OUT_DIR}/pretrain_deit_${MODEL}_MVfractal${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_nakashima/last.pth.tar \
    --log-wandb

# --resume ${OUT_DIR}/pretrain_deit_${MODEL}_MVfractal${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_nakashima/last.pth.tar

# --color-jitter, --recount, --mixup-prob, --mixup-switch-prob, --mixup-mode,
# --smoothing, --lr-cycle-mul, --decay-rate, --lr-cycle-limit, --clip_grad and --clip-mode
# are used default setting in pretrain.py
