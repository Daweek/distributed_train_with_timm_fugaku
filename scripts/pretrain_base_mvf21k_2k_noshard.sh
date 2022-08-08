#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_deit_base_mvf21k_2k_lr1e-3_90epochs_bs8192_noshard_$JOB_ID.out

# ======== Virtual Env/ ========
export PATH="~/anaconda3/bin:${PATH}"
source activate distributed_train

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=base
export LR=1.0e-3
export CLS=21
export EPOCHS=90
export BATCH_SIZE=8192
export OUT_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/mvf${CLS}k_2k/pre_training

export NGPUS=128
export NUM_PROC=4
export LOCAL_BATCH_SIZE=64

mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /groups/gcd50691/datasets/MV-FractalDB/var0.05/MVFractalDB21k/cat21k_ins2k \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_MVfractal${CLS}k_2k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_noshard \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw --num-classes ${CLS}000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb

    # --resume ${OUT_DIR}/pretrain_deit_${MODEL}_MVfractal${CLS}k_2k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_noshard/last.pth.tar
