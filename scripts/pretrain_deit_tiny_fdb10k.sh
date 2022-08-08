#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_deit_tiny_fdb10k_lr8e-3_100epochs_bs8192_shards_$JOB_ID.out

# ======== Virtual Env/ ========
# export PATH="~/anaconda3/bin:${PATH}"
# source activate distributed_train

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"


# ======== Modules ========
. /etc/profile.d/modules.sh
module purge
module load openmpi/3.1.6 cuda/11.4/11.4.1 cudnn/8.2/8.2.4 nccl/2.11/2.11.4-1 

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=tiny
export LR=8.0e-3
export CLS=10
export EPOCHS=100
export BATCH_SIZE=8192
export OUT_DIR=/home/acc12930pb/working/transformer/timm_main_sora/checkpoint/${MODEL}/fdb${CLS}k/pre_training

export NGPUS=128
export NUM_PROC=4
export LOCAL_BATCH_SIZE=64

mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /NOT/WORKING \
    -w --trainshards "/groups/gcd50691/datasets/FractalDB_Shards_10k/fdb_10k-train-{000000..000999}.tar" \
    --model deit_${MODEL}_patch16_224 --experiment pretrain_deit_${MODEL}_fdbfractal${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_shards \
    --sched cosine_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw --num-classes ${CLS}000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 5 --eval-metric loss --no-prefetcher \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb

    # --resume ${OUT_DIR}/pretrain_deit_${MODEL}_MVfractal${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_shards/last.pth.tar
