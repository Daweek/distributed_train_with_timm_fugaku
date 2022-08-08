#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/pretrain_deit_with_vit_large_rc21k_lr5e-4_90epochs_bs2048_shard_$JOB_ID.out

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

export MODEL=large
export LR=5.0e-4 # when use larger model, should use lower lr
export CLS=21
export EPOCHS=90
export BATCH_SIZE=2048 # can work only with local batch size <= 16 for limit size of GPU memory
export OUT_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/rc${CLS}k/pre_training

export NGPUS=128
export NUM_PROC=4
export LOCAL_BATCH_SIZE=16

mpirun -npernode $NUM_PROC -np $NGPUS \
python pretrain.py /NOT/WORKING \
    -w --trainshards "/groups/gcd50691/datasets/RCDB_Shards_21k/rcdb_21k-train-{000000..002099}.tar" \
    --model vit_${MODEL}_patch16_224 --experiment pretrain_deit_with_vit_${MODEL}_RCDB${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_shards \
    --sched linear_iter --epochs ${EPOCHS} --lr ${LR} --weight-decay 0.1 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw --num-classes ${CLS}000 \
    --warmup-iter 10000 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --interval-saved-epochs 10 --output ${OUT_DIR} \
    --log-wandb \
    --resume ${OUT_DIR}/pretrain_deit_with_vit_${MODEL}_RCDB${CLS}k_lr${LR}_epochs${EPOCHS}_bs${BATCH_SIZE}_shards/last.pth.tar
    