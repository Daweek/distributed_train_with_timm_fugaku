#!/bin/bash
#$ -cwd
#$ -l rt_F=16
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/finetune_deit_with_vit_large_i1k_from_mvf21k_lr5e-4_90epochs_bs2048_shards_tuning-lr5e-4_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

export MODEL=large
export PRE_CLS=21
export PRE_LR=5.0e-4
export PRE_EPOCHS=90
export PRE_BATCH=2048

export CP_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/mvf${PRE_CLS}k/pre_training/pretrain_deit_with_vit_${MODEL}_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards/model_best.pth.tar
export OUT_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/mvf${PRE_CLS}k/fine_tuning

export NGPUS=64
export NPERNODE=4
export BATCH_SIZE=1024
export LOCAL_BATCH_SIZE=16
export LR=5.0e-4

mpirun -npernode $NPERNODE -np $NGPUS \
python finetune.py /groups/gcd50691/datasets/ImageNet \
    --model vit_${MODEL}_patch16_224 --experiment finetune_deit_with_vit_large_i1k_from_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards_tuning-lr${LR} \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 300 --lr ${LR} --weight-decay 0.05 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ${CP_DIR}

#   --resume  ${OUT_DIR}/finetune_deit_with_vit_large_i1k_from_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards/last.pth.tar
