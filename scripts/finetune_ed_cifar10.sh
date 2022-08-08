#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/finetune_deit_tiny_cifar10_from_fdb10k_lr8.0e-3_100epochs_bs8192_render_pretrained_by_Ed_$JOB_ID.out

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"

# ======== Modules ========
. /etc/profile.d/modules.sh
module purge
module load openmpi/4.1.3 cuda/11.4/11.4.4 cudnn/8.2/8.2.4 nccl/2.11/2.11.4-1 

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)

# export MODEL=base
# export PRE_CLS=100
# export PRE_LR=1.0e-3
# export PRE_EPOCHS=20
# export PRE_BATCH=8192

export MODEL=tiny
export PRE_CLS=10
export PRE_LR=8.0e-3
export PRE_EPOCHS=100
export PRE_BATCH=8192

# export CP_DIR=/groups/gcd50691/yokota_check_points/${MODEL}/mvf${PRE_CLS}k/pre_training/pretrain_deit_${MODEL}_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards/model_best.pth.tar

# export CP_DIR=/home/acc12930pb/working/transformer/timm_main_sora/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/pretrain_deit_${MODEL}_fdbfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_shards/model_best.pth.tar

export CP_DIR=/home/acc12930pb/working/transformer/timm_main_sora/checkpoint/${MODEL}/fdb${PRE_CLS}k/pre_training/pretrain_deit_${MODEL}_fdbfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_render/model_best.pth.tar

export OUT_DIR=/home/acc12930pb/working/transformer/timm_main_sora/checkpoint/${MODEL}/fdb${PRE_CLS}k/fine_tuning

export NGPUS=16
export NUM_PROC=4
export BATCH_SIZE=768
export LOCAL_BATCH_SIZE=48

mpirun -npernode $NUM_PROC -np $NGPUS \
python finetune.py /groups/gcd50691/datasets/cifar10 \
    --model deit_${MODEL}_patch16_224 --experiment finetune_deit_${MODEL}_cifar10_from_fdbfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_pretrained_by_Ed_render \
    --input-size 3 224 224 --num-classes 10 \
    --sched cosine_iter --epochs 1000 --lr 0.01 --weight-decay 0.0001 \
    --batch-size ${LOCAL_BATCH_SIZE} --opt sgd \
    --warmup-epochs 10 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    -j 5 \
    --output ${OUT_DIR} \
    --log-wandb \
    --pretrained-path ${CP_DIR}

    # --resume ${OUT_DIR}/finetune_deit_${MODEL}_cifar10_from_MVfractal${PRE_CLS}k_lr${PRE_LR}_epochs${PRE_EPOCHS}_bs${PRE_BATCH}_pretrained_by_nakashima/last.pth.tar
