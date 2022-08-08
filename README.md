# Requirements

* Python 3 (worked at 3.8.2)
* OpenMPI 4.0.5
* CUDA Toolkit 10.2
* cuDNN 8.0
* NCCL 2.7
​

please install packages with the command.

```bash
$ pip install -r requirements.txt
```

(Reproducing a large scale experiment requires a large amount of computational resources.)

# Pre-Train with shard datasets

Run the python script ```pretrain.py```, you can pre-train with your shard dataset. (noshard dataset is also available.)

(To make shard dataset, please refer to this repository: https://github.com/webdataset/webdataset)

## Running the code

Basically, you can run the python script ```pretrain.py``` with the command.

example : deit_base, ExFractalDB-21k(shard)

```bash
$ python pretrain.py /NOT/WORKING \
    -w --trainshards /PATH/TO/ExFractalDB21000/SHARDS-{000000..002099}.tar \
    --model deit_base_patch16_224 --experiment pretrain_deit_base_ExFractalDB21000_1.0e-3_shards \
    --sched cosine_iter --epochs 90 --lr 1.0e-3 --weight-decay 0.05 \
    --batch-size 64 --opt adamw --num-classes 21000 \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --remode pixel --interpolation bicubic --hflip 0.0 \
    -j 1 --eval-metric loss --no-prefetcher \
    --interval_saved_epochs 10 --output ./output/pretrain \
    --log-wandb
```

Or you can run the job script ```scripts/pretrain.sh``` (spport multi-node training).

​
When running with the script above, please make your shard directory structure as following.

```misc
/PATH/TO/ExFractalDB21000/
    SHARDS-000000.tar
    SHARDS-000001.tar
    ...
    SHARDS-002099.tar
```

Please see the script and code files for details on each argument.

# Fine-Tuning with small datasets

Run the python script ```finetune.py```, you can fine-tune to other datasets from your pre-trained model.
​
## Running the code

Basically, you can run the python script ```finetune.py``` with the command.

example : deit_base, ImageNet1k from model pre-trained with ExFractalDB-21k(shard)

```bash
$ python finetune.py /PATH/TO/IMAGENET \
    --model deit_base_patch16_224 --experiment finetune_deit_base_ImageNet1k_from_ExFractalDB21000_1.0e-3 \
    --input-size 3 224 224 --num-classes 1000 \
    --sched cosine_iter --epochs 300 --lr 1.0e-3 --weight-decay 0.05 \
    --batch-size 64 --opt adamw \
    --warmup-epochs 5 --cooldown-epochs 0 \
    --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug --mixup 0.8 --cutmix 1.0 \
    --drop-path 0.1 --reprob 0.25 -j 16 \
    --output ./output/finetune \
    --log-wandb \
    --pretrained-path ./output/pretrain/pretrain_deit_base_ExFractalDB21000_1.0e-3_shards/model_best.pth.tar
```
​
Or run the job script ```scripts/finetune.sh``` (spport multi-node training).

When running with the script above, please make your dataset directory structure as following.
​
```misc
/PATH/TO/IMAGENET/
  train/
    class1/
      img1.jpeg
      ...
    class2/
      img2.jpeg
      ...
    ...
  val/
    class1/
      img3.jpeg
      ...
    class/2
      img4.jpeg
      ...
    ...
```

Please see the script and code files for details on each argument.

# Acknowledgements

The codes are inspired by [timm](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit).
