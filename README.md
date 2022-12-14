# RelViT
[![arXiv](https://img.shields.io/badge/arXiv-2104.09159-red)](https://arxiv.org/abs/2206.00481)
[![arXiv](https://img.shields.io/badge/CVPRw-2022-yellow)](https://sites.google.com/view/t4v-cvpr22)
[![arXiv](https://img.shields.io/badge/BMVC-2022-blue)](https://bmvc2022.mpi-inf.mpg.de/0032.pdf)

```console
If you use the code of this repo and you find this project useful, 
please consider to give a star ⭐!
```

This repository hosts the official code related to the paper "Where are my Neighbors? Exploiting Patches Relations in Self-Supervised Vision Transformer", Guglielmo Camporese, Elena Izzo, Lamberto Ballan - BMVC 2022. [[arXiv](https://arxiv.org/abs/2206.00481)] [[video](http://vimp.math.unipd.it/downloads/relvit_spotlight.mp4)]

![relvit](https://guglielmocamporese.github.io/static/static/publications/Camporese2022WhereAM.png)

## BibTex Citation
```bibtex
@inproceedings{Camporese2022WhereAM,
  title     = {Where are my Neighbors? Exploiting Patches Relations in Self-Supervised Vision Transformer},
  author    = {Guglielmo Camporese, Elena Izzo, Lamberto Ballan},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2022}
}
```

## Updates
* **[22/10/13]** Our paper has been accepted to BMVC 2022 (oral spotlight)!
* **[22/06/02]** Our paper is on arXiv! Here you can find the [link](https://arxiv.org/abs/2206.00481).
* **[22/05/24]** Our paper has been selected for a spotlight oral presentation at the CVPR 2022 "T4V: Transformers for Vision" workshop!
* **[22/05/23]** Our paper just got accepted at the CVPR 2022 "T4V: Transformers for Vision" workshop!
<br><br>

# Install
<details>
<summary>Install</summary>

```console
# clone the repo
git clone https://github.com/guglielmocamporese/relvit.git

# install and activate the conda env
cd relvit
conda env create -f env.yml
conda activate relvit
```
</details>
<br>

# Training

All the commands are based on the training scripts in the `scripts` folder.

<details>
<summary>Self-Supervised Pre-Training + Supervised Finetuning</summary>

## Self-Supervised Pre-Training + Supervised Finetuning

Here you can find the commands for:

1. Running the self-supervised learning pre-training
```console
# SSL upstream pre-training
bash scripts/upstream.sh \
    --exp_id upstream_cifar10 --backbone vit \
    --model_size small --num_gpus 1 --epochs 100 --dataset cifar10  \
    --weight_decay 0.1 --drop_path_rate 0.1 --dropout 0.0
```

2. Running the supervised finetuning using the checkpoint obtained in the previous step.

> After running the upstream pre-training, the directory `tmp/relvit` will contain the file checkpoint `checkpoints/best.ckpt` file that has to be passed to the finetuning script in the `--model_checkpoint` argument.

```console
# supervised downstream
bash scripts/downstream.sh \
    --exp_id downstream_cifar10 --backbone vit --num_gpus 1 \
    --epochs 100 --dataset cifar10  --weight_decay 0.1 --drop_path_rate 0.1 \
    --model_size small --dropout 0.0 --model_checkpoint checkpoint_path
```
</details>


<details>
<summary>Downstream-Only Experiment</summary>

## Downstream-Only Experiment

Here you can find the commands for training the `ViT`, `Swin`, and `T2T` models for the downstram-only supervised task.
```console
# ViT downstream-only
bash scripts/downstream-only.sh \
    --seed 2022 --exp_id downstream-only_vit_cifar10 \
    --backbone vit --dataset cifar10 --weight_decay 0.1 --drop_path_rate 0.1 \
    --model_size small --dropout 0.0 --patch_trans colJitter:0.8-grayScale:0.2

# Swin downstream-only
bash scripts/downstream-only.sh \
    --seed 2022 --exp_id downstream-only_swin_cifar10 \
    --backbone swin  --dataset cifar10_224 --batch_size 64 --weight_decay 0.1 \
    --drop_path_rate 0.1 --model_size tiny --dropout 0.0 \
    --patch_trans colJitter:0.8-grayScale:0.2 

# T2T downstream-only
bash scripts/downstream-only.sh \
    --seed 2022--exp_id downstream-only_t2t_cifar10 \
    --backbone t2t_vit --dataset cifar10_224 --batch_size 64 --weight_decay 0.1 \
    --drop_path_rate 0.1 --model_size 14 --dropout 0.0
```
</details>

<details>
<summary>Mega-Patches Ablation</summary>

## Mega-Patches Ablation
Here you can find the experiments with the use of the `mega-patches` described in the paper. Also in this case, you can find the commands for the SSL upstream with the mega-patches and the subsequent supervised finetuning.
```console
# SSL upstream pre-training with 6x6 megapatches 
bash scripts/upstream_MEGApatch.sh \
    --exp_id upstream_megapatch_imagenet100 \
    --backbone vit --model_size small --dataset imagenet100 \
    --batch_size 256 --weight_decay 0.1 --drop_path_rate 0.1 \
    --dropout 0.0 --side_megapatches 6
```
> After running the upstream pre-training, the directory `tmp/relvit` will contain the file checkpoint `checkpoints/best.ckpt` file that has to be passed to the finetuning script in the `--model_checkpoint` argument.

```console
# downstream finetuning
bash scripts/downstream.sh \
    --exp_id downstream_imagenet100 --backbone vit \
    --dataset imagenet100 --weight_decay 0.1 --drop_path_rate 0.1 \
    --model_size small --dropout 0.0 --model_checkpoint checkpoint_path
```
</details>
<br>

<details>
<summary>Supported Datasets</summary>

## Supported Datasets
Here you can find the list of all the supported datasets in the repo that can be specified using the `--datasets` input argument in the previous commands.

### Datasets
* CIFAR10
* CIFAR100
* Flower102
* SVHN
* Tiny ImageNet
* ImageNet100
</details>
<br>

# Validate
<details>
<summary>Validation Scripts</summary>

```console
# validation on the upstream task
bash scripts/upstream.sh --dataset cifar10 --mode validation

# validation on the downstream task
bash scripts/downstream.sh --dataset cifar10 --mode validation
```
</details>