# RelViT
[![arXiv](https://img.shields.io/badge/arXiv-2104.09159-red)](https://arxiv.org/abs/2206.00481)
[![arXiv](https://img.shields.io/badge/CVPRw-2022-yellow)](https://sites.google.com/view/t4v-cvpr22)
[![arXiv](https://img.shields.io/badge/BMVC-2022-blue)](https://bmvc2022.mpi-inf.mpg.de/0032.pdf)

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

```bash
If you use the code of this repo and you find this project useful, 
please consider to give a star ⭐!
```

## Updates
* **[22/10/13]** Our paper has been accepted to BMVC 2022 (oral spotlight)!
* **[22/06/02]** Our paper is on arXiv! Here you can find the [link](https://arxiv.org/abs/2206.00481).
* **[22/05/24]** Our paper has been selected for a spotlight oral presentation at the CVPR 2022 "T4V: Transformers for Vision" workshop!
* **[22/05/23]** Our paper just got accepted at the CVPR 2022 "T4V: Transformers for Vision" workshop!

## Install

```console
# clone the repo
git clone https://github.com/guglielmocamporese/relvit.git

# install and activate the conda env
cd relvit
conda env create -f env.yml
conda activate relvit
```

## Training Scripts

All the training scripts are on the `scripts` folder.