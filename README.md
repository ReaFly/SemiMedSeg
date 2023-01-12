#  Self-Supervised Correction Learning for Semi-Supervised Biomedical Image Segmentation

##  Introduction

This repository contains the PyTorch implementation of:

Self-Supervised Correction Learning for Semi-Supervised Biomedical Image Segmentation, MICCAI 2021.

##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

##  Usage

####  1. Training

```bash
python main.py  --mode train  --manner semi --ratio 2 
--root {project_path} --dataset polyp --polyp {data_path}

```



####  2. Inference

```bash
python main.py  --mode test  --manner test --load_ckpt checkpoint 
--root {project_path} --dataset polyp --polyp {data_path}
```



##  Citation

If you feel this work is helpful, please cite our paper

```
@inproceedings{zhang2021self,
  title={Self-supervised Correction Learning for Semi-supervised Biomedical Image Segmentation},
  author={Zhang, Ruifei and Liu, Sishuo and Yu, Yizhou and Li, Guanbin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={134--144},
  year={2021},
  organization={Springer}
}
```





