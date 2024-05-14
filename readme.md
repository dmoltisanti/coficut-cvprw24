# Introduction

This repository contains the code and data for the paper 
**Coarse or Fine? Recognising Action End States without Labels**.

The paper was accepted at 
[The Eleventh Workshop on Fine-Grained Visual Categorisation](https://sites.google.com/view/fgvc11/home) workshop
hosted at CVPR 24.

If you find our work useful, please cite our paper:

```
@inproceedings{moltisanti24coarse,
author = {Moltisanti, Davide and Bilen, Hakan and Sevilla-Lara, Laura and Keller, Frank},
title = {{Coarse or Fine? Recognising Action End States without Labels}},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
year = {2024}}
```

### Read the paper

You can find our paper on [arXiv](https://arxiv.org/abs/2405.07723).

### Authors

- [Davide Moltisanti](https://www.davidemoltisanti.com/research/) (University of Bath, work done while at Edinburgh)
- [Hakan Bilen](https://homepages.inf.ed.ac.uk/hbilen/) (University of Edinburgh)
- [Laura Sevilla-Lara](https://laurasevilla.me/) (University of Edinburgh)
- [Frank Keller](https://homepages.inf.ed.ac.uk/keller/) (University of Edinburgh)

# Install

We provide a conda environment to install all the necessary libraries (see file `environment.yml`). 
Note: `lama-cleaner` needs a Rust compiler to build the `tokeninzer` library. 
This will fail if you use a very recent version of Python (3.8 is fine).  

# Download VOST-AUG

You can download the dataset we augmented from [VOST](https://www.vostdataset.org/data.html) 
[here](https://github.com/dmoltisanti/coficut-cvprw24/releases/download/v1.0/VOST_aug.zip). 
This is the dataset we generated as detailed in the paper, which we use for all experiments.  

Note that images are scaled to `256x144` pixels to save space. 
If you want full resolution images you can use the script `scripts/augment_vost.py` to generate new images 
from the original VOST dataset at the desired resolution.

## Generate VOST-AUG

1. Download the VOST dataset from [this link](https://www.vostdataset.org/data.html)
2. Adjust the paths in the script located at `scripts/augment_vost.py`
3. Launch the script, adjusting the number of processes if needed. The script will spawn `n_proc` processes on the
   GPU to generated images in parallel.

By default, the script will generate the augmented dataset using the same parameters as in our released dataset.
However, as random sampling is involved during augmentation, your generated images may not be identical to our 
released ones even if you use the same parameters. 

# Download the COFICUT evaluation dataset 

Get in touch via [email](dm2460@bath.ac.uk).

# Training and testing

We provide training and testing scripts in the `scripts/` folder to train and test our model and the main baselines.
Don't hesitate to get in touch by opening a GitHub issue if you need help!
