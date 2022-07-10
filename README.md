# Pepper: Yet Another Framework for Image and Video Re-ID

## Motivation

There are a couple popular deep learning based ReID frameworks such as [`torchreid`](https://github.com/KaiyangZhou/deep-person-reid) and [`fastreid`](https://github.com/JDAI-CV/fast-reid).
There projects are very helpful for benchmarking SOTA methods as well as implementing new ideas quickly.
For my personal projects, I've been heavily using these projects.
The one problem that reduced my productivity was how I needed to add configuration defaults for every module that I added.
Inspired by [OpenMMLab's projects](https://github.com/open-mmlab), I created my own modular framework that uses [`mmcv`](https://github.com/open-mmlab/mmcv) that significantly reduced this complexity.

## Why use this framework?

- __Customisable__: experiments can be configured easily with the help of `mmcv`; no more bloated configs!
- __Extensible__: easy to swap modules and implement new ideas quickly without the hassle of breaking things
- __Fast__: training and evaluations are done using distributed processing
- __Robust__: borrows and implements techniques from other projects

__Notes__:
- Please open issues or PR if you spot any bugs or improvements.

## Installation

Clone the project:
```Bash
git clone --recursive git@github.com:haruishi43/pepper.git
cd pepper
```

### Dependencies:

- `torch` and `torchvision`
- `mmcv`
- `faiss-gpu`

Other dependencies can be installed using the following command:

```Bash
pip install -r requirements.txt
```

### Build evaluation module:

```Bash
cd pepper/core/evaluation/rank_cylib; make all
```

## Preparing for training/evaluation:

WIP

## Projects

- [Bag-of-Tricks](configs/projects/reproduce_BoT/README.md)
- [Video-Person-ReID](configs/video/resnet/README.md)


## TODO:

- [ ] README
- [ ] Documentation
- [ ] Upload model weights
- [ ] Test codes
