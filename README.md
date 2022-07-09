# Pepper: Yet Another Framework for Image and Video Re-ID

## Motivation

There are a couple popular deep learning based ReID frameworks such as [`torchreid`](https://github.com/KaiyangZhou/deep-person-reid) and [`fastreid`](https://github.com/JDAI-CV/fast-reid).
There projects are very helpful for benchmarking SOTA methods as well as implementing new ideas quickly.
Inspired by [OpenMMLab's projects](https://github.com/open-mmlab), I created my own modular framework that uses [`mmcv`](https://github.com/open-mmlab/mmcv) (not affiliated).

Goals:
- Configurable: experiments can be configured easily with the help of `mmcv`
- Extensible: easy to swap modules and implementing new ideas quickly
- Fast: training and evaluations are done using distributed processing
- Robust: borrows techniques from other frameworks and projects to make training robust

## Projects

- [Bag-of-Tricks](configs/projects/reproduce_BoT/README.md)
- [Video-Person-ReID](configs/video/resnet/README.md)
