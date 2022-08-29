# Metric Learning Playground

Depends on `mmcls`.

What I want to reproduce:
- [x] center loss
- [x] ArcFace
- [x] CosFace
- [x] SphereFace
- [x] CircleLoss


# Center Loss

```Bash
./tools/dist_train.sh configs/lenet/center_loss_mnist.py 2 --cfg-options data.samples_per_gpu=256
```

__NOTES__:
- very unstable training (needs to tune learning rate)
  - learning rate varies on what hardware (# of gpus) you use
  - from my experience, using smaller learning rates for lower number of gpus (e.g., 0.005 for single gpu, 0.01 for dual gpus) tends to yield stable training.

## Results

<div align="center">
  <img src=".readme/ce.png" alt="ce" width="30%">
  <img src=".readme/center.png" alt="center" width="30%">
</div>


# SphereFace

- Not reproduced completely. Feature embedding seems buggy

```Bash
./tools/dist_train.sh configs/lenet/sphereface_mnist.py 2 --cfg-options data.samples_per_gpu=128
```

## Results

<div align="center">
  <img src=".readme/sphereface.png" alt="sphereface" width="30%">
</div>


# ArcFace, CosFace

```Bash
./tools/dist_train.sh configs/lenet/arcface_mnist.py 2 --cfg-options data.samples_per_gpu=128

./tools/dist_train.sh configs/lenet/cosface_mnist.py 2 --cfg-options data.samples_per_gpu=128
```

__NOTES__:
- Embedding feature space must be more than 3 (doesn't train well with `vis_dim=2`).
- `CosFace` is not that robust and needs good hyperparameters
- The embedded features for `arcface` does not seem like it is well reproduced.

## Results

<div align="center">
  <img src=".readme/arcface.png" alt="arcface" width="30%">
  <img src=".readme/cosface.png" alt="cosface" width="30%">
</div>


# CircleLoss

```Bash
./tools/dist_train.sh configs/lenet/circleloss_mnist.py 2 --cfg-options data.samples_per_gpu=128
```

## Results

<div align="center">
  <img src=".readme/circleloss.png" alt="circleloss" width="30%">
</div>
