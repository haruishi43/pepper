# Metric Learning Playground

Depends on `mmcls`.

What I want to reproduce:
- [x] center loss
- [x] ArcFace
- [x] CosFace
- [x] SphereFace
- [x] CircleLoss


# Center Loss


## Results

<div align="center">
  <img src=".readme/ce.png" alt="ce" width="30%">
  <img src=".readme/center.png" alt="center" width="30%">
</div>

# ArcFace, CosFace

__NOTES__:
- Embedding feature space must be more than 3 (doesn't train well with `vis_dim=2`).
- `CosFace` is not that robust and needs good hyperparameters


# CircleLoss

