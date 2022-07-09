# Bag-of-Tricks

```
@InProceedings{Luo_2019_CVPR_Workshops,
  author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM,
  author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}},
  journal={IEEE Transactions on Multimedia},
  title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification},
  year={2019},
  pages={1-1},
  doi={10.1109/TMM.2019.2958756},
  ISSN={1941-0077},
}
```

## ResNet50 Backbones

### Market1501

Results using `pepper`:

| Model                                                                | mAP  | Rank-1 |
|----------------------------------------------------------------------|------|--------|
| [Baseline](bot_resnet/baseline_market1501.py)                        | 77.8 | 91.7   |
| [+warmup](bot_resnet/baseline_warmup_market1501.py)                  | 77.9 | 91.2   |
| [+REA](bot_resnet/baseline_warmup_REA_market1501.py)                 | 81.4 | 92.3   |
| [+LS](bot_resnet/baseline_warmup_REA_LS_market1501.py)               | 81.7 | 92.9   |
| [+stride=1](bot_resnet/baseline_warmup_REA_LS_stride1_market1501.py) | 83.8 | 93.9   |
| [+BNNeck](bot_resnet/bot_market1501.py)                              | 82.3 | 92.8   |
| +center loss                                                         | --   | --     |

Results from the original paper:

| Model        | mAP  | Rank-1 |
|--------------|------|--------|
| Baseline     | 74.0 | 87.7   |
| +warmup      | 75.2 | 88.7   |
| +REA         | 79.3 | 91.3   |
| +LS          | 80.3 | 91.4   |
| +stride=1    | 81.7 | 92.0   |
| +BNNeck      | 85.7 | 94.1   |
| +center loss | 85.9 | 94.5   |

Notes:
- Differences from the original implementation:
  - original paper uses `Padding` + `RandomCropping`; I used [`Random2DTranslation`](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/transforms.py)
- `BNNeck` should be the same as the original implementation, but there are not much improvements.
- "center loss" has not been implemented here yet

---

Reference:
- [Fast-ReID](https://github.com/JDAI-CV/fast-reid)
- [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline)
