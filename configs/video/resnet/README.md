# Video-Person-ReID

```
@article{gao2018revisiting,
  title={Revisiting Temporal Modeling for Video-based Person ReID},
  author={Gao, Jiyang and Nevatia, Ram},
  journal={arXiv preprint arXiv:1805.02104},
  year={2018}
}
```

## ResNet50 Backbone

Results using `pepper`:

| Model (T=4)                                         | mAP  | Rank-1 |
|-----------------------------------------------------|------|--------|
| [Temporal Avg. Pooling](tp_resnet50_b32_t4_mars.py) | --   | --     |
| [Temporal Att.](ta_resnet50_b32_t4_mars.py)         | --   | --     |
| [RNN](rnn_resnet50_b32_t4_mars.py)                  | 76.1 | 83.2   |


Results from the original paper:

| Model (T=4)           | mAP  | Rank-1 |
|-----------------------|------|--------|
| Temporal Avg. Pooling | 76.5 | 83.3   |
| Temporal Att.         | 76.7 | 83.3   |
| RNN                   | 73.9 | 81.6   |


---

References:
- [Video-Person-ReID](https://github.com/jiyanggao/Video-Person-ReID)
- [Revisiting Temporal Modeling for Video-based Person ReID](https://arxiv.org/pdf/1805.02104.pdf)
