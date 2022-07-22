#!/usr/bin/env python3

import torch
import mmcv


def single_gpu_metric_test(
    model,
    data_loader,
    show=False,
    out_dir=None,
    **show_kwargs,
):
    """Test model with local single gpu.
    This method tests model with a single gpu and supports showing results.
    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    preds = []
    feats = []

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            pred, feat = model(return_loss=False, return_feats=True, **data)

        batch_size = len(pred)
        preds.extend(pred)
        feats.extend(feat)

        batch_size = data["img"].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    # TODO: could visualize after collecting everything
    return preds, feats
