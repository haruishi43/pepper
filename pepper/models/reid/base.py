#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Sequence

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16


class BaseReID(BaseModule, metaclass=ABCMeta):
    """Base class for classifiers."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, "head") and self.head is not None

    @abstractmethod
    def extract_feat(self, imgs, stage=None):
        pass

    def extract_feats(self, imgs, stage=None):
        assert isinstance(imgs, Sequence)
        kwargs = {} if stage is None else {"stage": stage}
        for img in imgs:
            yield self.extract_feat(img, **kwargs)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        pass

    def forward_test(self, imgs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, "imgs")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            # FIXME: when testing video sequence, this will result in error as well
            raise NotImplementedError("aug_test has not been implemented")

    @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f"{loss_name} is not a tensor or list of tensors"
                )

        loss = sum(
            _value for _key, _value in log_vars.items() if "loss" in _key
        )

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.detach().clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.
        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data["img"].detach())
        )

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict, optional): The
                optimizer of runner is passed to ``train_step()``. This
                argument is unused and reserved.
        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data["img"].detach())
        )

        return outputs
