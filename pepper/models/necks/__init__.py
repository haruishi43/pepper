#!/usr/bin/env python3

from .gap import GlobalAveragePooling, KernelGlobalAveragePooling
from .gem import GeneralizedMeanPooling

__all__ = [
    "GlobalAveragePooling",
    "GeneralizedMeanPooling",
    "KernelGlobalAveragePooling",
]
