#!/usr/bin/env python3

from .gap import GlobalAveragePooling, KernelGlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .pcb import PartPooling, RefinedPartPooling

__all__ = [
    "GlobalAveragePooling",
    "GeneralizedMeanPooling",
    "KernelGlobalAveragePooling",
    "PartPooling",
    "RefinedPartPooling",
]
