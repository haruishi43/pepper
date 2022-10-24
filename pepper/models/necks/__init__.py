#!/usr/bin/env python3

from .gap import GlobalAveragePooling, KernelGlobalAveragePooling
from .gem import GeneralizedMeanPooling
from .pcb import PartPooling, RefinedPartPooling
from .mgn_neck import MGNPooling, AMGNPooling

__all__ = [
    "GlobalAveragePooling",
    "GeneralizedMeanPooling",
    "KernelGlobalAveragePooling",
    "PartPooling",
    "RefinedPartPooling",
    "MGNPooling",
    "AMGNPooling",
]
