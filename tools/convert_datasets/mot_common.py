#!/usr/bin/env python3

"""Functions for preprocessing MOT datasets for ReID
"""

import glob
import os.path as osp
import warnings
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd


def crop_person(
    img: np.ndarray, bbox: np.ndarray, save_path: Optional[str] = None
):
    canvas = img.copy()
    x1, y1, x2, y2 = bbox.astype("i")
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cropped_img = img[y1:y2, x1:x2].copy()

    if save_path is not None:
        if cropped_img.size == 0:
            warnings.warn(
                f"could not save image to {save_path} since image is blank"
            )
            return None
        else:
            cv2.imwrite(save_path, cropped_img)
            return cropped_img
    else:
        return cropped_img


def get_gts(gt_file: str) -> pd.DataFrame:
    """Get gts from dataset directory"""
    gts = pd.read_csv(gt_file)
    gts.columns = [
        "frame",
        "id",
        "x1",
        "y1",
        "w",
        "h",
        "is_ped",
        "class",
        "vis_ratio",
    ]
    return gts


def get_det(det_file: str, detector: str) -> pd.DataFrame:
    """Get detection results from dataset directory"""
    dets = pd.read_csv(det_file)
    if detector == "DPM":
        dets.columns = [
            "frame",
            "id",
            "x1",
            "y1",
            "w",
            "h",
            "score",
            "ig1",
            "ig2",
            "ig3",
        ]
    else:
        dets.columns = [
            "frame",
            "id",
            "x1",
            "y1",
            "w",
            "h",
            "score",
        ]
    return dets


def get_frame_paths(img_dir: str) -> List[str]:
    """Get all frames in a sequence"""
    assert osp.isdir(img_dir)
    return glob.glob(osp.join(img_dir, "*.jpg"))
