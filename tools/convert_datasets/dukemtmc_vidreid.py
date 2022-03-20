#!/usr/bin/env python3

"""DukeMTMCVidReID.

Reference:
    - Ristani et al. Performance Measures and a Data Set for Multi-Target,
    Multi-Camera Tracking. ECCVW 2016.
    - Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.

URL: `<https://github.com/Yu-Wu/DukeMTMC-VideoReID>`_

Dataset statistics:
    - identities: 702 (train) + 702 (test).
    - tracklets: 2196 (train) + 2636 (test).
"""

import argparse
import glob
import json
import os.path as osp
import warnings

from tqdm import tqdm

from mmcv.utils import mkdir_or_exist

DATASET_DIR = "dukemtmc-vidreid/DukeMTMC-VideoReID"
DATASET_URL = (
    "http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert/format dukemtmc-videoreid dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=f"data/{DATASET_DIR}",
        help="root directory",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="gtPepper",
        help="where to save the processed gts (`gtPepper`)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def process_dir(dir_path, relabel=False, min_seq_len=0):

    pdirs = glob.glob(osp.join(dir_path, "*"))  # avoid .DS_Store
    print(
        'Processing "{}" with {} person identities'.format(dir_path, len(pdirs))
    )
    pid_container = set()
    for pdir in pdirs:
        pid = int(osp.basename(pdir))
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    tracklets = []
    for pdir in tqdm(pdirs):
        pid = int(osp.basename(pdir))
        if relabel:
            pid = pid2label[pid]

        tdirs = glob.glob(osp.join(pdir, "*"))
        for tdir in tdirs:
            raw_img_paths = glob.glob(osp.join(tdir, "*.jpg"))
            num_imgs = len(raw_img_paths)

            if num_imgs < min_seq_len:
                continue

            img_paths = []
            for img_idx in range(num_imgs):
                # some tracklet starts from 0002 instead of 0001
                img_idx_name = "F" + str(img_idx + 1).zfill(4)
                res = glob.glob(osp.join(tdir, "*" + img_idx_name + "*.jpg"))
                if len(res) == 0:
                    warnings.warn(
                        "Index name {} in {} is missing, skip".format(
                            img_idx_name, tdir
                        )
                    )
                    continue
                img_paths.append(res[0])
            img_name = osp.basename(img_paths[0])
            if img_name.find("_") == -1:
                # old naming format: 0001C6F0099X30823.jpg
                camid = int(img_name[5]) - 1
            else:
                # new naming format: 0001_C6_F0099_X30823.jpg
                camid = int(img_name[6]) - 1
            tracklets.append(
                dict(
                    pid=pid,
                    camid=camid,
                    img_paths=img_paths,
                    tracklet_length=len(img_paths),
                )
            )

    return tracklets


if __name__ == "__main__":

    args = parse_args()

    assert osp.exists(args.root)

    train_dir = osp.join(args.root, "train")
    query_dir = osp.join(args.root, "query")
    gallery_dir = osp.join(args.root, "gallery")

    assert osp.exists(train_dir)
    assert osp.exists(query_dir)
    assert osp.exists(gallery_dir)

    split_dirs = dict(
        train=train_dir,
        query=query_dir,
        gallery=gallery_dir,
    )

    save_root = osp.join(args.root, args.out_dir)
    mkdir_or_exist(save_root)

    for split, split_dir in split_dirs.items():
        relabel = split == "train"
        data = process_dir(split_dir, relabel=relabel)

        print(f">>> parsed {split}, contains {len(data)} samples")

        if not args.test_mode:
            save_fp = osp.join(save_root, f"{split}.json")
            with open(save_fp, "w") as f:
                json.dump(data, f, indent=4)
        else:
            print(">>> skipped save")
