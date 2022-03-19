#!/usr/bin/env python3

"""Market1501

Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

URL: `<http://www.liangzheng.org/Project/project_reid.html>`
EXTRA: `<https://www.kaggle.com/drmatters/distractors-500k>`

Dataset statistics:
    - identities: 1501 (+1 for background).
    - images: 12936 (train) + 3368 (query) + 15913 (gallery).
"""

import argparse
import json
import os.path as osp
import re

from mmcv.utils import scandir, mkdir_or_exist

DATASET_DIR = "market1501/Market-1501-v15.09.15"
DATASET_URL = "http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip"
EXTRA_DATASET_URL = "http://188.138.127.15:81/Datasets/distractors_500k.zip"
JUNK_PIDS = [0, -1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert/format market1501 dataset"
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
        "--nproc",
        default=4,
        type=int,
        help="number of processes",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def parse_market1501(image_paths):

    pattern = re.compile(r"([-\d]+)_c(\d)")

    def _split(path):
        _pid, _camid = map(int, pattern.search(path).groups())
        return _pid, _camid

    persons = []

    for img_path in image_paths:
        pid, camid = _split(img_path)
        persons.append(
            dict(
                pid=pid,
                camid=camid,
                img_path=img_path,
            )
        )

    return persons


if __name__ == "__main__":

    args = parse_args()

    # Hard-coded variables:
    img_suffix = ".jpg"

    assert osp.exists(args.root)

    train_path = osp.join(args.root, "bounding_box_train")
    query_path = osp.join(args.root, "query")
    gallery_path = osp.join(args.root, "bounding_box_test")
    assert osp.exists(train_path)
    assert osp.exists(query_path)
    assert osp.exists(gallery_path)

    split_paths = dict(
        train=train_path,
        query=query_path,
        gallery=gallery_path,
    )
    if args.extra:
        extra_path = osp.join(args.root, "images")  # distractors
        assert osp.exists(extra_path)
        split_paths.update(dict(extra=extra_path))

    # for this dataset, there are no preprocessing for the images,
    # just getting an annotation file for unified loading

    save_root = osp.join(args.root, args.out_dir)
    mkdir_or_exist(save_root)

    # create a list of dict
    for split, split_path in split_paths.items():
        img_paths = scandir(split_path, suffix=img_suffix)
        data = parse_market1501(img_paths)

        print(f">>> parsed {split}, contains {len(data)} samples")

        if not args.test_mode:
            # save data as json file
            save_fp = osp.join(save_root, f"{split}.json")
            with open(save_fp, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(">>> skipped save")
