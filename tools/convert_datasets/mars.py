#!/usr/bin/env python3

"""MARS

Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_

Dataset statistics:
    - identities: 1261.
    - tracklets: 8298 (train) + 1980 (query) + 9330 (gallery).
    - cameras: 6.
"""

import argparse
import json
import os.path as osp

from scipy.io import loadmat

from mmcv.utils import mkdir_or_exist

DATASET_DIR = "mars"
JUNK_PIDS = [-1]
BACKGROUND_PID = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Convert/format mars dataset")
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


def get_names(fpath):
    names = []
    with open(fpath, "r") as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names


def parse_mars(names, img_dir, meta_data, relabel=False, min_seq_len=0):
    # tracks meta_data [start_index, end_index, pid, camid]
    num_tracklets = meta_data.shape[0]
    pid_list = list(set(meta_data[:, 2]))

    if relabel:
        pid2label = {pid: label for label, pid, in enumerate(pid_list)}
    tracklets = []

    for tracklet_idx in range(num_tracklets):
        data = meta_data[tracklet_idx, ...]
        start_index, end_index, pid, camid = data

        if pid == -1:
            continue  # junk images are ignored

        assert 1 <= camid <= 6
        if relabel:
            pid = pid2label[pid]

        camid -= 1
        img_names = names[start_index - 1 : end_index]

        # make sure images  names correspond to the same person
        pnames = [img_name[:4] for img_name in img_names]
        assert len(set(pnames)) == 1

        # make sure all images are captured under the same camera
        camnames = [img_name[5] for img_name in img_names]
        assert len(set(camnames)) == 1

        # append image names with directory information
        img_paths = [
            osp.join(img_dir, img_name[:4], img_name) for img_name in img_names
        ]
        if len(img_paths) >= min_seq_len:
            tracklets.append(
                dict(
                    pid=int(pid),
                    camid=int(camid),
                    img_paths=img_paths,
                    tracklet_length=len(img_paths),
                )
            )

    return tracklets


if __name__ == "__main__":

    args = parse_args()

    # Hard-coded variables:
    train_dir = "bbox_train"
    test_dir = "bbox_test"

    assert osp.exists(args.root)

    train_name_path = osp.join(args.root, "info", "train_name.txt")
    test_name_path = osp.join(args.root, "info", "test_name.txt")
    tracks_train_path = osp.join(args.root, "info", "tracks_train_info.mat")
    tracks_test_path = osp.join(args.root, "info", "tracks_test_info.mat")
    query_idx_path = osp.join(args.root, "info", "query_IDX.mat")

    assert osp.exists(train_name_path)
    assert osp.exists(test_name_path)
    assert osp.exists(tracks_train_path)
    assert osp.exists(tracks_test_path)
    assert osp.exists(query_idx_path)

    # get names
    train_names = get_names(train_name_path)
    test_names = get_names(test_name_path)

    # load .mat
    # (8298, 4)
    tracks_train = loadmat(tracks_train_path)["track_train_info"]
    # (12180, 4)
    tracks_test = loadmat(tracks_test_path)["track_test_info"]
    # (1980,)
    query_idx = loadmat(query_idx_path)["query_IDX"].squeeze()
    query_idx -= 1  # index from 0
    gallery_idx = [i for i in range(tracks_test.shape[0]) if i not in query_idx]

    tracks_query = tracks_test[query_idx, :]
    tracks_gallery = tracks_test[gallery_idx, :]

    train_data = parse_mars(train_names, train_dir, tracks_train, relabel=True)
    query_data = parse_mars(test_names, test_dir, tracks_query, relabel=False)
    gallery_data = parse_mars(
        test_names, test_dir, tracks_gallery, relabel=False
    )

    # for this dataset, there are no preprocessing for the images,
    # just getting an annotation file for unified loading

    save_root = osp.join(args.root, args.out_dir)
    mkdir_or_exist(save_root)

    if not args.test_mode:
        # save data as json file
        train_fp = osp.join(save_root, "train.json")
        with open(train_fp, "w") as f:
            json.dump(train_data, f, indent=4)

        query_fp = osp.join(save_root, "query.json")
        with open(query_fp, "w") as f:
            json.dump(query_data, f, indent=4)

        gallery_fp = osp.join(save_root, "gallery.json")
        with open(gallery_fp, "w") as f:
            json.dump(gallery_data, f, indent=4)
