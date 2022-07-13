#!/usr/bin/env python3

"""MOT* preprocess script (* generally the same with other MOT datasets)

- no validation or testing splits (if you have some proposals, please send me a PR!)
- outputs both image and video ReID dataset splits
- MOTImageDataset and MOTVideoDataset provides ways to filter out based on criterias
  (which sequences, what vis_ratio, etc...)
"""

import argparse
import json
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

from mmcv.utils import mkdir_or_exist

from tools.convert_datasets.mot_common import (
    crop_person,
    get_frame_paths,
    get_gts,
)

# Globals
DATASET_DIR = "MOT16"
DET_FILE = "{root}/{split}/MOT16-{seq}/det/det.txt"
GT_FILE = "{root}/{split}/MOT16-{seq}/gt/gt.txt"
IMG_DIR = "{root}/{split}/MOT16-{seq}/img1"
SEQUENCES = ("02", "04", "05", "09", "10", "11", "13")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert/format MOT16 dataset")
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


if __name__ == "__main__":

    args = parse_args()

    save_path_tmp = "{root}/{save_root}/{seq}/{pid}_{frame}.jpg"

    assert osp.exists(args.root)

    mkdir_or_exist(osp.join(args.root, args.out_dir))

    # save all images
    total_pids = 0
    img_metas = []
    vid_metas = []
    for seq in SEQUENCES:

        meta = []
        vid_data = {}

        gt_file = GT_FILE.format(
            root=args.root,
            split="train",
            seq=seq,
        )
        gts = get_gts(gt_file)

        img_dir = IMG_DIR.format(
            root=args.root,
            split="train",
            seq=seq,
        )
        frame_paths = get_frame_paths(img_dir)

        pids = {}
        max_pids = 1

        for frame_path in tqdm(frame_paths):

            frame_img = cv2.imread(frame_path)

            # get frame id (frame starts with 1 and not 0)
            frame = int(osp.split(frame_path)[-1].split(".")[0])

            # filter out
            frame_gts = gts[(gts.frame == frame) & (gts.is_ped == 1)]
            # for not, just save all (including vis_ratio==0)
            # and refine it later
            # the saved crops will be used for video dataset too

            for _, person_gt in frame_gts.iterrows():
                # bbox: [x1, y1, w, h]
                bbox = np.array(person_gt[2:6]).astype("i").copy()
                # pid:
                pid = int(person_gt["id"])
                if pid in pids.keys():
                    # seen pid
                    pid = pids[pid]
                else:
                    # new pid
                    pids[pid] = max_pids + total_pids
                    max_pids += 1
                    pid = pids[pid]

                # vis: ratio
                vis_ratio = float(person_gt["vis_ratio"])

                # string_format
                frame_str = str(frame).zfill(6)
                pid_str = str(pid).zfill(6)

                # save crop (convert to xxyxy)
                bbox[2:] += bbox[:2]
                bbox[bbox < 0] = 0

                # create save path for crop
                img_save_path = save_path_tmp.format(
                    root=args.root,
                    save_root=args.out_dir,
                    seq=seq,
                    pid=pid_str,
                    frame=frame_str,
                )

                # crop and save
                if not args.test_mode:
                    mkdir_or_exist(osp.dirname(img_save_path))
                    cropped_img = crop_person(
                        img=frame_img,
                        bbox=bbox,
                        save_path=img_save_path,
                    )

                    if cropped_img is None:
                        continue

                meta.append(
                    dict(
                        pid=pid,
                        camid=None,
                        img_path=img_save_path,
                        seq=seq,
                        vis_ratio=vis_ratio,
                        frame=frame,
                    )
                )

                if pid in vid_data.keys():
                    vid_data[pid].append(
                        dict(
                            frame=frame,
                            vis_ratio=vis_ratio,
                            img_path=img_save_path,
                        )
                    )
                else:
                    vid_data[pid] = [
                        dict(
                            frame=frame,
                            vis_ratio=vis_ratio,
                            img_path=img_save_path,
                        )
                    ]

        num_unique = len(list(pids.keys()))

        total_pids += num_unique
        img_metas += meta

        # checks
        print("img seq", seq, num_unique)
        assert total_pids == max(
            np.unique(np.array([m["pid"] for m in img_metas]))
        )

        vid_meta = []

        # parameters
        max_gap = 16
        min_track_len = 8
        max_track_len = 128

        # create video dataset
        for pid, data in vid_data.items():
            tracklets = []
            tracklet = []

            sorted_data = sorted(data, key=lambda d: d["frame"])

            frame_ids = [d["frame"] for d in sorted_data]
            img_paths = [d["img_path"] for d in sorted_data]
            vis_ratios = [d["vis_ratio"] for d in sorted_data]

            prev_fid = -1
            for fid, img_path, vis_ratio in zip(
                frame_ids, img_paths, vis_ratios
            ):
                if len(tracklet) > 0:
                    if fid - prev_fid > max_gap:
                        if len(tracklet) >= min_track_len:
                            tracklets.append(tracklet)
                        tracklet = []
                    if len(tracklet) >= max_track_len:
                        tracklets.append(tracklet)
                        tracklet = []

                tracklet.append(dict(img_path=img_path, vis_ratio=vis_ratio))
                prev_fid = fid

            # last tracklet
            if len(tracklet) >= min_track_len:
                tracklets.append(tracklet)

            # add tracks to meta
            for track in tracklets:
                sorted_track = sorted(track, key=lambda d: d["img_path"])
                tracks = [d["img_path"] for d in sorted_track]
                vrs = [d["vis_ratio"] for d in sorted_track]
                vid_meta.append(
                    dict(
                        pid=pid,
                        camid=None,
                        seq=seq,
                        img_paths=tracks,
                        vis_ratios=vrs,
                    )
                )

        print(
            "vid seq",
            seq,
            len(np.unique(np.array([m["pid"] for m in vid_meta]))),
        )

        vid_metas += vid_meta

    # do some checks!
    assert total_pids == max(np.unique(np.array([m["pid"] for m in img_metas])))
    print("total pids", total_pids)

    print("total img samples", len(img_metas))
    print("total video samples", len(vid_metas))

    # dump metas
    if args.test_mode:
        print(">>> skipped save")
    else:
        save_fp = osp.join(args.root, args.out_dir, "train_img.json")
        with open(save_fp, "w") as f:
            json.dump(img_metas, f, indent=4)

        save_fp = osp.join(args.root, args.out_dir, "train_vid.json")
        with open(save_fp, "w") as f:
            json.dump(vid_metas, f, indent=4)
