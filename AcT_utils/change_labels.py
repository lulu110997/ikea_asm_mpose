import sys

import numpy as np
import os

PATH = '/home/louis/.ikea_asm_2d_pose/openpose_coco/'
SPLIT = '1'

NA = [(0,), 0]
ALIGN = [(1, 2), 1]
ATTACH = [(3, 4, 5), 2]
FLIP = [(6, 7, 8), 3]
INSERT = [(9,), 4]
LAY_DOWN = [(10, 11, 12, 13, 14, 15, 16), 5]
OTHER = [(17,), 6]
PICK_UP = [(18, 19, 20, 21, 22, 23, 24, 25), 7]
POSITION = [(26,), 8]
PUSH = [(27, 28), 9]
ROTATE = [(29,), 10]
SLIDE = [(30,), 11]
SPIN = [(31,), 12]
TIGHTEN = [(32,), 13]
ACTIONS = [NA, ALIGN, ATTACH, FLIP, INSERT, LAY_DOWN, OTHER, PICK_UP, POSITION, PUSH, ROTATE, SLIDE, SPIN, TIGHTEN]

for set in ['train', 'test']:
    labels_file = os.path.join(PATH, SPLIT, f"y_{set}_VO.npy")
    labels = np.load(labels_file)
    new_labels = np.full(labels.shape, -1, dtype=np.int64)
    for idx, sample in enumerate(labels):
        for a in ACTIONS:
            if sample in a[0]:
                new_labels[idx] = a[1]
                break

    assert not any(new_labels == -1)
    np.save(os.path.join(PATH, SPLIT, f"y_{set}.npy"), new_labels)