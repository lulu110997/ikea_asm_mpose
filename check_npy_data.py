import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import statistics
import sys
import sklearn
import cv2
from mpose_pkg.mpose import MPOSE
import numpy as np
import tensorflow as tf

LABELS = ['NA',
          'align leg screw with table thread',
          'align side panel holes with front panel dowels',
          'attach drawer back panel',
          'attach drawer side panel',
          'attach shelf to table',
          'flip shelf',
          'flip table',
          'flip table top',
          'insert drawer pin',
          'lay down back panel',
          'lay down bottom panel',
          'lay down front panel',
          'lay down leg',
          'lay down shelf',
          'lay down side panel',
          'lay down table top',
          'other',
          'pick up back panel',
          'pick up bottom panel',
          'pick up front panel',
          'pick up leg',
          'pick up pin',
          'pick up shelf',
          'pick up side panel',
          'pick up table top',
          'position the drawer right side up',
          'push table',
          'push table top',
          'rotate table',
          'slide bottom of drawer',
          'spin leg',
          'tighten leg'
          ]
ROOT_DIR = "/home/louis/.ikea_asm_2d_pose/openpose_coco/"
SPLIT = "1"


def check_xy(x, y, img_path, label_to_check=None, seed=9):
    if seed is not None:
        x, y, img_path = sklearn.utils.shuffle(x, y, img_path, random_state=seed)
    try:
        cv2.namedWindow("win")
        input("press a key when ready")
        for sample_idx, frames in enumerate(x):
            label = LABELS[y[sample_idx]]
            if (label_to_check is not None) and (label != label_to_check):
                continue
            frame_count = 0
            for frame_idx, kp in enumerate(frames):
                frame_count += 1
                frame1 = cv2.imread(img_path[sample_idx][frame_idx])
                cv2.putText(frame1, label, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

                for kp_idx, coords in enumerate(kp):
                    cv2.circle(frame1, (int(coords[0]), int(coords[1])), 5, (0, 0, 255))
                    cv2.putText(frame1, str(kp_idx), (int(coords[0]), int(coords[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(frame1, str(frame_count), (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                cv2.imshow('win', frame1)
                cv2.waitKey(2*x.shape[1])
    except Exception as e:
        print(e)
        cv2.destroyAllWindows()


def load_mpose(dataset, split, verbose=False, legacy=False):

    d = MPOSE(pose_extractor=dataset,
              split=split,
              preprocess=None,
              velocities=True,
              remove_zip=False)

    d.reduce_keypoints()
    d.scale_and_center()
    d.remove_confidence()
    d.flatten_features()
    return d.get_data()


def get_label_counts(y):
    """
    Obtain the occurrences for each label. Takes into account labels that are potentially missing
    Args:
        y: np ndarray | label array

    Returns: list of occurrences for each label

    """
    labels, counts = np.unique(y, return_counts=True)
    counts = counts.tolist()
    labels = labels.tolist()
    if len(labels) != len(LABELS):
        print(f"Missing {33-len(labels)} labels. Current labels and count: \n{list(zip(labels, counts))}")
        for idx, val in enumerate(labels):
            if idx != val:
                labels.insert(idx, idx)
                labels.sort()
                counts.insert(idx, 0)
        print(f"Final labels and count: \n{list(zip(labels, counts))}\n")
    return counts


def get_data():
    X_train = np.load(os.path.join(ROOT_DIR, SPLIT, 'X_train.npy'))
    y_train = np.load(os.path.join(ROOT_DIR, SPLIT, 'y_train.npy'))
    train_img_paths = np.load(os.path.join(ROOT_DIR, SPLIT, 'vid_paths_train.npy'))

    train_masks = np.where(y_train >= 0)
    X_train = X_train[train_masks[0]]
    y_train = y_train[train_masks[0]]
    train_img_paths = train_img_paths[train_masks[0]]

    X_test = np.load(os.path.join(ROOT_DIR, SPLIT, 'X_test.npy'))
    y_test = np.load(os.path.join(ROOT_DIR, SPLIT, 'y_test.npy'))
    test_img_paths = np.load(os.path.join(ROOT_DIR, SPLIT, 'vid_paths_test.npy'))

    test_masks = np.where(y_test >= 0)
    X_test = X_test[test_masks[0]]
    y_test = y_test[test_masks[0]]
    test_img_paths = test_img_paths[test_masks[0]]

    try:
        # Check We have the same number of samples for the training features and labels
        assert X_train.shape[0] == y_train.shape[0] and len(y_train.shape) == 1
        assert X_test.shape[0] == y_test.shape[0] and len(y_test.shape) == 1
        if not ("clean" in SPLIT):
            assert np.unique(y_train).shape[0] == 33

        # Check the training features have the shape (n_frames=30, n_keypoints=18, n_coords=3)
        assert X_train.shape[1:] == (30, 18, 3)
        assert X_test.shape[1:] == (30, 18, 3)
        if not ("clean" in SPLIT):
            assert np.unique(y_test).shape[0] == 33
    except Exception as e:
        print(X_train.shape)
        print(y_train.shape)
        print(train_img_paths)
        print(np.unique(y_train, return_counts=True))

        print(X_test.shape)
        print(y_test.shape)
        print(test_img_paths)
        print(np.unique(y_test, return_counts=True))
        raise e

    return X_train, y_train, X_test, y_test, train_img_paths, test_img_paths

def get_class_dist(y):
    counts = get_label_counts(y)

    total_label_counts = sum(counts)
    dist = []
    for i in list(zip(LABELS, counts)):
        # print(f"{i[0]}, {i[1]}, {i[1]/total_label_counts}")
        dist.append(i[1] / total_label_counts)
    return dist

########################################################################################################################
X_train, y_train, X_test, y_test, train_img_paths, test_img_paths = get_data()

check_xy(X_test, y_test, test_img_paths, seed=None)

X_train_list = []
y_train_list = []
video_train_list = []
ds = []

for i in range(33):
    mask = np.where(y_train == i)
    curr_x = X_train[mask]
    curr_y = y_train[mask]
    curr_v = train_img_paths[mask]

    rep_val = 400 - curr_x.shape[0]
    # if n_samples < 800:
    #     curr_x = np.repeat(curr_x, 800 - n_samples, axis=0)
    #     curr_y = np.repeat(curr_y, 800 - n_samples)
    #     curr_v = np.repeat(curr_v, 800 - n_samples, axis=0)

    X_train_list.append(curr_x)
    y_train_list.append(curr_y)
    video_train_list.append(curr_v)
    tmp_ds = tf.data.Dataset.from_tensor_slices((X_train[mask], y_train[mask]))
    if rep_val > 0:
        tmp_ds = tmp_ds.repeat(math.ceil(400.0 / curr_x.shape[0]))

    ds.append(tmp_ds)

resampled_ds = tf.data.experimental.sample_from_datasets(ds, weights=33*[0.5], stop_on_empty_dataset=True)

# print(len(list(ds_train.as_numpy_iterator())))
#

