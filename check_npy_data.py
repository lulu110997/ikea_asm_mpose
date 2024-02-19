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
tf.keras.utils.set_random_seed(9)

LABELS_VO = ['NA', 'align leg screw with table thread', 'align side panel holes with front panel dowels',
             'attach drawer back panel', 'attach drawer side panel', 'attach shelf to table', 'flip shelf',
             'flip table', 'flip table top', 'insert drawer pin', 'lay down back panel', 'lay down bottom panel',
             'lay down front panel', 'lay down leg', 'lay down shelf', 'lay down side panel', 'lay down table top',
             'other', 'pick up back panel', 'pick up bottom panel', 'pick up front panel', 'pick up leg', 'pick up pin',
             'pick up shelf', 'pick up side panel', 'pick up table top', 'position the drawer right side up', 'push table',
             'push table top', 'rotate table', 'slide bottom of drawer', 'spin leg', 'tighten leg']

LABELS_V = ['NA', 'align', 'attach', 'flip', 'insert', 'lay down', 'other', 'pick up', 'position', 'push', 'rotate',
            'slide', 'spin', 'tighten']
ROOT_DIR = "/home/louis/.ikea_asm_2d_pose/openpose_coco/"
SPLIT = "1"
LABELS = LABELS_V
N_CLASSES = len(LABELS)
def viz_imgs_with_xy(x, y, img_path, label_to_check=None, seed=None, save=False):
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
                curr_path = img_path[sample_idx][frame_idx]
                frame1 = cv2.imread(curr_path)
                cv2.putText(frame1, label, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

                for kp_idx, coords in enumerate(kp):
                    cv2.circle(frame1, (int(coords[0]), int(coords[1])), 5, (0, 0, 255))
                    cv2.putText(frame1, str(kp_idx), (int(coords[0]), int(coords[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

                cv2.putText(frame1, str(frame_count), (20, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                if not save:
                    cv2.imshow('win', frame1)
                    cv2.waitKey(x.shape[1]*3)
                else:
                    new_path, tmp = curr_path.split('ikea_asm_dataset_RGB_top_frames')
                    new_path = os.path.join(new_path, "labelled_images")
                    if not os.path.exists(new_path):
                        os.mkdir(new_path)
                    cv2.imwrite(os.path.join(new_path, tmp.replace("/", "_")), frame1)
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
    return labels, counts


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
            assert np.unique(y_train).shape[0] == 33 or np.unique(y_train).shape[0] == 14

        # Check the training features have the shape (n_frames=30, n_keypoints=18, n_coords=3)
        assert X_train.shape[1:] == (30, 18, 3)
        assert X_test.shape[1:] == (30, 18, 3)
        if not ("clean" in SPLIT):
            assert np.unique(y_test).shape[0] == 33 or np.unique(y_test).shape[0] == 14
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
    __, counts = get_label_counts(y)

    total_label_counts = sum(counts)
    dist = []
    for i in list(zip(LABELS, counts)):
        dist.append(i[1] / total_label_counts)
        print(f"{i[0]}, {i[1]}, {i[1]/total_label_counts}")
    return dist


########################################################################################################################
X_train, y_train, X_test, y_test, train_img_paths, test_img_paths = get_data()
viz_imgs_with_xy(X_train, y_train, train_img_paths, label_to_check='push')
# viz_imgs_with_xy(X_test, y_test, test_img_paths, label_to_check='push')

# X_train_list = []
# y_train_list = []
# video_train_list = []
# ds = []
# class_weights = sklearn.utils.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
# THRESH = 500
# for i in range(N_CLASSES):
#     mask = np.where(y_train == i)
#     curr_x = X_train[mask]
#     curr_y = y_train[mask]
#     curr_v = train_img_paths[mask]
#
#
#     X_train_list.append(curr_x)
#     y_train_list.append(curr_y)
#     video_train_list.append(curr_v)
#     tmp_ds = tf.data.Dataset.from_tensor_slices((X_train[mask], y_train[mask]))
#     rep_val = THRESH - tmp_ds.cardinality()
#     tmp_ds = tmp_ds.shuffle(tmp_ds.cardinality()//2, reshuffle_each_iteration=True).repeat()
#
#     ds.append(tmp_ds)
#
# resampled_ds = tf.data.Dataset.sample_from_datasets(ds, stop_on_empty_dataset=True, rerandomize_each_iteration=True)
#
# for a in range(5):
#     label_count = N_CLASSES*[0]
#     c = 0
#     for i in resampled_ds.as_numpy_iterator():
#         if c==0:
#             print(i[0][0][0])
#             c=1
#         if max(label_count) == THRESH:
#             break
#         label_count[i[1]] += 1
#     print(label_count, sum(label_count))

