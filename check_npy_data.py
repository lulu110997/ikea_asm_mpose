import statistics
import sys

import cv2
from mpose_pkg.mpose import MPOSE
import numpy as np
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

X_train = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/X_train.npy')
X_test = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/X_test.npy')
train_img_paths = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/vid_paths_train.npy')
y_train = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy')
y_test = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_test.npy')
test_img_paths = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/vid_paths_test.npy')

# TODO: figure out data distribution of clip labels
try:
    # Check We have the same number of samples for the training features and labels
    assert X_train.shape[0] == y_train.shape[0] and len(y_train.shape) == 1
    assert X_test.shape[0] == y_test.shape[0] and len(y_test.shape) == 1

    # Check the training features have the shape (n_frames=30, n_keypoints=18, n_coords=3)
    assert X_train.shape[1:] == (30, 18, 3)
    assert X_test.shape[1:] == (30, 18, 3)
except:
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

try:
    cv2.namedWindow("win")
    for sample_idx, frames in enumerate(X_train):
        label = LABELS[y_train[sample_idx]]
        for frame_idx, kp in enumerate(frames):
            frame1 = cv2.imread(train_img_paths[sample_idx][frame_idx])
            cv2.putText(frame1, label, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

            for joints in kp:
                cv2.circle(frame1, (int(joints[0]), int(joints[1])), 5, (0, 0, 255))
            cv2.imshow('win', frame1)
            cv2.waitKey(0)
except:
    cv2.destroyAllWindows()

