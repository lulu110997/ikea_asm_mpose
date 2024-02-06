import statistics

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


# load_mpose('openpose_coco', 1)

ik_y = np.load('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy')
new_ik_y = np.full(ik_y.shape[0], -1, dtype=np.long)

for idx, row in enumerate(ik_y):
    label = statistics.multimode(row)
    if len(label) > 1:
        if 0 in label:  # Remove 'none' label
            label.remove(0)
        if 17 in label:  # Remove 'other' label
            label.remove(17)
        if len(label) != 1:  # Either empty as it (only) contains [0,7]
            label = [0]
    new_ik_y[idx] = np.long(label[0])

if (new_ik_y==-1).any():
    raise "Error"

np.save('/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy', new_ik_y)