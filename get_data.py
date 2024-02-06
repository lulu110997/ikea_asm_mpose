from mpose_pkg.mpose import MPOSE

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



load_mpose('openpose_coco', 1)