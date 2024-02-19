import os.path
import sys

import numpy as np
import smote_variants as sv
import logging
logging.getLogger('smote_variants').setLevel(logging.ERROR)
import sklearn.datasets as datasets

# dataset= datasets.load_wine(); X, y= dataset['data'], dataset['target']
X_org = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/X_train.npy")
y_org = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy")
SAVE_DIR = "/home/louis/.ikea_asm_2d_pose/openpose_coco/SMOTE_AMSR_all"
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

print(sv.get_multiclass_oversamplers())#; sys.exit()

oversampler= sv.MulticlassOversampling(oversampler='SMOTE_AMSR',
                                       oversampler_params={'random_state': 9})#, 'proportion': 0.6})

X_new = np.ones((144662, *X_org.shape[1:]), dtype=np.float32)
y_new = []

for i in range(X_org.shape[2]):
    X = X_org[:, :, i, :].reshape(X_org.shape[0], -1)

    # X_samp and y_samp contain the oversampled dataset
    X_samp, y_samp= oversampler.sample(X, y_org)
    X_new[:, :, i, :] = (X_samp.reshape(X_samp.shape[0], 30, 3))

y_new.append(y_samp); y_f = np.concatenate(y_new)

assert X_new.shape[0] == y_f.shape[0]

np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_new)
np.save(os.path.join(SAVE_DIR, 'y_train.npy'), y_f)