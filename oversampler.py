import os.path

import numpy as np
import smote_variants as sv
import sklearn.datasets as datasets

# dataset= datasets.load_wine(); X, y= dataset['data'], dataset['target']
X_org = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/X_train.npy")
y_org = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy")
SAVE_DIR = "/home/louis/.ikea_asm_2d_pose/openpose_coco/SMOTED2"
NOPE = [4,5,8,9]
oversampler= sv.MulticlassOversampling(oversampler='distance_SMOTE',
                                       oversampler_params={'random_state': 9,
                                                           'proportion': 0.6})

X_new = np.ones((1048, *X_org.shape[1:]), dtype=np.float32)
y_new = []
Xn = np.concatenate((X_org[y_org==9], X_org[y_org==5], X_org[y_org==8], X_org[y_org==4]), axis=0)
for i in range(Xn.shape[2]):
    X = Xn[:, :, i, :].reshape(Xn.shape[0], -1)
    y = np.concatenate((y_org[y_org==9], y_org[y_org==5], y_org[y_org==8], y_org[y_org==4]), axis=0)

    # X_samp and y_samp contain the oversampled dataset
    X_samp, y_samp= oversampler.sample(X, y)
    X_new[:, :, i, :] = (X_samp.reshape(X_samp.shape[0], 30, 3))
y_new.append(y_samp)
y_f = np.concatenate(y_new)
X_avoided = []
y_avoided = []
for i in range(14):
    if i in NOPE:
        continue
    X_avoided.append(X_org[y_org==i])
    y_avoided.append(y_org[y_org==i])
X_avoided_np = np.concatenate(X_avoided)
y_avoided_np = np.concatenate(y_avoided)
X_save_me = np.concatenate((X_avoided_np, X_new), axis=0)
y_save_me = np.concatenate((y_avoided_np, y_f))
assert X_save_me.shape[0] == (X_org.shape[0] - Xn.shape[0] + X_new.shape[0])
assert X_save_me.shape[0] == y_save_me.shape[0]

np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_save_me)
np.save(os.path.join(SAVE_DIR, 'y_train.npy'), y_save_me)