import numpy as np
import smote_variants as sv
import sklearn.datasets as datasets

# dataset= datasets.load_wine(); X, y= dataset['data'], dataset['target']
X_org = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/X_train.npy")
y = np.load("/home/louis/.ikea_asm_2d_pose/openpose_coco/1/y_train.npy")

oversampler= sv.MulticlassOversampling(oversampler='MSMOTE',
                                       oversampler_params={'random_state': 5,
                                                           'n_neighbors': 2,
                                                           'proportion': 0.6})

X_new = np.ones((1048, *X_org.shape[1:]), dtype=np.float32)
y_new = []
Xn = np.concatenate((X_org[y==9], X_org[y==5], X_org[y==8], X_org[y==4]), axis=0)
for i in range(Xn.shape[2]):
    X = Xn[:, :, i, :].reshape(Xn.shape[0], -1)
    y = np.concatenate((y[y==9], y[y==5], y[y==8], y[y==4]), axis=0)

    # X_samp and y_samp contain the oversampled dataset
    X_samp, y_samp= oversampler.sample(X, y)
    X_new[:, :, i, :] = (X_samp.reshape(X_samp.shape[0], 30, 3))
    y_new.append(y_samp)
y_f = np.concatenate(y_new)