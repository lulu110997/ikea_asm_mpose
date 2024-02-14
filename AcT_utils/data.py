# Copyright 2021 Simone Angarano. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from mpose_pkg.mpose import MPOSE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


LABELS = {'NA': 0,
          'align leg screw with table thread': 1,
          'align side panel holes with front panel dowels': 2,
          'attach drawer back panel': 3,
          'attach drawer side panel': 4,
          'attach shelf to table': 5,
          'flip shelf': 6,
          'flip table': 7,
          'flip table top': 8,
          'insert drawer pin': 9,
          'lay down back panel': 10,
          'lay down bottom panel': 11,
          'lay down front panel': 12,
          'lay down leg': 13,
          'lay down shelf': 14,
          'lay down side panel': 15,
          'lay down table top': 16,
          'other': 17,
          'pick up back panel': 18,
          'pick up bottom panel': 19,
          'pick up front panel': 20,
          'pick up leg': 21,
          'pick up pin': 22,
          'pick up shelf':23,
          'pick up side panel': 24,
          'pick up table top': 25,
          'position the drawer right side up': 26,
          'push table': 27,
          'push table top': 28,
          'rotate table': 29,
          'slide bottom of drawer': 30,
          'spin leg': 31,
          'tighten leg': 32
          }


def load_mpose(dataset, split, velocity):
    
    d = MPOSE(pose_extractor=dataset, 
                    split=split, 
                    preprocess=None, 
                    velocities=velocity,
                    remove_zip=False)

    d.reduce_keypoints()
    d.scale_and_center()
    d.remove_confidence()
    d.flatten_features()
    #d.reduce_labels()
    return d.get_data()
        

def random_flip(x, y):
    time_steps = x.shape[0]
    n_features = x.shape[1]
    if not n_features % 2:
        x = tf.reshape(x, (time_steps, n_features//2, 2))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0])
    else:
        x = tf.reshape(x, (time_steps, n_features//3, 3))

        choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        if choice >= 0.5:
            x = tf.math.multiply(x, [-1.0,1.0,1.0])
    x = tf.reshape(x, (time_steps,-1))
    return x, y


def random_noise(x, y):
    time_steps = tf.shape(x)[0]
    n_features = tf.shape(x)[1]
    noise = tf.random.normal((time_steps, n_features), mean=0.0, stddev=0.03, dtype=tf.float32)
    x = x + noise
    return x, y


def one_hot(x, y, n_classes):
    return x, tf.one_hot(y, n_classes)


def kinetics_generator(X, y, batch_size):
    while True:
        ind_list = [i for i in range(X.shape[0])]
        shuffle(ind_list)
        X  = X[ind_list,...]
        y = y[ind_list]
        for count in range(len(y)):
            yield (X[count], y[count])
        
        
def callable_gen(_gen):
        def gen():
            for x,y in _gen:
                 yield x,y
        return gen
    
    
def transform_labels(y):
    y_new = []
    for i in y:
        y_new.append(labels[i])
    return np.array(y_new)


def load_dataset_legacy(data_folder, verbose=True):
    X_train = np.load(data_folder + 'X_train.npy')
    y_train = np.load(data_folder + 'Y_train.npy', allow_pickle=True)
    y_train = transform_labels(y_train)
    
    X_test = np.load(data_folder + 'X_test.npy')
    y_test = np.load(data_folder + 'Y_test.npy', allow_pickle=True)
    y_test = transform_labels(y_test)
    
    if verbose:
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test