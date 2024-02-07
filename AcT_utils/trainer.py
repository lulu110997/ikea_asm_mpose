# GENERAL LIBRARIES 
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import joblib
# MACHINE LEARNING LIBRARIES
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES
from AcT_utils import TransformerEncoder, PatchClassEmbedding
from AcT_utils import load_mpose, random_flip, random_noise, one_hot
from AcT_utils import CustomSchedule
from AcT_utils.data import LABELS
from AcT_utils import pickle_wrapper as _pw

# TRAINER CLASS
class Trainer:
    def __init__(self, config, logger, model_sz, split=1):

        self.logger = logger
        self.split = split
        self.trial = None
        self.bin_path = config['MODEL_DIR']
        
        self.model_size = model_sz #self.config['MODEL_SIZE']
        self.n_heads = config[self.model_size]['N_HEADS']
        self.n_layers = config[self.model_size]['N_LAYERS']
        self.embed_dim = config[self.model_size]['EMBED_DIM']  # Size of embedded input. dv = 64, made constant according to paper
        self.dropout = config[self.model_size]['DROPOUT']
        self.mlp_head_size = config[self.model_size]['MLP']  # Output size of the ff layer prior the classification layer
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = self.d_model * 4  # Output size of the first non-linear layer in the transformer encoder
        self.pos_emb = config['POS_EMB']
        assert self.d_model == self.embed_dim  # Should be the same

        self.DATASET = config["DATASET"]
        self.DATA_TYPE = config['DATA_TYPE']
        self.SCHEDULER = config["SCHEDULER"]
        self.N_EPOCHS = config["N_EPOCHS"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.WEIGHT_DECAY = config["WEIGHT_DECAY"]
        self.WARMUP_PERC = config["WARMUP_PERC"]
        self.STEP_PERC = config["STEP_PERC"]
        self.N_FOLD = config["FOLDS"]
        self.N_SPLITS = config["SPLITS"]
        self.LABELS = config["LABELS"]
        self.N_FRAMES = config[self.DATASET]["FRAMES"]
        self.N_CLASSES = config[self.DATASET]["CLASSES"]
        self.N_KEYPOINTS = config[self.DATASET]["KEYPOINTS"]
        self.FEATURES_PER_KP = 4

        self.results_dir = config['RESULTS_DIR']
        self.weights_path = config["WEIGHTS"]

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.N_FRAMES, self.N_KEYPOINTS * self.FEATURES_PER_KP))
        x = tf.keras.layers.Dense(self.d_model)(inputs)  # Projection layer
        x = PatchClassEmbedding(self.d_model, self.N_FRAMES, pos_emb=None)(x)  # Positional embedding layer
        x = transformer(x)  # Transformer
        x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)  # Obtain cls
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)  # Dense layer
        outputs = tf.keras.layers.Dense(self.N_CLASSES)(x)  # Classification layer
        return tf.keras.models.Model(inputs, outputs)

    
    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = self.build_act(transformer)
        
        self.train_steps = np.ceil(float(self.train_len)/self.BATCH_SIZE)
        self.test_steps = np.ceil(float(self.test_len)/self.BATCH_SIZE)


        lr = CustomSchedule(self.d_model,
                            warmup_steps=self.train_steps*self.N_EPOCHS*self.WARMUP_PERC,
                            decay_step=self.train_steps*self.N_EPOCHS*self.STEP_PERC)
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.WEIGHT_DECAY)

        self.model.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

        self.name_model_bin = f"{self.model_size}_{self.DATA_TYPE}.h5"

        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.bin_path + self.name_model_bin,
                                                               monitor="val_loss",
                                                               mode="min",
                                                               save_best_only=True,
                                                               save_weights_only=True)
        return
    
    def get_data(self):

        X_train, y_train, X_test, y_test = load_mpose(self.DATASET, self.split, legacy=False, verbose=False)
        self.train_len = len(y_train)
        self.test_len = len(y_test)

        # Count how many times each label appears in the training data
        weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced",
                                                                  classes=np.unique(y_train), y=y_train)
        self.class_weights = {i: w for i, w in enumerate(weights)}
        self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        self.ds_train = self.ds_train.map(lambda x,y : one_hot(x, y, self.N_CLASSES),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(X_train.shape[0])
        self.ds_train = self.ds_train.batch(self.BATCH_SIZE)
        self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test = self.ds_test.map(lambda x,y : one_hot(x, y, self.N_CLASSES),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.batch(self.BATCH_SIZE)
        self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    def get_random_hp(self):
        # self.config['RN_STD'] = self.trial.suggest_discrete_uniform("RN_STD", 0.0, 0.03, 0.01)
        self.WEIGHT_DECAY = self.trial.suggest_discrete_uniform("WD", 1e-5, 1e-3, 1e-5)
        self.N_EPOCHS = int(self.trial.suggest_discrete_uniform("EPOCHS", 50, 100, 10))
        self.config['WARMUP_PERC'] = self.trial.suggest_discrete_uniform("WARMUP_PERC", 0.1, 0.4, 0.1)
        self.config['LR_MULT'] = self.trial.suggest_discrete_uniform("LR_MULT", -5, -4, 1)
        self.config['SUBSAMPLE'] = int(self.trial.suggest_discrete_uniform("SUBSAMPLE", 4, 8, 4))
        self.config['SCHEDULER'] = self.trial.suggest_categorical("SCHEDULER", [False, False])
        
        # self.logger.save_log('\nRN_STD: {:.2e}'.format(self.config['RN_STD']))
        self.logger.save_log('\nEPOCHS: {}'.format(self.config['N_EPOCHS']))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.config['WARMUP_PERC']))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}'.format(self.config['WEIGHT_DECAY']))
        self.logger.save_log('LR_MULT: {:.2e}'.format(self.config['LR_MULT']))
        self.logger.save_log('SUBSAMPLE: {}'.format(self.config['SUBSAMPLE']))
        self.logger.save_log('SCHEDULER: {}\n'.format(self.config['SCHEDULER']))
        
    def do_training(self):
        self.get_data()
        self.get_model()
        self.model.fit(self.ds_train,
                       epochs=self.N_EPOCHS, initial_epoch=0,
                       validation_data=self.ds_test,
                       callbacks=[self.checkpointer],
                       class_weight=self.class_weights)
        accuracy_test, balanced_accuracy = self.evaluate(weights=self.bin_path+self.name_model_bin)      
        return accuracy_test, balanced_accuracy

    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_weights(self.bin_path+self.name_model_bin)  
        else:
            self.model.load_weights(self.weights_path)

        loss, accuracy_test = self.model.evaluate(self.ds_test, steps=self.test_steps)

        X, y = tuple(zip(*self.ds_test))
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1),axis=1)
        y_true = tf.math.argmax(tf.concat(y, axis=0), axis=1)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        # conf_matr = sklearn.metrics.confusion_matrix(y_true, y_pred, LABELS)
        # _pw.save_pickle(f"model_{self.model_size}_data_{self.DATA_TYPE}", conf_matr)
        text = f"Accuracy Test: {accuracy_test} <> Balanced Accuracy: {balanced_accuracy}\n"
        self.logger.save_log(text)
        
        return accuracy_test, balanced_accuracy

    def do_test(self):
        for split in range(1, self.N_SPLITS + 1):
            self.logger.save_log(f"model {self.model_size} is being trained")
            self.split = split

            self.get_data()
            self.get_model()
            acc, bal_acc = self.evaluate()

            self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
            self.logger.save_log(f"Accuracy: {acc}")
            self.logger.save_log(f"Balanced Accuracy: {bal_acc}")

    def objective(self, trial):
        self.trial = trial     
        self.get_random_hp()
        _, bal_acc = self.do_training()
        return bal_acc
        
    def do_benchmark(self):
        for split in range(1, self.N_SPLITS+1):
            self.logger.save_log(f"model {self.model_size} is being trained")

            self.split = split

            acc, bal_acc = self.do_training()
                
            np.save(self.results_dir + self.DATA_TYPE + '_' + self.DATASET + f'_{split}_accuracy.npy', acc)
            np.save(self.results_dir + self.DATA_TYPE + '_' + self.DATASET + f'_{split}_balanced_accuracy.npy', bal_acc)

            self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
            self.logger.save_log(f"Accuracy: {acc}")
            self.logger.save_log(f"Balanced Accuracy: {bal_acc}")

    def do_random_search(self):
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.config['MODEL_NAME']),
                                         direction="maximize", pruner=pruner)
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.config['N_TRIALS'])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")

        self.logger.save_log("Best trial:")

        self.logger.save_log(f"  Value: {self.study.best_trial.value}")

        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        joblib.dump(self.study,
          f"{self.results_dir}/{self.config['MODEL_NAME']}_{self.config['DATASET']}_random_search_{str(self.study.best_trial.value)}.pkl")
        
    def return_model(self):
        return self.model
