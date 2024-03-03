# GENERAL LIBRARIES 
import os
import math
import sys
import time

import yaml
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import joblib
# MACHINE LEARNING LIBRARIES
import sklearn
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
# OPTUNA
import optuna
from optuna.trial import TrialState
# CUSTOM LIBRARIES
from AcT_utils import TransformerEncoder, PatchClassEmbedding
from AcT_utils import load_mpose, random_flip, random_noise, one_hot
from AcT_utils import CustomSchedule
from AcT_utils import pickle_wrapper as _pw
tf.random.set_seed(11)
np.random.seed(11)
random.seed(11)

class clean_up(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()

# TRAINER CLASS
class Trainer:
    def __init__(self, config, logger, model_sz, root_results_dir):

        self.trial = None
        self.norm_input = None
        self.logger = logger
        self.split = config["SUBDIR"]
        L2_PENALTY = 0.001
        self.l2_reg = tf.keras.regularizers.l2(L2_PENALTY); self.logger.save_log(f'l2: {L2_PENALTY}')
        self.model_size = model_sz; self.logger.save_log(self.model_size) #self.config['MODEL_SIZE'];
        self.n_heads = config[self.model_size]['N_HEADS']
        self.n_layers = config[self.model_size]['N_LAYERS']
        self.embed_dim = config[self.model_size]['EMBED_DIM']  # Size of embedded input. dv = 64, made constant according to paper
        self.dropout = config[self.model_size]['DROPOUT']
        self.mlp_head_size = config[self.model_size]['MLP']  # Output size of the ff layer prior the classification layer
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = 4 * self.d_model  # Output size of the first non-linear layer in the transformer encoder
        self.label_smoothing = config['LABEL_SMOOTHING']
        assert self.d_model == self.embed_dim  # Should be the same

        self.DATASET = config["DATASET"]
        self.DATA_TYPE = config['DATA_TYPE']
        self.SCHEDULER = config["SCHEDULER"]
        self.velocity = config["VELOCITY"]
        self.N_EPOCHS = config["N_EPOCHS"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.WEIGHT_DECAY = config["WEIGHT_DECAY"]
        self.WARMUP_PERC = config["WARMUP_PERC"]
        self.STEP_PERC = config["STEP_PERC"]
        self.LABELS = config["LABELS_V"]
        self.N_FRAMES = config[self.DATASET]["FRAMES"]
        self.N_KEYPOINTS = config[self.DATASET]["KEYPOINTS"]
        self.N_TRIALS = config["N_TRIALS"]
        self.FEATURES_PER_KP = 4 if self.velocity else 2
        self.N_CLASSES = len(self.LABELS)

        self.bin_path = root_results_dir[0]
        self.results_dir = root_results_dir[1]
        p = os.path.join(self.results_dir, "config_file.yaml")
        with open(p, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    def build_act(self, transformer):
        inputs = tf.keras.layers.Input(shape=(self.N_FRAMES, self.N_KEYPOINTS * self.FEATURES_PER_KP))
        if self.norm_input is None:
            x = tf.keras.layers.Dense(self.d_model, kernel_regularizer=self.l2_reg)(inputs)  # Projection layer
        else:
            x = self.norm_input(inputs); self.logger.save_log("normalise")
            x = tf.keras.layers.Dense(self.d_model, kernel_regularizer=self.l2_reg)(x)  # Projection layer

        x = PatchClassEmbedding(self.d_model, self.N_FRAMES, pos_emb=None)(x)  # Positional embedding layer
        x = transformer(x)  # Transformer
        x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)  # Obtain cls
        x = tf.keras.layers.Dense(self.mlp_head_size, kernel_regularizer=self.l2_reg)(x)  # Dense layer
        outputs = tf.keras.layers.Dense(self.N_CLASSES, kernel_regularizer=self.l2_reg)(x)  # Classification layer
        return tf.keras.models.Model(inputs, outputs)

    
    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = self.build_act(transformer)
        self.train_steps = np.ceil(float(self.train_len)/self.BATCH_SIZE)
        self.logger.save_log(f"train steps: {self.train_steps}")

        lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=5e-7,
                                                       decay_steps=self.STEP_PERC*self.N_EPOCHS*self.train_steps,
                                                       alpha=1e-2,
                                                       warmup_target=self.max_learning_rate,
                                                       warmup_steps=self.WARMUP_PERC*self.N_EPOCHS*self.train_steps)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.label_smoothing)
        optim = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=self.WEIGHT_DECAY)

        self.model.compile(optimizer=optim,
                           loss=loss,
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                                    tf.keras.metrics.F1Score(average="macro"),
                                    tf.keras.metrics.AUC(multi_label=True, from_logits=True, num_labels=self.N_CLASSES)])
        for layer in self.model.layers:
            self.logger.save_log(f"{layer.name}, {len(layer.trainable_weights)}, {len(layer.non_trainable_weights)}")
        self.logger.save_log("")
        self.name_model_bin = f"{self.model_size}_{self.DATA_TYPE}.h5"
        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.bin_path + self.name_model_bin,  #"{epoch:02d}_"
                                                               monitor="val_accuracy", mode="max",
                                                               save_best_only=True, save_weights_only=True)
        return

    def resample(self, x, y):
        self.logger.save_log("resampled")
        ds = []
        for i in range(self.N_CLASSES):
            mask = np.where(y == i)
            tmp_ds = tf.data.Dataset.from_tensor_slices((x[mask], y[mask]))
            BUFFER_SIZE = tmp_ds.cardinality() // 2
            tmp_ds = tmp_ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
            tmp_ds = tmp_ds.repeat()
            ds.append(tmp_ds)

        return tf.data.Dataset.sample_from_datasets(ds, stop_on_empty_dataset=True, rerandomize_each_iteration=True)
    def get_data(self):

        X_train, y_train, X_test, y_test = load_mpose(self.DATASET, self.split, velocity=self.velocity)
        self.norm_input = tf.keras.layers.Normalization(axis=None)
        self.norm_input.adapt(X_train)

        # self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)); self.train_len = len(y_train)
        self.ds_train = self.resample(X_train, y_train); self.train_len = 5628
        self.ds_train = self.ds_train.map(lambda x, y: one_hot(x, y, self.N_CLASSES),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.batch(self.BATCH_SIZE)
        self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        self.ds_test = self.ds_test.map(lambda x,y : one_hot(x, y, self.N_CLASSES),
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.batch(self.BATCH_SIZE)
        self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    def do_training(self):
        self.get_data()
        self.get_model()
        print("start_training")
        history = self.model.fit(self.ds_train, verbose=2,
                                 epochs=self.N_EPOCHS, initial_epoch=0,
                                 validation_data=self.ds_test, steps_per_epoch=self.train_steps,
                                 callbacks=[self.checkpointer, clean_up()])
        h = (history.history['loss'], history.history['val_loss'],
             history.history['accuracy'], history.history['val_accuracy'],
             history.history["f1_score"], history.history["val_f1_score"],
             history.history["auc"], history.history["val_auc"])

        if self.N_TRIALS is None:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history"
        else:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history_trial_{self.trial.number}"
        _pw.save_pickle(os.path.join(self.results_dir, filename), h)
        return self.evaluate(weights=self.bin_path+self.name_model_bin)

    def do_benchmark(self):
        self.logger.save_log(f"model {self.model_size} is being trained")

        acc, bal_acc, f1, auc, loss = self.do_training()

        np.save(os.path.join(self.results_dir, f"model_{self.model_size}_data_{self.DATA_TYPE}_acc.npy"), acc)
        np.save(os.path.join(self.results_dir, f"model_{self.model_size}_data_{self.DATA_TYPE}_bal_acc.npy"), bal_acc)

        self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
        self.logger.save_log(f"Accuracy: {acc}")
        self.logger.save_log(f"Balanced Accuracy: {bal_acc}")
        self.logger.save_log(f"f1: {f1}")
        self.logger.save_log(f"auc: {auc}\n")
    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_weights(self.bin_path+self.name_model_bin)  
        else:
            self.model.load_weights(self.weights_path)

        loss, accuracy_test, f1, auc = self.model.evaluate(self.ds_test, verbose=2)

        X, y = tuple(zip(*self.ds_test))
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0), verbose=2), axis=-1), axis=1)
        y_true = tf.math.argmax(tf.concat(y, axis=0), axis=1)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        conf_matr = sklearn.metrics.confusion_matrix(y_true, y_pred)
        if self.N_TRIALS is None:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr"
        else:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr_trial_{self.trial.number}"
        _pw.save_pickle(os.path.join(self.results_dir, filename), conf_matr)
        self.save_plots()
        return accuracy_test, balanced_accuracy, f1, auc, loss

    def do_test(self):
        self.logger.save_log(f"model {self.model_size} is being trained")

        self.get_data()
        self.get_model()
        acc, bal_acc = self.evaluate()

        self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
        self.logger.save_log(f"Accuracy: {acc}")
        self.logger.save_log(f"Balanced Accuracy: {bal_acc}")

    def do_random_search(self):
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.DATA_TYPE),
                                         directions=["maximize", "maximize"])
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.N_TRIALS, gc_after_trial=True)

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")

        self.logger.save_log("Best trial values:")
        self.logger.save_log(    self.study.best_trials)

        self.logger.save_log("\n  Params: ")
        for s in self.study.best_trials:
            for key, value in s.params.items():
                self.logger.save_log(f"    {key}: {value}")

        self.logger.save_log("Saving study...")
        joblib.dump(self.study,
                    f"{self.results_dir}/{self.DATA_TYPE}_{self.DATASET}_random_search_{self.trial.number}.pickle")

    def objective(self, trial):
        tf.keras.backend.clear_session()
        gc.collect()
        self.trial = trial
        self.get_random_hp()
        acc, bal_acc, f1, auc, loss = self.do_training()
        self.logger.save_log(f'Trial #{trial.number}')
        self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
        self.logger.save_log(f"Accuracy: {acc}")
        self.logger.save_log(f"Balanced Accuracy: {bal_acc}")
        self.logger.save_log(f"f1: {f1}")
        self.logger.save_log(f"auc: {auc}\n")
        return bal_acc, f1

    def get_random_hp(self):
        self.WEIGHT_DECAY = round(self.trial.suggest_float("WD", 1e-4, 1e-2, log=True), 5)
        self.STEP_PERC = round(self.trial.suggest_float("STEP_PERC", 0.5, 0.8, step=0.05), 5)
        self.WARMUP_PERC = round(self.trial.suggest_float("WARMUP_PERC", 0.1, 0.3, step=0.05), 5)

        self.label_smoothing = round(self.trial.suggest_float("label_smoothing", 0, 0.3), 5)
        self.dropout = round(self.trial.suggest_float("dropout", 0.1, 0.8, step=0.05), 5)
        self.mlp_head_size = self.trial.suggest_int("MLP", 32, 256, step=16)
        self.n_heads = self.trial.suggest_int("n_heads", 1, 3, step=1)
        self.d_model = self.d_model*self.n_heads
        self.d_ff = self.trial.suggest_int("d_ff", 2, 4, step=1)*self.d_model
        self.n_layers = self.trial.suggest_int("n_layers", 2, 5, step=1)
        self.max_learning_rate = 5*(10 ** -self.trial.suggest_int("max_learning_rate", 2, 5, step=1))

        self.logger.save_log('WEIGHT_DECAY: {:.2e}'.format(self.WEIGHT_DECAY))
        self.logger.save_log('STEP_PERC: {:.2e}'.format(self.STEP_PERC))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.WARMUP_PERC))

        self.logger.save_log('label_smoothing: {:.2e}'.format(self.label_smoothing))
        self.logger.save_log('dropout: {:.2e}'.format(self.dropout))
        self.logger.save_log('MLP: {}'.format(self.mlp_head_size))
        self.logger.save_log('n_heads: {}'.format(self.n_heads))
        self.logger.save_log('d_model: {}'.format(self.d_model))
        self.logger.save_log('d_ff: {}'.format(self.d_ff))
        self.logger.save_log('n_layers: {}'.format(self.n_layers))
        self.logger.save_log('max_learning_rate: {}'.format(self.max_learning_rate))

    def save_plots(self):
        if self.N_TRIALS is None:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history.pickle"
        else:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history_trial_{self.trial.number}.pickle"
        history = _pw.open_pickle(os.path.join(self.results_dir, filename))
        train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, *_ = history
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"Loss and accuracy for model_{self.model_size}_data_{self.DATA_TYPE}")
        fig.set_size_inches(10.8, 7.2)
        ax1.plot(train_loss, label="Training loss")
        ax1.plot(val_loss, label="Validation loss")
        ax1.set_ylabel("Loss", color='r', fontsize=14)
        ax1.legend()
        ax2.plot(train_acc, label="Training accuracy")
        ax2.plot(val_acc, label="Validation accuracy")
        ax2.plot(train_f1, label="Training f1")
        ax2.plot(val_f1, label="Validation f1")
        ax2.set_ylabel("Accuracy and f1", color='r', fontsize=14)
        ax2.legend()
        plt.savefig(os.path.join(self.results_dir, filename.replace(".pickle", ".jpg")), dpi=100)

        if self.N_TRIALS is None:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr.pickle"
        else:
            filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr_trial_{self.trial.number}.pickle"
        FONT_SIZE = 5
        X_SCALE = 1 if self.N_CLASSES == 33 else 1.4
        cm = _pw.open_pickle(os.path.join(self.results_dir, filename))
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.LABELS)
        disp.plot()
        fig = plt.gcf()
        fig.set_size_inches(10.8, 10.8)
        ax = plt.gca()
        ax.set_title(f"Confusion matrix for model_{self.model_size}_data_{self.DATA_TYPE}_norm")
        ax.set_xticklabels(self.LABELS, rotation=25, ha='right', fontsize=X_SCALE*FONT_SIZE)
        ax.set_yticklabels(self.LABELS, fontsize=1.4 * FONT_SIZE)
        for labels in disp.text_.ravel():
            labels.set_fontsize(1.4 * FONT_SIZE)
        p = os.path.join(self.results_dir, filename.replace(".pickle", ".jpg"))
        plt.savefig(p, dpi=100, bbox_inches='tight')
        plt.close('all')
