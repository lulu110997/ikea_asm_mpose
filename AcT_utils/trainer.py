# GENERAL LIBRARIES 
import os
import math
import sys
import time

import yaml
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import joblib
# MACHINE LEARNING LIBRARIES
import sklearn
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

# TRAINER CLASS
class Trainer:
    def __init__(self, config, logger, model_sz, root_results_dir, split=1):

        self.trial = None
        self.norm_input = None
        self.logger = logger
        self.split = split

        self.model_size = model_sz #self.config['MODEL_SIZE']
        self.n_heads = config[self.model_size]['N_HEADS']
        self.n_layers = config[self.model_size]['N_LAYERS']
        self.embed_dim = config[self.model_size]['EMBED_DIM']  # Size of embedded input. dv = 64, made constant according to paper
        self.dropout = config[self.model_size]['DROPOUT']
        self.mlp_head_size = config[self.model_size]['MLP']  # Output size of the ff layer prior the classification layer
        self.activation = tf.nn.gelu
        self.d_model = 64 * self.n_heads
        self.d_ff = 4 * self.d_model  # Output size of the first non-linear layer in the transformer encoder
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
        self.N_FOLD = config["FOLDS"]
        self.N_SPLITS = config["SPLITS"]
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
            x = tf.keras.layers.Dense(self.d_model)(inputs)  # Projection layer
        else:
            x = self.norm_input(inputs); print("normalise")
            x = tf.keras.layers.Dense(self.d_model)(x)  # Projection layer

        x = PatchClassEmbedding(self.d_model, self.N_FRAMES, pos_emb=None)(x)  # Positional embedding layer
        x = transformer(x)  # Transformer
        x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)  # Obtain cls
        x = tf.keras.layers.Dense(self.mlp_head_size)(x)  # Dense layer
        outputs = tf.keras.layers.Dense(self.N_CLASSES)(x)  # Classification layer
        return tf.keras.models.Model(inputs, outputs)

    
    def get_model(self):
        transformer = TransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout, self.activation, self.n_layers)
        self.model = self.build_act(transformer)
        # weights = "AcT_small_1_0.h5"
        # weights = weights.replace("small", self.model_size)
        # self.model.load_weights(weights)
        # self.model = tf.keras.Model(inputs=self.model.inputs,
        #                             outputs=tf.keras.layers.Dense(self.N_CLASSES)(self.model.layers[-2].output))
        
        self.train_steps = np.ceil(float(self.train_len)/self.BATCH_SIZE) if self.train_len > 7000 else 24

        lr = CustomSchedule(self.d_model,
                                 warmup_steps=self.train_steps * self.N_EPOCHS * self.WARMUP_PERC,
                                 decay_step=self.train_steps * self.N_EPOCHS * self.STEP_PERC)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        optim = tf.keras.optimizers.AdamW(lr, weight_decay=self.WEIGHT_DECAY)

        self.model.compile(optimizer=optim,
                           loss=loss,
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                                    tf.keras.metrics.F1Score(average="macro"),
                                    tf.keras.metrics.AUC(multi_label=True, from_logits=True, num_labels=self.N_CLASSES)])
        self.name_model_bin = f"{self.model_size}_{self.DATA_TYPE}.h5"
        self.checkpointer = tf.keras.callbacks.ModelCheckpoint(self.bin_path + self.name_model_bin, monitor="val_loss",
                                                               mode="min", save_best_only=True, save_weights_only=True)
        return

    def resample(self, x, y):
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
        # self.norm_input = tf.keras.layers.Normalization()
        # self.norm_input.adapt(X_train)

        self.ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)); self.train_len = len(y_train)
        # self.ds_train = self.resample(X_train, y_train); self.train_len = 6500

        # self.class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight="balanced",
                                                                  # classes=np.unique(y_train), y=y_train)
        # self.class_weights = {key: val for key, val in enumerate(weights)}

        self.ds_train = self.ds_train.map(lambda x, y: one_hot(x, y, self.N_CLASSES),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.map(random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #self.ds_train = self.ds_train.map(random_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds_train = self.ds_train.shuffle(3000, reshuffle_each_iteration=True)
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
        history = self.model.fit(self.ds_train,
                                 epochs=self.N_EPOCHS, initial_epoch=0,
                                 validation_data=self.ds_test,
                                 callbacks=[self.checkpointer])
                                 #,class_weight=self.class_weights)
        h = (history.history['loss'], history.history['val_loss'],
             history.history['accuracy'], history.history['val_accuracy'],
             history.history["f1_score"], history.history["val_f1_score"],
             history.history["auc"], history.history["val_auc"])

        filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history"
        _pw.save_pickle(os.path.join(self.results_dir, filename), h)
        return self.evaluate(weights=self.bin_path+self.name_model_bin)

    def do_benchmark(self):
        for split in range(1, self.N_SPLITS + 1):
            self.logger.save_log(f"model {self.model_size} is being trained")

            self.split = split

            acc, bal_acc, f1, auc = self.do_training()

            np.save(os.path.join(self.results_dir, f"model_{self.model_size}_data_{self.DATA_TYPE}_acc.npy"), acc)
            np.save(os.path.join(self.results_dir, f"model_{self.model_size}_data_{self.DATA_TYPE}_bal_acc.npy"), bal_acc)

            self.logger.save_log(f"Model {self.model_size} metrics with {self.DATA_TYPE}")
            self.logger.save_log(f"Accuracy: {acc}")
            self.logger.save_log(f"Balanced Accuracy: {bal_acc}")
            self.logger.save_log(f"f1: {f1}")
            self.logger.save_log(f"auc: {auc}")
    def evaluate(self, weights=None):
        if weights is not None:
            self.model.load_weights(self.bin_path+self.name_model_bin)  
        else:
            self.model.load_weights(self.weights_path)

        loss, accuracy_test, f1, auc = self.model.evaluate(self.ds_test)

        X, y = tuple(zip(*self.ds_test))
        y_pred = np.argmax(tf.nn.softmax(self.model.predict(tf.concat(X, axis=0)), axis=-1), axis=1)
        y_true = tf.math.argmax(tf.concat(y, axis=0), axis=1)
        balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        conf_matr = sklearn.metrics.confusion_matrix(y_true, y_pred)
        filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr"
        _pw.save_pickle(os.path.join(self.results_dir, filename), conf_matr)
        self.save_plots()
        return accuracy_test, balanced_accuracy, f1, auc

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

    def do_random_search(self):
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(study_name='{}_random_search'.format(self.DATA_TYPE),
                                         direction="maximize", pruner=pruner)
        self.study.optimize(lambda trial: self.objective(trial),
                            n_trials=self.N_TRIALS)

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
                    f"{self.results_dir}/{self.DATA_TYPE}_{self.DATASET}_random_search_{str(self.study.best_trial.value)}.pkl")

    def objective(self, trial):
        self.trial = trial
        self.get_random_hp()
        _, bal_acc = self.do_training()
        return bal_acc

    def get_random_hp(self):
        self.N_EPOCHS = int(self.trial.suggest_discrete_uniform("EPOCHS", 50, 100, 10))
        self.WARMUP_PERC = self.trial.suggest_discrete_uniform("WARMUP_PERC", 0.1, 0.4, 0.1)
        self.WEIGHT_DECAY = self.trial.suggest_discrete_uniform("WD", 1e-5, 1e-3, 1e-5)
        self.STEP_PERC = self.trial.suggest_discrete_uniform("STEP_PERC", 0.1, 0.4, 0.1)

        self.logger.save_log('\nEPOCHS: {}'.format(self.N_EPOCHS))
        self.logger.save_log('WARMUP_PERC: {:.2e}'.format(self.WARMUP_PERC))
        self.logger.save_log('WEIGHT_DECAY: {:.2e}'.format(self.WEIGHT_DECAY))
        self.logger.save_log('LR_MULT: {:.2e}'.format(self.STEP_PERC))

    def save_plots(self):
        filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_history.pickle"
        history = _pw.open_pickle(os.path.join(self.results_dir, filename))
        train_loss, val_loss, train_acc, val_acc, *_ = history
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"Loss and accuracy for model_{self.model_size}_data_{self.DATA_TYPE}")
        fig.set_size_inches(10.8, 7.2)
        ax1.plot(train_loss, label="Training loss")
        ax1.plot(val_loss, label="Validation loss")
        ax1.set_ylabel("Loss", color='r', fontsize=14)
        ax1.legend()
        ax2.plot(train_acc, label="Training accuracy")
        ax2.plot(val_acc, label="Validation accuracy")
        ax2.set_ylabel("Accuracy", color='r', fontsize=14)
        ax2.legend()
        plt.savefig(os.path.join(self.results_dir, filename.replace(".pickle", ".jpg")), dpi=100)

        filename = f"model_{self.model_size}_data_{self.DATA_TYPE}_conf_matr.pickle"
        FONT_SIZE = 5
        X_SCALE = 1 if self.N_CLASSES == 33 else 1.4
        cm = _pw.open_pickle(os.path.join(self.results_dir, filename))
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.LABELS)
        disp.plot()
        fig = plt.gcf()
        fig.set_size_inches(10.8, 10.8)
        ax = plt.gca()
        ax.set_title(f"Confusion matrix for model_{self.model_size}_data_{self.DATA_TYPE}")
        ax.set_xticklabels(self.LABELS, rotation=25, ha='right', fontsize=X_SCALE*FONT_SIZE)
        ax.set_yticklabels(self.LABELS, fontsize=1.4 * FONT_SIZE)
        for labels in disp.text_.ravel():
            labels.set_fontsize(1.4 * FONT_SIZE)
        p = os.path.join(self.results_dir, filename.replace(".pickle", ".jpg"))
        plt.savefig(p, dpi=100, bbox_inches='tight')
        plt.close('all')


