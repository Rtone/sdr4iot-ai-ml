import kerastuner as kt
import pandas as pd
import numpy as np
import csv
import os
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers, utils, initializers
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras import layers, models, metrics, optimizers, utils
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import train_test_split
from data_prep import data_preparation_nn
from models_nn import proj_block, id_block
from metrics_nn import test_metrics
from tensorflow.keras.utils import plot_model
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

#tuners for various models. Tunes: length of slices, shift between slices, batch size, number of Conv layers, number of filters, kernel size and activation function for each Conv layer, number of nodes for each Dense layer


#AlexNet
def build_model_alexnet(h):
    try:
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'), 2)))
        model.add(layers.GaussianNoise(0.1))

        for i in range(h.Int('blocks', 1, 4, 1, default=4)):
            model.add(
                layers.Conv1D(filters=h.Int('filters_1', 10, 500, 10, default=128),
                              kernel_size=h.Int('kernal_1', 2, 10, 1, default=7),
                              padding='valid',
                              activation=h.Choice(
                                  'activation_1',
                                  values=['relu', 'tanh', 'sigmoid'],
                                  default='relu')))
            model.add(
                layers.Conv1D(filters=h.Int('filters_1_2',
                                            10,
                                            500,
                                            10,
                                            default=128),
                              kernel_size=h.Int('kernal_1_2', 2, 10, 1, default=5),
                              padding='valid',
                              activation=h.Choice(
                                  'activation_1_2',
                                  values=['relu', 'tanh', 'sigmoid'],
                                  default='relu')))
            model.add(layers.MaxPooling1D((2)))

        model.add(
            layers.Dense(
                units=h.Int('filters_dense_1', 100, 1000, 50, default=256)))
        model.add(
            layers.Dense(
                units=h.Int('filters_dense_2', 100, 1000, 50, default=128)))

        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    except:
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'),2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    return model


#CNN2
def build_model_cnntwo(h):
    try:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    patience=10)

        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'), 2)))
        model.add(layers.GaussianNoise(0.1))
        for i in range(h.Int('blocks', 1, 4, 1, default=4)):
            model.add(
                layers.Conv1D(filters=h.Int('filters_1', 10, 500, 10, default=128),
                              kernel_size=(h.Int('kernal_1', 2, 10, 1, default=7)),
                              padding='valid',
                              activation=h.Choice(
                                  'activation_1',
                                  values=['relu', 'tanh', 'sigmoid'],
                                  default='relu')))
            model.add(layers.MaxPooling1D((2)))
            model.add(
                layers.Dense(
                    units=h.Int('filters_dense_1', 100, 1000, 50, default=128)))

        model.add(
            layers.Dense(
                units=h.Int('filters_dense_2', 100, 1000, 50, default=128)))

        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    except:
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'),2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    return model


#ConvRNN
def build_model_convrnn(h):
    try:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    patience=10)

        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'), 2)))
        model.add(layers.GaussianNoise(0.1))
        for i in range(h.Int('blocks', 1, 4, 1, default=4)):
            model.add(
                layers.Conv1D(filters=h.Int('filters_1', 10, 500, 10, default=128),
                              kernel_size=(h.Int('kernal_1', 2, 10, 1, default=7)),
                              padding='valid',
                              activation=h.Choice(
                                  'activation_1',
                                  values=['relu', 'tanh', 'sigmoid'],
                                  default='relu')))
            model.add(layers.MaxPooling1D((2)))

        model.add(
            layers.SimpleRNN(units=h.Int('filters_rnn', 10, 500, 10, default=512),
                             activation=h.Choice(
                                 'activation_rnn',
                                 values=['relu', 'tanh', 'sigmoid'],
                                 default='relu')))

        model.add(
            layers.Dense(
                units=h.Int('filters_dense_1', 100, 1000, 50, default=256)))

        model.add(
            layers.Dense(
                units=h.Int('filters_dense_2', 100, 1000, 50, default=128)))

        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    except:
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'),2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    return model


#ResNet
def build_model_resnet(h):
    try:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    patience=10)

        inputs = layers.Input(shape=(h.get('len_slice'), 2))

        t = layers.BatchNormalization()(inputs)
        t = layers.GaussianNoise(0.1)(t)
        t = layers.Conv1D(kernel_size=h.Int('kernel_1', 2, 20, 1, default=7),
                          filters=h.Int('filters_1', 10, 200, 10, default=64),
                          activation=h.Choice('activation_1',
                                              values=['relu', 'tanh', 'sigmoid'],
                                              default='relu'))(t)
        t = layers.MaxPool1D(2)(t)
        f_b_1 = h.Int('filters_block_1', 10, 200, 10, default=64)
        t = proj_block(t, f_b_1)
        for i in range(h.Int('nb_id_1', 1, 5, 1, default=2)):
            t = id_block(t, f_b_1)

        f_b_2 = h.Int('filters_block_2', 10, 200, 10, default=128)
        t = proj_block(t, f_b_2)
        for j in range(h.Int('nb_id_2', 1, 5, 1, default=3)):
            t = id_block(t, f_b_2)
        t = layers.AveragePooling1D(2)(t)
        t = layers.Flatten()(t)
        outputs = layers.Dense(3, activation='softmax')(t)

        model = tf.keras.models.Model(inputs, outputs)

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    except:
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'),2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    return model


#CNNConv2D
def build_model_cnnconvtwod(h):
    try:
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0.0001,
                                                    patience=10)

        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=input_shape))
        model.add(layers.GaussianNoise(0.1))

        model.add(
            layers.Conv2D(filters=h.Int('filters_1', 10, 200, 10, default=40),
                          kernel_size=(h.Int('kernel_1', 2, 20, 1, default=7), 1),
                          padding='valid',
                          activation=h.Choice('activation_1',
                                              values=['relu', 'tanh', 'sigmoid'],
                                              default='relu')))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(
            layers.Conv2D(filters=h.Int('filters_2', 10, 200, 10, default=40),
                          kernel_size=(h.Int('kernel_2', 2, 20, 1, default=5), 1),
                          padding='valid',
                          activation=h.Choice('activation_2',
                                              values=['relu', 'tanh', 'sigmoid'],
                                              default='relu')))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(
            layers.Conv2D(filters=h.Int('filters_3', 10, 200, 10, default=40),
                          kernel_size=(h.Int('kernel_3', 2, 20, 1, default=7), 2),
                          padding='valid',
                          activation=h.Choice('activation_3',
                                              values=['relu', 'tanh', 'sigmoid'],
                                              default='relu')))
        model.add(layers.MaxPooling2D((2, 1)))

        model.add(
            layers.Dense(units=h.Int('filters_d_1', 100, 1000, 10, default=1024),
                         activation=h.Choice('activation_4',
                                             values=['relu', 'tanh', 'sigmoid'],
                                             default='relu')))
        model.add(
            layers.Dense(units=h.Int('filters_d_2', 100, 1000, 10, default=256),
                         activation=h.Choice('activation_5',
                                             values=['relu', 'tanh', 'sigmoid'],
                                             default='relu')))

        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    except:
        print("VALUE ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        model = models.Sequential()
        model.add(layers.BatchNormalization(input_shape=(h.get('len_slice'),2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

    return model


#run trial each trial with a different length of slice, shift between slices and batch_size (and model parameters)
class MyTuner(kt.Tuner):
    def run_trial(self, trial, idata):
        hp = trial.hyperparameters
        X_train, X_test, y_train, y_test, input_shape, nb_slice = data_preparation_nn(
            idata,hp.get('len_slice'), hp.get('shift'))
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    min_delta=0.0001,
                                                    patience=5)

        batch_size = hp.get('batch_size')
        model = self.hypermodel.build(trial.hyperparameters)
        
        history = model.fit(X_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=300,
                            verbose=1,
                            validation_split=0.2,
                            callbacks=[callback])
        acc = history.history.get('val_accuracy')[-1]
        self.oracle.update_trial(trial.trial_id, {'val_acc': acc})
        self.save_model(trial.trial_id, model)


#launch tuner for given type of model, saves the best model according to val_accuracy
def launch_tuner(max_trial, idata, experiment_name, nn_type):
    log_file = experiment_name + '/log_tune_model.log'
    log_file = open(log_file, "a")
    sys.stdout = log_file
    
    tf.debugging.set_log_device_placement(False)
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        hp = kt.HyperParameters()
        hp.Int('len_slice', 50, 1000, 100, default=200)
        hp.Int('shift', 10, 500, 10, default=50)

        if nn_type == 'AlexNet':
            hp.Int('batch_size', 10, 70, 10, default=30)
            build_model = build_model_alexnet
        elif nn_type == 'CNN2':
            hp.Int('batch_size', 10, 70, 10, default=30)
            build_model = build_model_cnntwo
        elif nn_type == 'ConvRNN':
            hp.Int('batch_size', 10, 70, 10, default=30)
            build_model = build_model_convrnn
        elif nn_type == 'ResNet':
            hp.Int('batch_size', 2, 20, 2, default=10)
            build_model = build_model_resnet
        elif nn_type == 'CNNConv2D':
            hp.Int('batch_size', 10, 70, 10, default=30)
            build_model = build_model_cnnconvtwod
        else:
            print('Unknown model')

    tuner_rs = MyTuner(oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective('val_acc', 'max'),
        max_trials=max_trial,
        hyperparameters=hp),
                       hypermodel=build_model,
                       overwrite=True)

    tuner_rs.search_space_summary()

    tuner_rs.search(idata=idata)

    print(tuner_rs.results_summary())

    best_model = tuner_rs.get_best_models(num_models=1)[0]
    model_file = experiment_name + '/tuned_model'
    best_model.save(model_file)
    file_model_graph = experiment_name + '/' + nn_type + '_graph.png'
    plot_model(best_model, show_shapes=True, to_file=file_model_graph)

    results=tuner_rs.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
    len_slice=results['len_slice']
    shift=results['shift']

    X_train, X_test, y_train, y_test, input_shape, nb_slice = data_preparation_nn(
            idata,len_slice, shift)
    test_metrics(best_model, y_test, X_test, experiment_name)
    log_file.close()
