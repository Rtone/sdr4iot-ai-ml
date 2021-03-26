import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Prepares input for the various NN used


# creates overlapping slices of the packet, of a given length and overlapping
def input_slices_pkt(idata, len_slice, shift):
    # creating input for CNN: IQ values for a whole pkt     output: robot_node
    X = list()
    Y = list()
    pkt_len = 1520
    i = 0
    while i < len(idata.index):
        data = idata.iloc[i:i + pkt_len]
        if len(data['Time'].unique()) == 1:
            data = np.array(data[['real', 'im']], dtype='float64')
            x = data.reshape((pkt_len), 2)
            # creates slices of the packet and gather them in a single batch
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.window(len_slice, shift, 1, True)
            count = 0
            for window in x:
                X.append(list(window.as_numpy_iterator()))
                count = count + 1

                # in each scene a different emitter is used ~ robot_node in that case
                Y.append(int(idata.iloc[i]['Scene']))
        else:
            print('Missing!!')
        i = i + pkt_len

    X = np.array(X)
    Y = np.array(Y, dtype=int)

    input_shape = X[0].shape
    batch_size = count

    print("Il y a " + str(len(Y)) + " échantillons")

    return (X, Y, input_shape, batch_size)


# for classification: balance the classes so that they all have the same amount of samples
def balance_classes(X, Y):
    unique_elements, counts_elements = np.unique(Y, return_counts=True)

    # balance classes
    min_samples = min(counts_elements)

    thirty_five_index = np.where(Y == 35)
    thirty_five_index = thirty_five_index[0][:min_samples]
    X_thirty_five = X[thirty_five_index]
    Y_thirty_five = Y[thirty_five_index]

    thirty_six_index = np.where(Y == 36)
    thirty_six_index = thirty_six_index[0][:min_samples]
    X_thirty_six = X[thirty_six_index]
    Y_thirty_six = Y[thirty_six_index]

    thirty_seven_index = np.where(Y == 37)
    thirty_seven_index = thirty_seven_index[0][:min_samples]
    X_thirty_seven = X[thirty_seven_index]
    Y_thirty_seven = Y[thirty_seven_index]

    X = np.concatenate((X_thirty_five, X_thirty_six, X_thirty_seven))
    Y = np.concatenate((Y_thirty_five, Y_thirty_six, Y_thirty_seven))

    print(np.unique(Y, return_counts=True))
    return (X, Y)


# for classification: each scene (35/36/37) corresponds to a given class => 35:0, 36:1 37:2


def name_classes(Y):
    Y_change = list()

    for i in Y:
        if i == 35:
            Y_change.append(0)
        elif i == 36:
            Y_change.append(1)
        elif i == 37:
            Y_change.append(2)

    Y_change = np.array(Y_change)

    return Y_change


# prepares the input data depending on the number of servers chosen using the slices approach, selects only packet of a given length, normalizes the data, creates X_train, y_train, X_test and y_test sets


def data_preparation_nn(idata, len_slice, shift):

    scaler = MinMaxScaler(feature_range=(0, 1))
    idata[['real', 'im']] = scaler.fit_transform(idata[['real', 'im']].values)

    X, Y, input_shape, batch_size = input_slices_pkt(idata, len_slice, shift)

    X, Y = balance_classes(X, Y)
    Y_change = name_classes(Y)

    unique_elements, counts_elements = np.unique(Y_change, return_counts=True)
    nb_class = max(unique_elements) + 1

    Y_cat = utils.to_categorical(Y_change, num_classes=nb_class)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y_cat,
                                                        test_size=0.2,
                                                        random_state=54)
    return (X_train, X_test, y_train, y_test, input_shape, batch_size)
