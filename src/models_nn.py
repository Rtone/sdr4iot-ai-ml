import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers, utils, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

#Various tuned models which can be trained on a given dataset


#AlexNet/CNN1
def AlexNet(input_shape):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=10)

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    model.add(layers.GaussianNoise(0.1, input_shape=input_shape))

    for i in range(3):
        model.add(
            layers.Conv1D(filters=140,
                          kernel_size=(6),
                          padding='valid',
                          activation='tanh'))
        model.add(
            layers.Conv1D(filters=300,
                          kernel_size=(6),
                          padding='valid',
                          activation='tanh'))
        model.add(layers.MaxPooling1D((2)))

    model.add(layers.Dense(units=650))
    model.add(layers.Dense(units=400))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model, callback


#CNN2
def CNN2(input_shape):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=100)

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    model.add(layers.GaussianNoise(0.1))
    for i in range(3):
        model.add(
            layers.Conv1D(filters=60,
                          kernel_size=(2),
                          padding='valid',
                          activation='relu'))
        model.add(layers.MaxPooling1D((2)))
        model.add(layers.Dense(units=300))

    model.add(layers.Dense(units=750))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model, callback


#ConvRNN : includes a RNN layer
def ConvRNN(input_shape):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=100)

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    model.add(layers.GaussianNoise(0.1))
    for i in range(2):
        model.add(
            layers.Conv1D(filters=430,
                          kernel_size=(6),
                          padding='valid',
                          activation='tanh'))
        model.add(layers.MaxPooling1D((2)))

    model.add(layers.SimpleRNN(units=400, activation="relu"))

    model.add(layers.Dense(units=250))
    model.add(layers.Dense(units=350))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model, callback


#ResNet
def proj_block(x: tf.Tensor, filters):
    y_first = layers.Conv1D(kernel_size=1,
                            filters=filters,
                            padding="same",
                            activation='relu')(x)
    y_first = layers.Conv1D(kernel_size=3, filters=filters,
                            activation='relu')(y_first)
    y_first = layers.Conv1D(kernel_size=1,
                            filters=4 * filters,
                            activation='relu')(y_first)

    y_second = layers.Conv1D(kernel_size=1,
                             filters=4 * filters,
                             padding="same",
                             activation='relu')(x)

    y = layers.concatenate([y_first, y_second], axis=1)
    y = layers.Activation(activation='relu')(y)

    return y


def id_block(x: tf.Tensor, filters):
    y_first = layers.Conv1D(kernel_size=1,
                            filters=filters,
                            padding="same",
                            activation='relu')(x)
    y_first = layers.Conv1D(kernel_size=3, filters=filters,
                            activation='relu')(y_first)
    y_first = layers.Conv1D(kernel_size=1,
                            filters=4 * filters,
                            activation='relu')(y_first)

    y = layers.concatenate([y_first, x], axis=1)
    y = layers.Activation(activation='relu')(y)

    return y


def ResNet(input_shape):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=100)

    inputs = layers.Input(shape=input_shape)

    t = layers.BatchNormalization()(inputs)
    t = layers.GaussianNoise(0.1)(t)
    t = layers.Conv1D(kernel_size=7, filters=64, activation='relu')(t)
    t = layers.MaxPool1D(2)(t)
    t = proj_block(t, 64)
    for i in range(2):
        t = id_block(t, 64)

    t = proj_block(t, 128)
    for j in range(3):
        t = id_block(t, 128)

    t = layers.AveragePooling1D(2)(t)
    t = layers.Flatten()(t)
    outputs = layers.Dense(3, activation='softmax')(t)

    model = tf.keras.models.Model(inputs, outputs)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model, callback


#CNNConv2D : includes Conv2D layers
def CNNConv2D(input_shape):
    X_train = X_train.reshape(
        (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape(
        (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=0.0001,
                                                patience=100)

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=input_shape))
    model.add(layers.GaussianNoise(0.1))

    model.add(
        layers.Conv2D(filters=110,
                      kernel_size=(5, 1),
                      padding='valid',
                      activation='tanh'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(
        layers.Conv2D(filters=150,
                      kernel_size=(6, 1),
                      padding='valid',
                      activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(
        layers.Conv2D(filters=40,
                      kernel_size=(9, 2),
                      padding='valid',
                      activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))

    model.add(layers.Dense(units=280, activation='relu'))
    model.add(layers.Dense(units=480, activation='tanh'))

    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model, callback


# build the selected model
def build_model(nn_type, input_shape, experiment_name):
    if nn_type == 'AlexNet':
        model, callback = AlexNet(input_shape)
    elif nn_type == 'CNN2':
        model, callback = CNN2(input_shape)
    elif nn_type == 'ConvRNN':
        model, callback = ConvRNN(input_shape)
    elif nn_type == 'ResNet':
        model, callback = ResNet(input_shape)
    elif nn_type == 'CNNConv2D':
        model, callback = CNNConv2D(input_shape)
    else:
        print('Unknown model')
    file_model = experiment_name + '/' + nn_type + '_graph.png'
    plot_model(model, to_file=file_model)
    return model, callback
