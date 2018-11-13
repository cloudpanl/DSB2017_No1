# coding=utf-8
from __future__ import division, print_function

import argparse
import os

import numpy as np

import tensorflow as tf
from suanpan import utils
from suanpan.arguments import Int
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import H5Model, Npy
from suanpan.docker.io import storage
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Convolution3D, Dense, Dropout, Flatten, MaxPool3D
from tensorflow.keras.models import Sequential


@dc.input(Npy(key="inputDataNpy", required=True, help="Preprocessed npy file."))
@dc.output(H5Model(key="outputH5Model", required=True, help="Output h5 model file."))
@dc.param(Int(key="epochs", default=5))
def SPTrainKeras(context):
    args = context.args

    tf.set_random_seed(42)
    np.random.seed(42)

    preprocessed = utils.loadFromNpy(args.inputDataNpy)
    IMG_SIZE_PX = preprocessed[0][0].shape[1]
    SLICE_COUNT = preprocessed[0][0].shape[0]

    X = np.array([d[0] for d in preprocessed])
    X = X.reshape(X.shape[0], SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1)
    Y = np.array([d[1] for d in preprocessed])

    # ## Keep an balanced dataset
    # `cancer : non-cancer = 1:1`

    sort_indexes = Y[:, 1].argsort()[::-1]
    Y = Y[sort_indexes]
    X = X[sort_indexes]
    balance_cut_index = int((Y[:, 1].sum() / len(Y) * 2) * len(Y))
    balance_cut_index
    X = X[:balance_cut_index]
    Y = Y[:balance_cut_index]

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    X, Y = unison_shuffled_copies(X, Y)

    # ## Train test split

    train_split = int(len(X) * 0.8)
    x_train = X[:train_split]
    y_train = Y[:train_split]

    x_test = X[train_split:]
    y_test = Y[train_split:]

    K.clear_session()
    tbCallBack = TensorBoard(
        log_dir=storage.getPathInTempStore("Graph"),
        histogram_freq=0,
        write_graph=True,
        write_images=True,
    )

    model = Sequential()
    model.add(
        Convolution3D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(20, 50, 50, 1),
        )
    )
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same"))
    model.add(
        Convolution3D(filters=64, kernel_size=3, padding="same", activation="relu")
    )
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=["accuracy"],
    )

    model.summary()

    batch_size = 1
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[tbCallBack],
    )

    modelPath = storage.getPathInTempStore("model.h5")
    model.save(modelPath)

    print(model.predict(x_test[:20]).argmax(axis=-1))
    print(y_test[:20].argmax(axis=-1))

    # ## Further improvement
    # * Train on full datasets
    # * Preprocessed data to have larger IMG_SIZE_PX and SLICE_COUNT.

    return modelPath


if __name__ == "__main__":
    SPTrainKeras()  # pylint: disable=no-value-for-parameter
