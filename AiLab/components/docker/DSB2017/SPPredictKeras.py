# coding=utf-8
from __future__ import division, print_function

import os

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import load_model
from suanpan import utils
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import H5Model, HiveTable, Npy
from suanpan.docker.io import storage
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution3D, Dense, Dropout, Flatten, MaxPool3D
from tensorflow.keras.models import Sequential


@dc.input(Npy(key="inputDataNpy", required=True))
@dc.input(H5Model(key="inputH5Model", required=True))
@dc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
def SPPredictKeras(context):
    args = context.args

    tf.set_random_seed(42)
    np.random.seed(42)

    preprocessed = utils.loadFromNpy(args.inputDataNpy)
    IMG_SIZE_PX = preprocessed[0][0].shape[1]
    SLICE_COUNT = preprocessed[0][0].shape[0]

    X = np.array([d[0] for d in preprocessed])
    X = X.reshape(X.shape[0], SLICE_COUNT, IMG_SIZE_PX, IMG_SIZE_PX, 1)
    Y = np.array([d[1] for d in preprocessed])

    model = load_model(args.inputH5Model)

    model.summary()

    batch_size = 1

    label = Y.argmax(axis=-1).astype(np.float)
    prediction = (
        model.predict(X, batch_size=batch_size).argmax(axis=-1).astype(np.float)
    )
    # List of predictions for input patients.
    output = pd.DataFrame({"label": label, "prediction": prediction})

    return output


if __name__ == "__main__":
    SPPredictKeras()  # pylint: disable=no-value-for-parameter
