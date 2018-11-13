# coding=utf-8
from __future__ import print_function

import os

import numpy as np
from keras.models import model_from_json

from suanpan import utils
from suanpan.arguments import Int, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, H5Model, HiveTable, JsonModel, Npy


@dc.input(
    JsonModel(
        key="inputModel", required=True, help="Direcotry path to load the model file."
    )
)
@dc.input(
    H5Model(key="inputWeights", required=True, help="Trained model weights file.")
)
@dc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@dc.input(Folder(key="inputDataFolder", required=True))
@dc.output(
    Npy(
        key="prediction",
        required=True,
        help="Input Numpy .npy file containing test samples.",
    )
)
@dc.column(String(key="dataColumn", default="data_path"))
@dc.param(
    Int(key="batchSize", default=128, help="Batch size to use during predicting.")
)
def SPDnnPredict(context):
    args = context.args

    data = args.inputData
    x_test = np.array(
        [
            utils.loadFromNpy(os.path.join(args.inputDataFolder, p))
            for p in data[args.dataColumn]
        ]
    )

    model_file = args.inputModel
    load_weights = args.inputWeights
    save_predicts = args.prediction
    batch_size = args.batchSize

    # Load model file from json.
    with open(model_file, "r") as file:
        config = file.read()

    model = model_from_json(config)
    model.summary()

    # Resume with pre-trained weights if exists.
    model.load_weights(load_weights)
    print("Load weights from: {}".format(load_weights))

    # Predict with the model.
    predicts = model.predict(x_test, batch_size=batch_size, verbose=1)

    np.save(save_predicts, predicts)
    print("Predicts saved to: {}".format(save_predicts))

    return save_predicts


if __name__ == "__main__":
    SPDnnPredict()  # pylint: disable=no-value-for-parameter
