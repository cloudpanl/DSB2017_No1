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
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@dc.output(Folder(key="outputDataFolder", required=True))
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="dataColumn", default="data_path"))
@dc.column(String(key="predictionColumn", default="prediction_path"))
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
    predictions_dir = args.args.outputDataFolder
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
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)

    data[args.predictionColumn] = data[args.idColumn] + "_prediction.npy"
    for path, prediction in zip(data[args.predictionColumn], predictions):
        utils.saveAsNpy(os.path.join(predictions_dir, path), prediction)

    return data, predictions_dir


if __name__ == "__main__":
    SPDnnPredict()  # pylint: disable=no-value-for-parameter
