# coding=utf-8
from __future__ import print_function

import os

import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam

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
@dc.column(String(key="dataColumn", default="data_path"))
@dc.column(String(key="labelColumn", default="label_path"))
@dc.param(
    Int(key="batchSize", default=128, help="Batch size to use during predicting.")
)
def SPDnnEvalute(context):
    args = context.args

    data = args.inputData
    x_test = np.array(
        [
            utils.loadFromNpy(os.path.join(args.inputDataFolder, p))
            for p in data[args.dataColumn]
        ]
    )
    y_test = np.array(
        [
            utils.loadFromNpy(os.path.join(args.inputDataFolder, p))
            for p in data[args.labelColumn]
        ]
    )

    model_file = args.inputModel
    load_weights = args.inputWeights
    batch_size = args.batchSize

    # Load model file from json.
    with open(model_file, "r") as file:
        config = file.read()

    model = model_from_json(config)
    model.summary()

    # Resume with pre-trained weights if exists.
    model.load_weights(load_weights)
    print("Load weights from: {}".format(load_weights))

    # Evaluate the model with reserved test sets.

    # Compile model.
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"]
    )

    score = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
    score_names = ["loss", "accuracy"]
    evaluateData = pd.DataFrame([score[:2]], columns=score_names)
    return evaluateData


if __name__ == "__main__":
    SPDnnEvalute()  # pylint: disable=no-value-for-parameter
