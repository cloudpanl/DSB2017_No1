# coding=utf-8
from __future__ import print_function

import os

import keras.optimizers as optimizers
import numpy as np
from keras.callbacks import TensorBoard
from keras.models import model_from_json

from suanpan import utils
from suanpan.arguments import Bool, Float, Int, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, H5Model, HiveTable, JsonModel, Npy


def getData(folder, paths):
    for p in paths:

        utils.loadFromNpy(os.path.join(folder, p))


@dc.input(
    JsonModel(
        key="inputModel", required=True, help="Direcotry path to load the model file."
    )
)
@dc.input(
    H5Model(
        key="resumeWeights",
        required=True,
        help="Optional path of *.h5 weights file to resume training.",
    )
)
@dc.input(HiveTable(key="inputData", table="inputTable", partition="inputPartition"))
@dc.input(Folder(key="inputDataFolder", required=True))
@dc.output(
    H5Model(
        key="saveWeights",
        required=True,
        help="File path to save the trained model weights.",
    )
)
@dc.output(
    Folder(
        key="tensorBoard",
        required=True,
        help="Directory to save tensor board files during training.",
    )
)
@dc.column(String(key="dataColumn", default="data_path"))
@dc.column(String(key="labelColumn", default="label_path"))
@dc.param(
    String(
        key="optimizer",
        default="Adam",
        help="Training optimizer, one of `Adagrad`, `Adam`, `RMSProp`, `SGD`",
    )
)
@dc.param(Float(key="lr", default=0.001, help="Learning rate for training optimizer."))
@dc.param(Int(key="batchSize", default=128, help="Batch size to use during training."))
@dc.param(Int(key="epochs", default=5, help="Number of total epochs to run."))
@dc.param(Bool(key="shuffle", default=True, help="Shuffle training datasets."))
@dc.param(
    Int(
        key="seed",
        default=22,
        help="Specify a numeric seed to use for random number generation. Leave blank to use the default seed.",
    )
)
def SPDnnTrain(context):
    args = context.args

    data = args.inputData
    x_train = np.array(
        [
            utils.loadFromNpy(os.path.join(args.inputDataFolder, p))
            for p in data[args.dataColumn]
        ]
    )
    y_train = np.array(
        [
            utils.loadFromNpy(os.path.join(args.inputDataFolder, p))
            for p in data[args.labelColumn]
        ]
    )

    model_file = args.inputModel
    resume_weights = args.resumeWeights
    save_weights = args.saveWeights

    optimizer = args.optimizer
    lr = args.lr
    batch_size = args.batchSize
    epochs = args.epochs
    shuffle = args.shuffle
    tensor_board = args.tensorBoard
    seed = args.seed

    # Seed to allow reproducible results.
    np.random.seed(seed)

    # Load model file from json.
    with open(model_file, "r") as file:
        config = file.read()

    model = model_from_json(config)
    model.summary()

    # Match optimizer by name.
    # `Adagrad`, `Adam`, `RMSProp`, `SGD`
    Optimizer = getattr(optimizers, optimizer, None)
    if not Optimizer:
        raise Exception("Unknown optimizer: {}".format(optimizer))
    optimizer = Optimizer(lr=lr)

    # Compile model.
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    # Resume with pre-trained weights if exists.
    if resume_weights:
        model.load_weights(resume_weights)
        print("Resume weights from: {}".format(resume_weights))

    # Training/fitting the model.
    tbCallBack = TensorBoard(
        log_dir=tensor_board, histogram_freq=0, write_graph=True, write_images=True
    )
    model.fit(
        x_train,
        y_train,
        shuffle=shuffle,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[tbCallBack],
    )

    # Save weights file.
    model.save_weights(save_weights)

    return model_file, tensor_board


if __name__ == "__main__":
    SPDnnTrain()  # pylint: disable=no-value-for-parameter
