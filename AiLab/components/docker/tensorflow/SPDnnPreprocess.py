# coding=utf-8
from __future__ import print_function

import itertools
import os

import keras
import pandas as pd
from keras.datasets import mnist

from suanpan import utils
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, HiveTable


@dc.output(
    HiveTable(key="outputData", table="outputTable", partition="outputPartition")
)
@dc.output(Folder(key="outputDataFolder", required=True))
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="dataColumn", default="data_path"))
@dc.column(String(key="labelColumn", default="label_path"))
def SPDnnPreprocess(context):
    args = context.args

    datasets_path = args.outputDataFolder
    id_column = args.idColumn
    data_column = args.dataColumn
    label_column = args.labelColumn

    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Turn values to range 0~1.
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    def _save():
        for index, data in enumerate(
            zip(itertools.chain(x_train, x_test), itertools.chain(y_train, y_test))
        ):
            x, y = data
            data_file = "{}_data.npy".format(index)
            label_file = "{}_label.npy".format(index)
            data_path = os.path.join(datasets_path, data_file)
            label_path = os.path.join(datasets_path, label_file)
            utils.saveAsNpy(data_path, x)
            utils.saveAsNpy(label_path, y)
            yield index, data_file, label_file

    data = pd.DataFrame(
        [item for item in _save()], columns=[id_column, data_column, label_column]
    )
    return data, datasets_path


if __name__ == "__main__":
    SPDnnPreprocess()  # pylint: disable=no-value-for-parameter
