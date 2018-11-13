# coding=utf-8
from __future__ import print_function

import os

from keras.layers import Dense, Dropout, Input
from keras.models import Model

from suanpan import path
from suanpan.arguments import Float, Int, ListOfInt
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import JsonModel


@dc.output(
    JsonModel(
        key="outputModel", required=True, help="Direcotry path to save the model file."
    )
)
@dc.param(
    Int(key="numClasses", required=True, help="Number of classes for the classifier.")
)
@dc.param(
    Int(
        key="inputShape",
        required=True,
        help="Number of input features for the classifier.",
    )
)
@dc.param(
    ListOfInt(
        key="hiddenUnits",
        required=True,
        help="Iterable of number hidden units per layer. All layers are fully connected. Ex. 64,32 means first layer has 64 nodes and second one has 32.",
    )
)
@dc.param(
    Float(
        key="dropout", default=0.2, help="Number of input features for the classifier."
    )
)
def SPDnn(context):
    args = context.args

    model_file = args.outputModel
    num_classes = args.numClasses
    input_shape = (args.inputShape,)
    hidden_units = args.hiddenUnits
    dropout = args.dropout

    inputs = Input(shape=input_shape)
    x = None
    for units in hidden_units:
        if x is None:
            x = Dense(units, activation="relu")(inputs)
        else:
            x = Dense(units, activation="relu")(x)
        if dropout != 0 and dropout < 1:
            x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()

    # Save model file.
    with open(model_file, "w") as file:
        file.write(model.to_json())

    return model_file


if __name__ == "__main__":
    SPDnn()  # pylint: disable=no-value-for-parameter
