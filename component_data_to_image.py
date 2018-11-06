# coding: utf-8
from __future__ import print_function

import os
import pandas as pd

# from suanpan.docker.arguments import File, Folder, HiveTable
from suanpan import asyncio, convert, path, utils
from suanpan.arguments import Bool, Int, String
from suanpan.components import Component as dc


@dc.input(
    String(
        key="inputData",
        # table="inputTable",
        # partition="inputPartition",
    )
)
@dc.input(
    String(
        key="inputDataFolder",
        required=True,
    )
)
@dc.output(
    String(
        key="outputImageFolder",
        required=True,
    )
)
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="dataColumn", default="data"))
def preprocess(context):
    args = context.args
    data = pd.read_csv(args.inputData)

    with asyncio.multiThread(args.workers) as pool:
        for _, row in data.iterrows():
            image = utils.loadFromNpy(os.path.join(args.outputDataFolder, row[args.dataColumn]))
            prefix = os.path.join(args.outputImagesFolder, row[args.idColumn])
            utils.saveAllAsImages(prefix, image, pool=pool)

    return args.outputImageFolder


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
