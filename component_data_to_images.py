# coding: utf-8
from __future__ import print_function

import os

import pandas as pd

from suanpan import asyncio, path, utils
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, HiveTable


@dc.input(
    HiveTable(key="inputData", table="inputDataTable", partition="inputDataPartition")
)
@dc.input(Folder(key="inputDataFolder", required=True))
@dc.output(Folder(key="outputImagesFolder", required=True))
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="dataColumn", default="data_path"))
def SPData2Images(context):
    args = context.args

    with asyncio.multiThread() as pool:
        for _, row in args.inputData.iterrows():
            image = utils.loadFromNpy(
                os.path.join(args.inputDataFolder, row[args.dataColumn])
            )
            prefix = os.path.join(args.outputImagesFolder, row[args.idColumn])
            utils.saveAllAsImages(prefix, image, pool=pool)

    return args.outputImagesFolder


if __name__ == "__main__":
    SPData2Images()  # pylint: disable=no-value-for-parameter
