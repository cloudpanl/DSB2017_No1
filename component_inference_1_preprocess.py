# coding: utf-8
from __future__ import print_function

import os
from multiprocessing import freeze_support

from dsb import preprocessing
from suanpan import asyncio, utils
from suanpan.arguments import Bool, Int, String
from suanpan.components import Component as dc
# from suanpan.docker.arguments import Folder, HiveTable


@dc.input(
    String(
        key="inputDataFolder",
        required=True,
        help="DSB stage1/2 or similar directory path.",
    )
)
@dc.output(
    String(
        key="outputData",
        # table="outputTable",
        # partition="outputPartition",
    )
)
@dc.output(
    String(
        key="outputDataFolder",
        required=True,
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.output(
    String(
        key="outputImagesFolder",
        required=True,
        help="Directory to save preprocessed images files to.",
    )
)
@dc.param(
    Int(
        key="workers",
        default=asyncio.WORKERS,
        help="Number of workers for multi-processing.",
    )
)
@dc.param(
    Bool(key="reuse", default=True, help="Reuse/skip existing preprocessed result.")
)
def preprocess(context):
    args = context.args

    data = preprocessing.full_prep(
        args.inputDataFolder,
        args.outputDataFolder,
        n_worker=args.workers,
        use_existing=args.reuse,
    )

    with asyncio.multiThread(args.workers) as pool:
        for _, row in data.iterrows():
            image = utils.loadFromNpy(os.path.join(args.outputDataFolder, row["image"]))
            prefix = os.path.join(args.outputImagesFolder, row["patient"])
            utils.saveAllAsImages(prefix, image, pool=pool)

    data.to_csv(args.outputData)

    return data, args.outputDataFolder, args.outputImagesFolder


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
