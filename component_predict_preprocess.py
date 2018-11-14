# coding: utf-8
from __future__ import print_function

import os

import pandas as pd

from dsb.preprocessing import full_prep
from suanpan import asyncio
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, HiveTable


def scan_prep_results(folder, id_column, image_column):
    data_suffix = "_clean.npy"
    return pd.DataFrame(
        [
            (file[: -len(data_suffix)], file)
            for file in os.listdir(folder)
            if file.endswith(data_suffix)
        ],
        columns=[id_column, image_column],
    )


@dc.input(
    Folder(
        key="inputDataFolder",
        required=True,
        help="DSB stage1/2 or similar directory path.",
    )
)
@dc.output(
    HiveTable(
        key="outputData", table="outputDataTable", partition="outputDataPartition"
    )
)
@dc.output(
    Folder(
        key="outputDataFolder",
        required=True,
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.output(String(key="idColumn", default="patient"))
@dc.output(String(key="imageColumn", default="image_path"))
def SPPredictPreprocess(context):
    args = context.args

    stage1Path = args.inputDataFolder
    preprocessResultPath = args.outputDataFolder

    full_prep(
        stage1Path, preprocessResultPath, n_worker=asyncio.WORKERS, use_existing=False
    )
    data = scan_prep_results(preprocessResultPath, args.idColumn, args.imageColumn)

    return data, args.outputDataFolder


if __name__ == "__main__":
    SPPredictPreprocess()  # pylint: disable=no-value-for-parameter
