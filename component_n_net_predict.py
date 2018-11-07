# coding: utf-8
from __future__ import print_function

import os
from multiprocessing import freeze_support

import pandas as pd
import torch
from torch.utils.data import DataLoader

import dsb.net_detector as nodmodel
from dsb import preprocessing
from dsb.data_detector import DataBowl3Detector, collate
from dsb.split_combine import SplitComb
from dsb.test_detect import test_detect
from suanpan import asyncio, path, utils
from suanpan.arguments import Bool, Int, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Checkpoint, Folder, HiveTable


@dc.input(
    HiveTable(key="inputData", table="inputDataTable", partition="inputDataPartition")
)
@dc.input(
    Folder(
        key="inputDataFolder",
        required=True,
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.input(Checkpoint(key="inputCheckpoint", required=True, help="Ckpt model file."))
@dc.output(
    HiveTable(
        key="outputBboxData",
        table="outputBboxDataTable",
        partition="outputBboxDataPartition",
    )
)
@dc.output(
    Folder(
        key="outputBboxFolder",
        required=True,
        help="Directory to save bbox npy files to.",
    )
)
@dc.column(String(key="idColumn", default="id", help="ID column of inputImagesData."))
@dc.column(
    String(key="lbbColumn", default="lbb_path", help="Lbb column of inputImagesData.")
)
@dc.column(
    String(key="pbbColumn", default="pbb_path", help="Pbb column of inputImagesData.")
)
@dc.param(Int(key="gpu", help="Number of GPU, 1~N."))
@dc.param(Int(key="margin", default=16, help="patch margin."))
@dc.param(Int(key="sidelen", default=64, help="patch side length."))
def SPNNetPredict(context):
    args = context.args

    data = args.inputData
    dataFolder = args.inputDataFolder
    bboxFolder = args.outputBboxFolder
    path.safeMkdirs(bboxFolder)
    checkpoint = torch.load(args.inputCheckpoint)
    idColumn = args.idColumn
    lbbColumn = args.lbbColumn
    pbbColumn = args.pbbColumn

    gpu = args.gpu
    useGpu = gpu is not None and torch.cuda.is_available()
    sidelen = args.sidelen
    margin = args.margin

    config, nodNet, loss, getPbb = nodmodel.get_model()
    nodNet.load_state_dict(checkpoint["state_dict"])
    nodNet = nodNet.cuda() if useGpu else nodNet.cpu()

    config["datadir"] = dataFolder
    splitComber = SplitComb(
        sidelen,
        config["max_stride"],
        config["stride"],
        margin,
        pad_value=config["pad_value"],
    )

    ids = data[idColumn]
    dataset = DataBowl3Detector(ids, config, phase="test", split_comber=splitComber)
    dataLoader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=asyncio.WORKERS,
        pin_memory=False,
        collate_fn=collate,
    )

    test_detect(dataLoader, nodNet, getPbb, bboxFolder, config, n_gpu=gpu)

    data[lbbColumn] = data[idColumn] + "_lbb.npy"
    data[pbbColumn] = data[idColumn] + "_pbb.npy"

    return data, bboxFolder


if __name__ == "__main__":
    SPNNetPredict()  # pylint: disable=no-value-for-parameter
