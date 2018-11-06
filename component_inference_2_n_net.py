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
from suanpan.components import Component as dc

# from suanpan.docker.arguments import File, Folder, HiveTable


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
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.input(String(key="inputModel", required=True, help="Ckpt model file."))
@dc.output(
    String(
        key="outputBboxData",
        # table="outputBboxTable",
        # partition="outputBboxPartition",
    )
)
@dc.output(
    String(
        key="outputBboxFolder",
        required=True,
        help="Directory to save bbox npy files to.",
    )
)
@dc.column(String(key="idColumn", default="id", help="ID column of inputImagesData."))
@dc.column(
    String(key="lbbColumn", default="lbb", help="Lbb column of inputImagesData.")
)
@dc.column(
    String(key="pbbColumn", default="pbb", help="Pbb column of inputImagesData.")
)
@dc.param(Bool(key="gpu", default=True, help="Use GPU if available."))
@dc.param(Int(key="gpuno", default=1, help="Number of GPU, 1~N."))
@dc.param(Int(key="margin", default=16, help="patch margin."))
@dc.param(Int(key="sidelen", default=64, help="patch side length."))
def preprocess(context):
    args = context.args

    useGpu = args.gpu and not torch.cuda.is_available()
    data = pd.read_csv(args.inputData)

    path.safeMkdirs(args.outputBboxFolder)

    config, nodNet, loss, getPbb = nodmodel.get_model()
    checkpoint = torch.load(args.inputModel)
    nodNet.load_state_dict(checkpoint["state_dict"])
    nodNet = nodNet.cuda() if useGpu else nodNet.cpu()

    config["datadir"] = args.inputDataFolder
    splitComber = SplitComb(
        args.sidelen,
        config["max_stride"],
        config["stride"],
        args.margin,
        pad_value=config["pad_value"],
    )

    ids = data[args.idColumn]
    dataset = DataBowl3Detector(ids, config, phase="test", split_comber=splitComber)
    dataLoader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=asyncio.WORKERS,
        pin_memory=False,
        collate_fn=collate,
    )

    test_detect(
        dataLoader, nodNet, getPbb, args.outputBboxFolder, config, n_gpu=args.gpuno
    )

    data[args.lbbColumn] = data[args.idColumn] + "_lbb.npy"
    data[args.pbbColumn] = data[args.idColumn] + "_pbb.npy"

    data.to_csv(args.outputBboxData)

    return data, args.outputBboxFolder


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
