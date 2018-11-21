# coding=utf-8
from __future__ import print_function

import os

import torch
from torch.utils.data import DataLoader

import dsb.net_detector as nodmodel
from dsb.data_detector import DataBowl3Detector, collate
from dsb.split_combine import SplitComb
from dsb.test_detect import test_detect
from suanpan import path
from suanpan.services import Handler as h
from suanpan.services import Service
from suanpan.services.arguments import Checkpoint, Folder


class ServicePredict(Service):

    @h.input(Checkpoint(key="inputCheckpoint", required=True))
    @h.input(Folder(key="inputDataFolder", required=True))
    @h.output(Folder(key="outputDataFolder", required=True))
    def call(self, context):
        args = context.args

        checkpoint = torch.load(args.inputCheckpoint)
        dataFolder = args.inputDataFolder

        outputDataFolder = args.outputDataFolder

        sidelen = 64
        margin = 16
        batchSize = 16
        workers = 0

        config, nodNet, loss, getPbb = nodmodel.get_model()
        nodNet.load_state_dict(checkpoint["state_dict"])
        nodNet = nodNet.cuda() if torch.cuda.is_available() else nodNet.cpu()

        config["datadir"] = dataFolder
        splitComber = SplitComb(
            sidelen,
            config["max_stride"],
            config["stride"],
            margin,
            pad_value=config["pad_value"],
        )

        ids = set(i.split("_")[0] for i in os.listdir(dataFolder))
        dataset = DataBowl3Detector(ids, config, phase="test", split_comber=splitComber)
        dataLoader = DataLoader(
            dataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=workers,
            pin_memory=False,
            collate_fn=collate,
        )

        test_detect(
            dataLoader,
            nodNet,
            getPbb,
            outputDataFolder,
            config,
            n_gpu=torch.cuda.device_count(),
        )

        for i in ids:
            folder = path.safeMkdirs(os.path.join(outputDataFolder, i))
            os.rename(
                os.path.join(outputDataFolder, "{}_lbb.npy".format(i)),
                os.path.join(folder, "lbb.npy"),
            )
            os.rename(
                os.path.join(outputDataFolder, "{}_pbb.npy".format(i)),
                os.path.join(folder, "pbb.npy"),
            )
            os.rename(
                os.path.join(dataFolder, "{}_clean.npy".format(i)),
                os.path.join(folder, "data.npy"),
            )

        return outputDataFolder


if __name__ == "__main__":
    ServicePredict().start()
