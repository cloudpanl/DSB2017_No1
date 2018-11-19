# coding=utf-8
from __future__ import print_function

import os
import tempfile

import torch
from suanpan import asyncio, path
from suanpan.docker.io import storage
from torch.utils.data import DataLoader

import dsb.net_detector as nodmodel
from dsb.data_detector import DataBowl3Detector, collate
from dsb.split_combine import SplitComb
from dsb.test_detect import test_detect
from service import DSBService


class ServicePredict(DSBService):
    def call(self, request, context):
        ossCkptPath = request.in1
        localCkptPath = storage.download(
            ossCkptPath, storage.getPathInTempStore(ossCkptPath)
        )
        checkpoint = torch.load(localCkptPath)

        ossDataFolder = request.in2
        localDataFolder = storage.download(
            ossDataFolder, storage.getPathInTempStore(ossDataFolder)
        )

        ossResultFolder = "majik_test/dsb3/service/predict"
        localResultFolder = storage.getPathInTempStore(ossResultFolder)

        sidelen = 64
        margin = 16
        batchSize = 16
        workers = asyncio.WORKERS

        config, nodNet, loss, getPbb = nodmodel.get_model()
        nodNet.load_state_dict(checkpoint["state_dict"])
        nodNet = nodNet.cuda() if torch.cuda.is_available() else nodNet.cpu()

        config["datadir"] = localDataFolder
        splitComber = SplitComb(
            sidelen,
            config["max_stride"],
            config["stride"],
            margin,
            pad_value=config["pad_value"],
        )

        ids = os.listdir(localDataFolder)
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
            localResultFolder,
            config,
            n_gpu=torch.cuda.device_count(),
        )

        for i in ids:
            folder = path.safeMkdirs(os.path.join(localResultFolder, i))
            os.rename(
                os.path.join(localResultFolder, "{}_lbb.npy".format(i)),
                os.path.join(folder, "lbb.npy"),
            )
            os.rename(
                os.path.join(localResultFolder, "{}_pbb.npy".format(i)),
                os.path.join(folder, "pbb.npy"),
            )
            os.rename(
                os.path.join(localDataFolder, "{}_clean.npy".format(i)),
                os.path.join(folder, "data.npy"),
            )

        storage.upload(ossResultFolder, localResultFolder)
        return dict(out1=ossResultFolder)


if __name__ == "__main__":
    ServicePredict().start()
