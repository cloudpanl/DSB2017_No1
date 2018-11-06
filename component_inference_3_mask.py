# coding: utf-8
from __future__ import print_function

import os
from multiprocessing import freeze_support

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader

import dsb.net_detector as nodmodel
from dsb import preprocessing
from dsb.data_detector import DataBowl3Detector, collate

# from suanpan.docker.arguments import File, Folder, HiveTable
from dsb.layers import iou, nms
from dsb.split_combine import SplitComb
from dsb.test_detect import test_detect
from suanpan import asyncio, convert, path, utils
from suanpan.arguments import Bool, Int, String
from suanpan.components import Component as dc

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def rectangle(image, box, color=RED, width=1, *arg, **kwargs):
    cy, cx, aa = box
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.rectangle(
        image, (cx - aa, cy - aa), (cx + aa, cy + aa), color, width, *arg, **kwargs
    )


def addMask(image, box, maskFunc=rectangle, *arg, **kwargs):
    return maskFunc(image, box, *arg, **kwargs)


def pickWithPbb(image, pbb, maskFunc=rectangle, *arg, **kwargs):
    image3D = convert.to3D(image)
    # Probabilities threshold
    pbb = pbb[pbb[:, 0] > -1]
    # NMS : Non-max suppression
    # Remove overlapping boxes.
    pbb = nms(pbb, 0.05)

    def _mask(box):
        box = box.astype("int")[1:]
        zindex, box = box[0], box[1:]
        return addMask(image3D[zindex], box, maskFunc=rectangle, *arg, **kwargs)

    return (_mask(box) for box in pbb)


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
@dc.input(
    String(
        key="inputBboxDataFolder",
        required=True,
        help="Directory to save bbox npy files to.",
    )
)
@dc.output(
    String(
        key="outputImageFolder",
        required=True,
        help="Directory to save masked image files to.",
    )
)
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="imageColumn", default="image"))
@dc.column(String(key="pbbColumn", default="pbb"))
def preprocess(context):
    args = context.args
    data = pd.read_csv(args.inputData)

    for _, row in data.iterrows():
        image = utils.loadFromNpy(
            os.path.join(args.inputDataFolder, row[args.imageColumn])
        )
        pdd = utils.loadFromNpy(
            os.path.join(args.inputBboxDataFolder, row[args.pbbColumn])
        )
        images = [img for img in pickWithPbb(image, pdd)]
        utils.saveImages(
            os.path.join(args.outputImageFolder, row[args.idColumn]), images
        )

    return args.outputImageFolder


if __name__ == "__main__":
    preprocess()  # pylint: disable=no-value-for-parameter
