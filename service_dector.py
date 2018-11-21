# coding=utf-8
from __future__ import print_function

import os
import tempfile

import cv2

from dsb.layers import nms
from suanpan import convert, utils
from suanpan.services import Handler as h
from suanpan.services import Service
from suanpan.services.arguments import Folder

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


class ServiceDector(Service):

    @h.input(Folder(key="inputDataFolder", required=True))
    @h.output(Folder(key="outputDataFolder", required=True))
    def call(self, context):
        args = context.args

        inputDataFolder = args.inputDataFolder
        outputDataFolder = args.outputDataFolder

        ids = os.listdir(inputDataFolder)

        for i in ids:
            image = utils.loadFromNpy(os.path.join(inputDataFolder, i, "data.npy"))
            pbb = utils.loadFromNpy(os.path.join(inputDataFolder, i, "pbb.npy"))
            images = [img for img in pickWithPbb(image, pbb)]
            utils.saveAsImages(os.path.join(outputDataFolder, i), images)

        return outputDataFolder


if __name__ == "__main__":
    ServiceDector().start()
