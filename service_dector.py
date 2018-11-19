# coding=utf-8
from __future__ import print_function

import os
import tempfile

import cv2
from suanpan import convert, utils
from suanpan.docker.io import storage

from dsb.layers import nms
from service import DSBService

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


class ServiceDector(DSBService):
    def call(self, request, context):
        ossDataFolder = request.in2
        localDataFolder = storage.download(
            ossDataFolder, storage.getPathInTempStore(ossDataFolder)
        )

        ossResultFolder = "majik_test/dsb3/service/dector"
        localResultFolder = storage.getPathInTempStore(ossResultFolder)

        ids = os.listdir(localDataFolder)

        for i in ids:
            image = utils.loadFromNpy(os.path.join(localDataFolder, i, "data.npy"))
            pbb = utils.loadFromNpy(os.path.join(localDataFolder, i, "pbb.npy"))
            images = [img for img in pickWithPbb(image, pbb)]
            utils.saveAsImages(os.path.join(localResultFolder, i), images)

        storage.upload(ossResultFolder, localResultFolder)
        return dict(out1=ossResultFolder)


if __name__ == "__main__":
    ServiceDector().start()
