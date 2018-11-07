# coding: utf-8
from __future__ import print_function

import os

import cv2

from dsb.layers import nms
from suanpan import convert, path, utils
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, HiveTable

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
    HiveTable(key="inputData", table="inputDataTable", partition="inputDataPartition")
)
@dc.input(
    Folder(
        key="inputDataFolder",
        required=True,
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.input(
    Folder(
        key="inputBboxDataFolder",
        required=True,
        help="Directory to save bbox npy files to.",
    )
)
@dc.output(
    Folder(
        key="outputImagesFolder",
        required=True,
        help="Directory to save masked image files to.",
    )
)
@dc.column(String(key="idColumn", default="id"))
@dc.column(String(key="dataColumn", default="data_path"))
@dc.column(String(key="pbbColumn", default="pbb_path"))
def SPData2MaskImages(context):
    args = context.args
    data = args.inputData

    for _, row in data.iterrows():
        image = utils.loadFromNpy(
            os.path.join(args.inputDataFolder, row[args.dataColumn])
        )
        pdd = utils.loadFromNpy(
            os.path.join(args.inputBboxDataFolder, row[args.pbbColumn])
        )
        images = [img for img in pickWithPbb(image, pdd)]
        utils.saveAsImages(
            os.path.join(args.outputImagesFolder, row[args.idColumn]), images
        )

    return args.outputImagesFolder


if __name__ == "__main__":
    SPData2MaskImages()  # pylint: disable=no-value-for-parameter
