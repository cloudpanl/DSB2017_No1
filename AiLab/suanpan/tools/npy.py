# coding=utf-8
from __future__ import division, print_function

import os
import shutil
import time
import webbrowser

import cv2
import numpy as np

from suanpan import convert, utils
from suanpan.arguments import Bool, String
from suanpan.components import Component as c


def showAsImage(data, temp="tmp", flag=None):
    imageType = "gif" if flag == "animated" else "png"
    filepath = os.path.join(temp, "{}.{}".format(time.time(), imageType))
    utils.saveAsImage(filepath, data, flag=flag)
    showImage(filepath)


def showImage(filepath):
    url = "file://" + os.path.abspath(filepath)
    webbrowser.open(url)


@c.input(String(key="npy", required=True))
@c.param(String(key="toImage"))
@c.param(String(key="flag"))
@c.param(Bool(key="show", default=False))
def SPNpyTools(context):
    args = context.args

    data = utils.loadFromNpy(args.npy)
    if args.toImage:
        filepath = utils.saveAsImage(args.toImage, data, flag=args.flag)
        if args.show:
            showImage(filepath)
    else:
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    SPNpyTools()  # pylint: disable=no-value-for-parameter
