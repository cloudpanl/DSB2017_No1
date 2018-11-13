# coding=utf-8
from __future__ import print_function

import cv2
import imageio
import numpy as np

from suanpan import asyncio, convert, path


def loadFromNpy(filepath):
    return np.load(filepath, encoding="latin1")


def saveAsNpy(filepath, data):
    path.safeMkdirsForFile(filepath)
    np.save(filepath, data, fix_imports=True)
    return filepath


def saveImage(filepath, image):
    path.safeMkdirsForFile(filepath)
    cv2.imwrite(filepath, image)
    return filepath


def saveAsFlatImage(filepath, data):
    image = convert.flatAsImage(data)
    return saveImage(filepath, image)


def saveAsAnimatedGif(filepath, data):
    image3D = convert.to3D(data)
    path.safeMkdirsForFile(filepath)
    imageio.mimsave(filepath, image3D)
    return filepath


def saveAsImage(filepath, data, flag=None):
    mapping = {
        None: saveImage,
        "flat": saveAsFlatImage,
        "animated": saveAsAnimatedGif,
    }
    func = mapping.get(flag)
    if not func:
        raise Exception("Unknow flag: {}".format(flag))
    return func(filepath, data)


def saveAsImages(filepathPrefix, images, workers=None):
    counts = len(images)
    n = len(str(counts))
    workers = workers or min(counts, asyncio.WORKERS)
    with asyncio.multiThread(workers) as pool:
        for index, image in enumerate(images):
            pool.apply_async(
                saveAsImage,
                args=("{}_{}.png".format(filepathPrefix, str(index).zfill(n)), image),
            )
    return filepathPrefix


def saveAllAsImages(filepathPrefix, data, workers=None, pool=None):
    def _save(filepathPrefix, image3D, pool):
        layers = len(image3D)
        n = len(str(layers))
        print(layers, n)
        for index, image in enumerate(image3D):
            pool.apply_async(
                saveImage,
                args=("{}_{}.png".format(filepathPrefix, str(index).zfill(n)), image),
            )
        pool.apply_async(
            saveAsFlatImage, args=("{}.png".format(filepathPrefix), image3D)
        )
        pool.apply_async(
            saveAsAnimatedGif, args=("{}.gif".format(filepathPrefix), image3D)
        )

    image3D = convert.to3D(data)
    if pool:
        _save(filepathPrefix, image3D, pool)
    else:
        layers = len(image3D)
        workers = workers or min(layers + 2, asyncio.WORKERS)
        with asyncio.multiThread(workers) as pool:
            _save(filepathPrefix, image3D, pool)
    return filepathPrefix
