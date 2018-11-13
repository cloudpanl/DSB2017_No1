# coding=utf-8
from __future__ import print_function

import os
import shutil


def safeMkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safeMkdirsForFile(filepath):
    return safeMkdirs(os.path.dirname(os.path.abspath(filepath)))


def merge(paths, dist, mode="move"):
    safeMkdirs(dist)
    mergeFunc = getattr(shutil, mode)
    for path in paths:
        if os.path.isfile(path):
            mergeFunc(path, dist)
        else:
            for p in os.listdir(path):
                subpath = os.path.join(path, p)
                mergeFunc(subpath, dist)
            if mode == "move":
                os.rmdir(path)
    return dist
