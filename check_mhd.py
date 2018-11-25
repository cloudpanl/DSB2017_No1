import os
import sys

from component_luna_preprocess import load_itk_image
from suanpan import asyncio

root = sys.argv[1]


def _check(filepath):
    try:
        load_itk_image(filepath)
        print("Good: ", filepath)
    except Exception as e:
        print("Error: ", filepath)
        # print(e)


with asyncio.multiProcess() as pool:
    files = [
        os.path.join(root, file)
        for root, folders, files in os.walk(root)
        for file in files
        if file.endswith(".mhd")
    ]
    pool.map(_check, files)
