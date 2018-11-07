# coding: utf-8
from __future__ import print_function

import os
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image

from dsb.preprocessing.step1 import step1_python
from suanpan import asyncio
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, HiveTable


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode="nearest", order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError("wrong shape")


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def lumTrans(img):
    lungwin = np.array([-1200.0, 600.0])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype("uint8")
    return newimg


def savenpy(id, annos, filelist, data_path, prep_folder):
    """Preprocess one annotated stage1 scan

    Arguments:
        id {int} --Integer index id in filelist.
        annos {np.array} -- stage 1 annotation labels.
        filelist {list} -- DSB stage1 or similar directory
        data_path {str} -- DSB stage1 or similar directory path.
        prep_folder {str} -- Directory to save preprocessed _clean and _label .npy files.
    """

    resolution = np.array([1, 1, 1])  # Resolution in mm for 3 axis (z, x, y).
    name = filelist[id]
    label = annos[annos[:, 0] == name]
    label = label[:, [3, 1, 2, 4]].astype("float")

    try:
        im, m1, m2, spacing = step1_python(os.path.join(data_path, name))
        Mask = m1 + m2

        newshape = np.round(np.array(Mask.shape) * spacing / resolution)
        xx, yy, zz = np.where(Mask)
        box = np.array(
            [
                [np.min(xx), np.max(xx)],
                [np.min(yy), np.max(yy)],
                [np.min(zz), np.max(zz)],
            ]
        )
        box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        box = np.floor(box).astype("int")
        margin = 5
        extendbox = np.vstack(
            [
                np.max([[0, 0, 0], box[:, 0] - margin], 0),
                np.min([newshape, box[:, 1] + 2 * margin], axis=0).T,
            ]
        ).T
        extendbox = extendbox.astype("int")

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask  # Fixed '-' -> '^'
        bone_thresh = 210
        pad_value = 170
        im[np.isnan(im)] = -2000
        sliceim = lumTrans(im)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype("uint8")
        bones = sliceim * extramask > bone_thresh
        sliceim[bones] = pad_value
        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[
            extendbox[0, 0] : extendbox[0, 1],
            extendbox[1, 0] : extendbox[1, 1],
            extendbox[2, 0] : extendbox[2, 1],
        ]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name + "_clean.npy"), sliceim)

        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        elif len(label[0]) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        elif label[0][0] == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            haslabel = 1
            label2 = np.copy(label).T
            label2[:3] = label2[:3][[0, 2, 1]]
            # (z, x, y axis labeled in pixels) * spacing(mm per pixel, diff for z and (x, y)) / resolution(in mm)
            label2[:3] = (
                label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            )
            # r/radius labeled in pixels * spacing of x (mm per pixel) / resolution of x(in mm)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(prep_folder, name + "_label.npy"), label2)
        print(name)
    except:
        print("bug in {}".format(name))


def full_prep(
    preprocess_result_path, stage1_data_path, stage1_annos_path, workers=1, force=False
):
    """Preprocess annotated stage1 or similar datasets.

    Arguments:
        preprocess_result_path {str} -- Directory to save preprocessed _clean and _label .npy files.
        stage1_data_path {str} -- DSB stage1 or similar directory path.
        stage1_annos_path {str} -- Directory where DSB stage 1 annotation csv files exists.

    Keyword Arguments:
        workers {int} -- Number of workers for multi-processing. (default: {1})
        force {bool} -- Force overwrite existing preprocessed result. (default: {False})
    """

    warnings.filterwarnings("ignore")

    alllabelfiles = [
        os.path.join(stage1_annos_path, fname)
        for fname in os.listdir(stage1_annos_path)
    ]
    tmp = []
    for f in alllabelfiles:
        # Turn the file path to absolute path if necessary.
        if not os.path.isabs(f):
            f = os.path.join(os.path.dirname(os.path.realpath(__file__)), f)
        content = np.array(pd.read_csv(f))
        content = content[content[:, 0] != np.nan]
        tmp.append(content[:, :5])
    alllabel = np.concatenate(tmp, 0)
    filelist = os.listdir(stage1_data_path)

    if not os.path.exists(preprocess_result_path):
        os.mkdir(preprocess_result_path)
    # eng.addpath('preprocessing/',nargout=0)

    print("starting preprocessing")
    pool = Pool(workers)  # Pool size for multiprocessing.
    # filelist = [f for f in os.listdir(stage1_data_path)]
    partial_savenpy = partial(
        savenpy,
        annos=alllabel,
        filelist=filelist,
        data_path=stage1_data_path,
        prep_folder=preprocess_result_path,
    )

    N = len(filelist)
    _ = pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()
    print("end preprocessing")


def scan_prep_results(folder, id_column, image_column, label_column):
    data_suffix = "_clean.npy"
    label_suffix = "_label.npy"
    return pd.DataFrame(
        [
            (file[: -len(data_suffix)], file, file.replace(data_suffix, label_suffix))
            for file in os.listdir(folder)
            if file.endswith(data_suffix)
        ],
        columns=[id_column, image_column, label_column],
    )


@dc.input(
    Folder(
        key="inputDataFolder",
        required=True,
        help="DSB stage1/2 or similar directory path.",
    )
)
@dc.input(
    Folder(
        key="inputLabelsFolder",
        required=True,
        help="DSB stage1/2 annos or similar directory path.",
    )
)
@dc.output(
    HiveTable(
        key="outputData", table="outputDataTable", partition="outputDataPartition"
    )
)
@dc.output(
    Folder(
        key="outputDataFolder",
        required=True,
        help="Directory to save preprocessed npy files to.",
    )
)
@dc.output(String(key="idColumn", default="patient"))
@dc.output(String(key="imageColumn", default="image_path"))
@dc.output(String(key="labelColumn", default="label_path"))
def SPDSB3Preprocess(context):
    args = context.args

    stage1Path = args.inputDataFolder
    preprocessResultPath = args.outputDataFolder
    stage1AnnosPath = args.inputLabelsFolder

    full_prep(
        preprocessResultPath, stage1Path, stage1AnnosPath, workers=asyncio.WORKERS
    )
    data = scan_prep_results(
        preprocessResultPath, args.idColumn, args.imageColumn, args.labelColumn
    )

    return data, args.outputDataFolder


if __name__ == "__main__":
    SPDSB3Preprocess()  # pylint: disable=no-value-for-parameter
