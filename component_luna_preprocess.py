# coding: utf-8
from __future__ import print_function

import os
import shutil
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image

from dsb import preprocessing
from dsb.preprocessing.step1 import step1_python
from suanpan import asyncio
from suanpan.arguments import String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, Folder, HiveTable
from suanpan.io import storage


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


def worldToVoxelCoord(worldCoord, origin, spacing):
    """Transform x, y, z annotation coordinates

    Arguments:
        worldCoord {np.array} -- [x, y, z] LUNA16 annotation coordinates.
        origin {np.array} -- [x, y, z] origin of the 3D scan.
        spacing {np.array} -- [x, y, z] spacing of the 3D scan.

    Returns:
        [np.array] -- Transformed x, y, z coordinates.
    """

    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith("TransformMatrix")][0]
        transformM = np.array(line.split(" = ")[1].split(" ")).astype("float")
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip


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


def savenpy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    """Preprocess luna datasets

    Arguments:
        id {int} --Integer index id in filelist.
        annos {pd.dataframe} -- Pandas dataframe loaded from csv contains rows of ids(as numbers) and annotations.
        filelist {list} -- A list of ids.
        luna_segment {str} -- Path to directory where <id>.mhd and <id>.raw files exists.
        luna_data {str} --  directory path where LUNA16 abbrivated/renamed files is stored.
        savepath {str} -- Directory to save preprocessed _clean and _label .npy files.
    """

    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    # resolution = np.array([2,2,2])
    name = filelist[id]
    print("reading luna_segment: ", os.path.join(luna_segment, name + ".mhd"))
    Mask, origin, spacing, isflip = load_itk_image(
        os.path.join(luna_segment, name + ".mhd")
    )
    if isflip:
        Mask = Mask[:, ::-1, ::-1]
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype("int")
    m1 = Mask == 3
    m2 = Mask == 4
    Mask = m1 + m2

    xx, yy, zz = np.where(Mask)
    box = np.array(
        [[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]]
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

    this_annos = np.copy(annos[annos[:, 0] == int(name)])

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1 + dm2
        Mask = m1 + m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim, origin, spacing, isflip = load_itk_image(
            os.path.join(luna_data, name + ".mhd")
        )
        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            print("flip!")
        sliceim = lumTrans(sliceim)
        sliceim = sliceim * dilatedMask + pad_value * (1 - dilatedMask).astype("uint8")
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value

        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[
            extendbox[0, 0] : extendbox[0, 1],
            extendbox[1, 0] : extendbox[1, 1],
            extendbox[2, 0] : extendbox[2, 1],
        ]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(savepath, name + "_clean.npy"), sliceim)

    if islabel:

        this_annos = np.copy(annos[annos[:, 0] == int(name)])
        label = []
        if len(this_annos) > 0:

            for c in this_annos:  # For each annotation.
                # transform x, y, z annotation coordinates.
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = (
                label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            )
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name + "_label.npy"), label2)

    print(name)


def preprocess_luna(
    luna_segment, preprocess_result_path, luna_data, luna_label, workers=1, force=False
):
    """Preprocess LUNA 16 or similar datasets.

    Arguments:
        luna_segment {str} -- LUNA16 `seg-lungs-LUNA16` directory path.
        preprocess_result_path {str} -- Directory to save preprocessed _clean and _label .npy files.
        luna_data {str} -- Provide an empty directory path to store LUNA16 abbrivated/renamed files.
        luna_label {str} -- LUNA16 annotations, lunaqualified.csv.

    Keyword Arguments:
        workers {int} -- Number of workers for multi-processing. (default: {1})
        force {bool} -- Force overwrite existing preprocessed result. (default: {False})
    """

    savepath = preprocess_result_path
    print("starting preprocessing luna")
    filelist = [f.split(".mhd")[0] for f in os.listdir(luna_data) if f.endswith(".mhd")]
    luna_label_csv = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), luna_label
    )
    annos = np.array(pd.read_csv(luna_label_csv))

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    pool = Pool(workers)  # Pool size for multiprocessing.
    partial_savenpy_luna = partial(
        savenpy_luna,
        annos=annos,
        filelist=filelist,
        luna_segment=luna_segment,
        luna_data=luna_data,
        savepath=savepath,
    )

    N = len(filelist)
    # savenpy(1)
    _ = pool.map(partial_savenpy_luna, range(N))
    pool.close()
    pool.join()
    print("end preprocessing luna")


def prepare_luna(luna_raw, luna_abbr, luna_data, luna_segment, force=False):
    """Prepare LUNA16 datasets, rename id to shorter ones, place in `luna_data` directory.

    Arguments:
        luna_raw {str} -- LUNA16 or similar directory path.
        luna_abbr {str} -- A mapping of luna file names to shorter ones. e.g. '012' <- '1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217'
        luna_data {str} -- tmp folder to move renamed .raw and .mhd files.
        luna_segment {str} -- `seg-lungs-LUNA16` folder path.

    Keyword Arguments:
        force {bool} -- Force overwrite existing preprocessed result. (default: {False})
    """

    print("start changing luna name")
    # A persistent file serves as a flag whether the preprocessing has been done before.

    subsetdirs = [
        os.path.join(luna_raw, f)
        for f in os.listdir(luna_raw)
        if f.startswith("subset") and os.path.isdir(os.path.join(luna_raw, f))
    ]
    if not os.path.exists(luna_data):
        os.mkdir(luna_data)

    luna_abbr_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), luna_abbr)
    abbrevs = np.array(pd.read_csv(luna_abbr_csv, header=None))
    namelist = list(abbrevs[:, 1])
    ids = abbrevs[:, 0]

    # Move renamed .raw and .mhd files to 'luna_data'(a.k.a luna/allset).
    for d in subsetdirs:  # For each luna subset folder.
        print("Subset: {}".format(d))
        files = os.listdir(d)
        files.sort()
        for f in files:
            name = f[:-4]  # remove the extension of file name.
            id = ids[namelist.index(name)]
            filename = "0" * (3 - len(str(id))) + str(id)
            fromfile, tofile = (
                os.path.join(d, f),
                os.path.join(luna_data, filename + f[-4:]),
            )
            shutil.move(fromfile, tofile)
            print("{} -> {}".format(fromfile, tofile))

    # Update the 'ElementDataFile' key in .mhd file with new .raw file name in 'luna_data'(a.k.a luna/allset).
    files = [f for f in os.listdir(luna_data) if f.endswith("mhd")]
    for file in files:
        with open(os.path.join(luna_data, file), "r") as f:
            content = f.readlines()
            id = file.split(".mhd")[0]
            filename = "0" * (3 - len(str(id))) + str(id)
            content[-1] = "ElementDataFile = " + filename + ".raw\n"
            # print(content[-1])
        with open(os.path.join(luna_data, file), "w") as f:
            f.writelines(content)

    # Renamed .mhd files in 'luna_segment'(a.k.a luna/seg-lungs-LUNA16) to abbreviated names.
    print("Segment: {}".format(luna_segment))
    seglist = os.listdir(luna_segment)
    for f in seglist:
        if f.endswith(".mhd"):

            name = f[:-4]
            lastfix = f[-4:]
        else:
            name = f[:-5]
            lastfix = f[-5:]
        if name in namelist:
            id = ids[namelist.index(name)]
            filename = "0" * (3 - len(str(id))) + str(id)
            fromfile, tofile = (
                os.path.join(luna_segment, f),
                os.path.join(luna_segment, filename + lastfix),
            )
            shutil.move(fromfile, tofile)
            print("{} -> {}".format(fromfile, tofile))

    # Update the 'ElementDataFile' key in lung segment .mhd file in
    # 'luna_segment'(a.k.a luna/seg-lungs-LUNA16) with abbreviated .zraw file name.
    files = [f for f in os.listdir(luna_segment) if f.endswith("mhd")]
    for file in files:
        with open(os.path.join(luna_segment, file), "r") as f:
            content = f.readlines()
            id = file.split(".mhd")[0]
            filename = "0" * (3 - len(str(id))) + str(id)
            content[-1] = "ElementDataFile = " + filename + ".zraw\n"
            # print(content[-1])
        with open(os.path.join(luna_segment, file), "w") as f:
            f.writelines(content)
    print("end changing luna name")


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
        key="inputRawFolder", required=True, help="Luna raw or similar directory path."
    )
)
@dc.input(
    Folder(
        key="inputSegmentFolder",
        required=True,
        help="Luna segment or similar directory path.",
    )
)
@dc.input(Csv(key="inputAbbr", required=True, help="Luna abbr path."))
@dc.input(Csv(key="inputLabels", required=True, help="Luna label path."))
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
def SPLunaPreprocess(context):
    args = context.args

    lunaRawPath = args.inputRawFolder
    lunaSegmentPath = args.inputSegmentFolder
    preprocessResultPath = args.outputDataFolder
    lunaAbbrPath = args.inputAbbr
    lunaLabelsPath = args.inputLabels

    lunaDataPath = storage.getPathInTempStore("luna_data_{}".format(time.time()))

    prepare_luna(lunaRawPath, lunaAbbrPath, lunaDataPath, lunaSegmentPath)
    preprocess_luna(
        lunaSegmentPath,
        preprocessResultPath,
        lunaDataPath,
        lunaLabelsPath,
        asyncio.WORKERS,
    )
    data = scan_prep_results(
        preprocessResultPath, args.idColumn, args.imageColumn, args.labelColumn
    )

    return data, preprocessResultPath


if __name__ == "__main__":
    SPLunaPreprocess()  # pylint: disable=no-value-for-parameter
