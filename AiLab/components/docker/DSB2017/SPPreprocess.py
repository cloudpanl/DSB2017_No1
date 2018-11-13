# coding=utf-8
from __future__ import division, print_function

import math
import os
from multiprocessing import Lock, Pool, Queue, Value, freeze_support

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom

from suanpan import path, utils
from suanpan.arguments import Int, String
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, Folder, Npy
from suanpan.docker.io import storage


def normalize_hu(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image


def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def mean(a):
    return sum(a) / len(a)


def preprocess_work(data):
    patient, label, IMG_SIZE_PX, SLICE_COUNT = data
    label = int(label)
    IMG_SIZE_PX = int(IMG_SIZE_PX)
    SLICE_COUNT = int(SLICE_COUNT)
    return process_data(
        patient, label, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT, visualize=False
    )


def process_data(path, label, img_px_size=50, hm_slices=20, visualize=False):
    """Preprocess a patient's slices.

    Arguments:
        path {str} -- patient directory path.
        label {int} -- label 1 or 0.

    Keyword Arguments:
        img_px_size {int} -- x, y pixels to resize x, y axis (default: {50})
        hm_slices {int} -- horizontal z axis slices count (default: {20})
        visualize {bool} -- plot and visualize the preprocessed slices interactively. (default: {False})

    Returns:
        tuple -- new_slices as 3D np.array and one-hot label, i.e. 1 -> [0, 1], 0 -> [1, 0]
    """

    print(path, label)
    slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]

    # Sorted slices by z axis.
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    # Only keep the lung part of the scan.
    # slices_len = len(slices)
    # slices = slices[int(0.25*slices_len): int(0.75*slices_len)]

    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # Empty/output bound pixels to 0
    image[image == -2000] = 0

    # Reconstruct from stored values (SV) Output units(HV) = m*SV + b.
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    image = normalize_hu(image)

    # Placeholder for preprocessed slices.
    new_slices = []
    slices = [i for i in image]

    # Resize x, y axis.
    slices = [
        cv2.resize(np.array(each_slice), (img_px_size, img_px_size))
        for each_slice in slices
    ]

    # Resize z axis.
    chunk_sizes = len(slices) // hm_slices + 1
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices - 1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices - 2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == hm_slices + 2:
        new_val = list(
            map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices]]))
        )
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if len(new_slices) == hm_slices + 1:
        new_val = list(
            map(mean, zip(*[new_slices[hm_slices - 1], new_slices[hm_slices]]))
        )
        del new_slices[hm_slices]
        new_slices[hm_slices - 1] = new_val

    if label == 1:
        label = np.array([0, 1])
    elif label == 0:
        label = np.array([1, 0])
    print("OK")
    return np.array(new_slices), label, os.path.split(path)[-1]


def flat_3d_image(img):
    layer_count = len(img)
    row_layer_count = int(math.ceil(math.sqrt(layer_count)))
    extra_layer_count = int(math.pow(row_layer_count, 2)) - layer_count
    extra_layers = np.zeros((extra_layer_count,) + img[0].shape)
    full_item = np.concatenate([img, extra_layers])
    rows = [
        np.concatenate(
            full_item[i * row_layer_count : (i + 1) * row_layer_count], axis=1
        )
        for i in range(row_layer_count)
    ]
    return np.concatenate(rows)


@dc.input(
    Folder(key="inputDataFolder", required=True, help="Kaggle DSB 2017 dataset folder.")
)
@dc.input(Csv(key="inputLabelCsv", required=True, help="Path to `stage1_labels.csv`."))
@dc.output(Npy(key="outputNpy", required=True, help="Preprocessed data as npy."))
@dc.output(Folder(key="outputImages", required=True))
@dc.param(Int(key="workers", default=10, help="Number of workers for multi-processing"))
@dc.param(Int(key="pixels", default=50, help="Pixels to resize for x,y axis."))
@dc.param(
    Int(key="slices", default=20, help="Number of slices to resize to for z axis.")
)
@dc.param(Int(key="max", help="Maximum number of patients to preprocess."))
@dc.param(String(key="prefix", default="preprocessed", help="Output file name prefix."))
def SPPreprocess(context):
    args = context.args

    data_dir = args.inputDataFolder
    pool_workers = args.workers
    prefix = args.prefix
    IMG_SIZE_PX = args.pixels
    SLICE_COUNT = args.slices

    patients = os.listdir(data_dir)
    labels_df = pd.read_csv(args.inputLabelCsv, index_col=0)

    pool = Pool(pool_workers)
    labeled_patient_paths = []
    for patient in patients:
        label = labels_df.get_value(patient, "cancer")
        patient_path = os.path.join(data_dir, patient)
        labeled_patient_paths.append((patient_path, label, IMG_SIZE_PX, SLICE_COUNT))

    # Multi-process pool.
    records = pool.map(preprocess_work, labeled_patient_paths)

    # Clean up multi-process pool.
    pool.terminate()
    pool.join()

    file_name = "{}-{}-{}-{}.npy".format(prefix, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT)
    file_name = storage.getPathInTempStore(file_name)
    utils.saveAsNpy(file_name, records)

    print("Number of samples:", len(records))
    print("Per sample shape:", records[0][0].shape)

    # See how many positive labels in the sample dataset.
    sample_lables = [s[1][1] for s in records]
    sample_lables = np.array(sample_lables)
    print("Positive labels rate:", sample_lables.mean())
    print("Preprocessed file saved to: {}".format(file_name))

    images_folder = storage.getPathInTempStore(args.outputImages)
    for image, label, patient in records:
        image2d = flat_3d_image(image)
        image_path = os.path.join(images_folder, "{}.png".format(patient))
        path.safeMkdirsForFile(image_path)
        cv2.imwrite(image_path, image2d * 255)
        print("Save image: {}".format(image_path))

    return file_name, images_folder


if __name__ == "__main__":
    SPPreprocess()  # pylint: disable=no-value-for-parameter
