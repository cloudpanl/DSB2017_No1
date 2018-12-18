import numpy as np
import pydicom as dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation, binary_erosion, binary_closing, convex_hull_image
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage.morphology import square
from skimage import measure
from skimage.segmentation import clear_border
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import SimpleITK as sitk
from scipy import ndimage as ndi
import pandas as pd
import glob

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

DATA_BASE_PATH = '/home/wly/prepare/Data/'
OUT_PUT_NP_PATH = '/home/wly/prepare/NP/'
OUT_PUT_IM_PATH = '/home/wly/prepare/IM/'
ANNOTATIONS_PATH = '/home/wly/prepare/RAW/annotations.csv'


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
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


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8) + 1

    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def get_segmented_lungs(im):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10)  # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    # get_high_vals = binary == 0
    # im[get_high_vals] = -2000

    if np.sum(binary) > 4:
        binary = dilation(binary, square(5))
        binary = convex_hull_image(binary)

    return binary


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    # c = np.sqrt(np.sum(np.square([x,y])))

    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                               truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                               truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()

        for prop in properties:
            if prop.area * spacing.prod() > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
    return bw


def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def get_normalized_img_unit8(img):
    img = img.astype(np.float)
    min = img.min()
    max = img.max()
    img -= min
    img /= max - min
    img *= 255
    res = img.astype(np.uint8)
    return res


def lumTrans(img):
    lungwin = np.array([MIN_BOUND, MAX_BOUND])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def resample(image, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    # spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    # resize_factor = spacing / new_spacing
    # new_real_shape = image.shape * resize_factor
    # new_shape = np.round(new_real_shape)
    # real_resize_factor = new_shape / image.shape
    # new_spacing = spacing / real_resize_factor
    # image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    # return image, new_spacing
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def seq(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return ([start + step * i for i in range(n + 1)])
        # return range(n+1)
    else:
        return ([])


def seq_c(start, stop, step=1):
    n = int(round((stop - start) / float(step)))
    if n > 1:
        return range(int(round(start)), int(round(stop)), step)
    else:
        return ([])


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def draw_circles(image, cands, origin, spacing):
    # make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]
    image_mask = np.zeros(image.shape)

    # run over all the nodules in the lungs
    for ca in cands:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the worldcoordinate
        # retute vox  weizhi
        image_coord = world_2_voxel(image_coord, origin, spacing)

        # determine the range of the nodule
        noduleRange = seq_c(-radius, radius, RESIZE_SPACING[0])

        # create the mask
        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    image_mask[int(np.round(image_coord[0])) + z, int(np.round(image_coord[1])) + y, int(
                        np.round(image_coord[2])) + x] = int(1)

                    # coords = world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, spacing)
                    # # print coords
                    # if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) <= radius:
                    #     image_mask[int(np.round(coords[0])), int(np.round(coords[1])), int(np.round(coords[2]))] = int(1)

    return image_mask


def draw_circles2(image, cands, origin, spacing):
    # make empty matrix, which will be filled with the mask
    RESIZE_SPACING = [1, 1, 1]
    # run over all the nodules in the lungs\
    circles = []
    for ca in cands:
        # get middel x-,y-, and z-worldcoordinate of the nodule
        radius = np.ceil(ca[4]) / 2
        coord_x = ca[1]
        coord_y = ca[2]
        coord_z = ca[3]
        image_coord = np.array((coord_z, coord_y, coord_x))

        # determine voxel coordinate given the worldcoordinate
        # retute vox  weizhi
        image_coord = world_2_voxel(image_coord, origin, spacing)

        circles.append(np.append(image_coord, radius*spacing[1]))

    return np.array(circles)


def step_1_prepare_luna(seriesuid):
    path = DATA_BASE_PATH + seriesuid + '.mhd'
    sliceim, origin, spacing, isflip = load_itk_image(path)

    # sliceim_n, cpacing_n = resample(sliceim, spacing)

    img_255 = lumTrans(sliceim)

    for i, lung in enumerate(img_255):
        cv2.imwrite(os.path.join(OUT_PUT_IM_PATH, seriesuid + '_orrg_%d.png' % i), lung)

    segmented_lungs_fill_2 = segment_lung_from_ct_scan(sliceim)

    out_npy = img_255 * segmented_lungs_fill_2
    # add 170
    clear_0 = segmented_lungs_fill_2 == 0
    # # get_high_vals2 = out_npy >= 230
    out_npy[clear_0] = 170
    print 'out_npy shape :', out_npy.shape

    np.save(os.path.join(OUT_PUT_NP_PATH, seriesuid + '_clean.npy'), out_npy)
    for i, lung in enumerate(out_npy):
        cv2.imwrite(os.path.join(OUT_PUT_IM_PATH, seriesuid + '_clean_%d.png' % i), lung)

    annotations = pd.read_csv(ANNOTATIONS_PATH)
    annos = np.array(annotations[annotations.seriesuid == seriesuid])
    if len(annos) == 0:
        nodule_label = np.array([[0, 0, 0, 0]])
    else:
        # nodule_mask = draw_circles(sliceim, annos, origin, spacing)
        nodule_label = draw_circles2(sliceim, annos, origin, spacing)
        #

    np.save(os.path.join(OUT_PUT_NP_PATH , seriesuid + '_label.npy'), nodule_label)

    # nodule_where = nodule_mask == 1
    # nodule_mask[nodule_where] = 255t
    # for i, nodule in enumerate(nodule_mask):
    #     cv2.imwrite(os.path.join(OUT_PUT_IM_PATH, seriesuid + '_label_%d.png' % i), nodule)

import multiprocessing
if __name__ == '__main__':
    files_mhd = glob.glob('/home/wly/prepare/Data/*.mhd')
    # print files_mhd
    pool = multiprocessing.Pool(processes=5)
    for file in files_mhd:
        sername = (file.split('.mhd')[0]).split('/')[-1]
        print sername
        pool.apply_async(step_1_prepare_luna, (sername,))
    pool.close()
    pool.join()
    print 'OVER !!!'
