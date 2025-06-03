# -*- coding: utf-8 -*-
"""
Created on 2023/12/08 15:28
@author: yuhang
This is a function library for nodule progression detection method based on HU ratio statistics inference.
"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import imageio
import os
from numpy import *
import numpy as np
from PIL import Image
import math
import pandas as pd
import pathlib
import collections
import matplotlib.patches as patches
import time
import gc

from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
import cv2
import csv
import copy
from statsmodels.distributions.empirical_distribution import ECDF
from datetime import datetime
import re
import pylab
from scipy.signal import convolve2d

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import cohen_kappa_score

import itk

ROOT_PATH = '/teams/Thymoma_1685081756'

LUNA16_ANNO_PATH = '/luna16_dataset/CSVFILES/annotations.csv'
LUNA16_DATA_PATH = '/luna16_dataset/'
LUNA16_PROCESSED_DATA_PATH = '/luna16_dataset/processed_luna_npy_data/'
LUNA16_GENERATED_DATA_PATH = '/luna16_dataset/generated_luna_npy_data/'
LUNA16_CSV_PATH = '/luna16_dataset/luna16_csv_data/SimulationData_new/'

LUNA16_THER_CSV_PATH = '/luna16_dataset/luna16_csv_data/Theoretical_Hypothesis/'
LUNA16_CLINICAL_CSV_PATH_ABNORMAL = '/luna16_dataset/luna16_csv_data/Clinical_Hypothesis_Abnormal/'
LUNA16_CLINICAL_CSV_PATH_NORMAL = '/luna16_dataset/luna16_csv_data/Clinical_Hypothesis_Normal/'
LUNA16_THER_TEST_CSV_PATH = '/luna16_dataset/luna16_csv_data/Theoretical_Test/'
LUNA16_CLINICAL_TEST_CSV_PATH = '/luna16_dataset/luna16_csv_data/Clinical_Test/'

LUNA16_SPLOT_PATH = '/luna16_dataset/luna16_simulation_plots/'

BF_REAL_DATA_PATH = '/CT_compare_data/RealData_60_reg/nii_before/'
AF_REAL_DATA_PATH = '/CT_compare_data/RealData_60_unreg/nii_after/'
BF_REAL_PROCESSED_DATA_PATH = '/CT_compare_data/RealData_60_npy/before/'
AF_REAL_PROCESSED_DATA_PATH = '/CT_compare_data/RealData_60_npy/after/'
REAL_ANNO_PATH = '/CT_compare_data/Lung_SegAndCls_20230925_gender_age.csv'
REAL_IMAGE_SAVE_PATH = '/CT_compare_data/RealDataImage/'
REAL_CSV_SAVE_PATH = '/CT_compare_data/RealDataCsv/'

EVA_SAVE_PATH = '/CT_compare_data/RealDataResult/'

black_list = ['1.3.6.1.4.1.14519.5.2.1.6279.6001.313334055029671473836954456733']


def load_itk_image_plus(filename):
    with open(filename) as f:
        contents = f.readlines()
        # 读取图像是否反转
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        # 读取质心
        offset = [k for k in contents if k.startswith('Offset')][0]
        # 读取步长
        Spacing = [k for k in contents if k.startswith('ElementSpacing')][0]
        transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform = np.round(transform)
        # 读取x质心坐标
        offset_x = np.array(offset.split(' = ')[1].split(' ')[0]).astype('float')
        offset_x = np.round(offset_x)
        # 读取y质心坐标
        offset_y = np.array(offset.split(' = ')[1].split(' ')[1]).astype('float')
        offset_y = np.round(offset_y)
        # 读取z质心坐标
        offset_z = np.array(offset.split(' = ')[1].split(' ')[2]).astype('float')
        offset_z = np.round(offset_z)
        # 读取x步长
        x_ElementSpacing = np.array(Spacing.split(' = ')[1].split(' ')[0]).astype('float')
        # 读取y步长
        y_ElementSpacing = np.array(Spacing.split(' = ')[1].split(' ')[1]).astype('float')
        # 读取z步长
        z_ElementSpacing = np.array(Spacing.split(' = ')[1].split(' ')[2]).astype('float')
        # 设置反转标志
        if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False
    # 读取图像
    itkimage = sitk.ReadImage(filename)
    numpyimage = sitk.GetArrayFromImage(itkimage)
    # print('max:',np.max(numpyimage))
    # print('min:',np.min(numpyimage))
    # 反转图像
    if (isflip == True):
        numpyimage = numpyimage[:, ::-1, ::-1]
    offset = np.zeros(3)
    Spacing = np.zeros(3)
    offset[0] = offset_x
    offset[1] = offset_y
    offset[2] = offset_z
    Spacing[0] = x_ElementSpacing
    Spacing[1] = y_ElementSpacing
    Spacing[2] = z_ElementSpacing
    return numpyimage, offset, Spacing, isflip


def load_dicom_image(dicom_files_path):
    """
    Load DICOM series and extract relevant information.
    :param dicom_files_path: Path to the DICOM series.
    :return: Tuple containing image array, origin, spacing, and whether a flip is needed.
    """
    # Create an instance of ImageSeriesReader
    reader = sitk.ImageSeriesReader()

    # Get the list of DICOM file names in the specified directory
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_files_path)

    # Set the DICOM file names to the reader
    reader.SetFileNames(dicom_names)

    # Execute the reader to read the DICOM series and create a SimpleITK image
    image = reader.Execute()

    # Get the image array and information
    numpyImage = sitk.GetArrayFromImage(image)
    numpyOrigin = np.array(list(reversed(image.GetOrigin())))
    numpySpacing = np.array(list(reversed(image.GetSpacing())))
    direction = image.GetDirection()

    # Check if a flip is needed based on the direction cosines
    transformM = np.array(direction).reshape(3, 3)
    transformM = np.round(transformM)
    isflip = np.any(transformM != np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    if (isflip == True):
        numpyImage = numpyImage[:, ::-1, ::-1]
    return numpyImage, numpyOrigin, numpySpacing, isflip


def convert_to_date(date_str):
    match = re.search(r'\d{2}-\d{2}-\d{4}', date_str)
    date_format = "%m-%d-%Y"
    return datetime.strptime(match.group(), date_format)


# 坐标系转换函数
# 世界坐标转voxel坐标
def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


# voxel坐标转世界坐标
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def get_segmented_lungs(im):
    binary = im < -400  # Step 1: 转换为二值化图像 <-400
    cleared = clear_border(binary)  # Step 2: 清除图像边界的小块区域
    label_image = label(cleared)  # Step 3: 分割图像

    areas = [r.area for r in regionprops(label_image)]  # Step 4: 保留两个最大的连通区域
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    selem = disk(2)  # Step 5: 图像腐蚀操作，将结节与血管剥离
    binary = binary_erosion(binary, selem)
    selem = disk(10)  # Step 6: 图像闭环操作，保留贴近肺壁的结节
    binary = binary_closing(binary, selem)
    edges = roberts(binary)  # Step 7: 进一步将肺部残余小孔部位填充
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0  # Step 8: 将二值逻辑图像叠加到输入图像上
    im[get_high_vals] = 0
    # print('lung segmentation complete.')
    return im, binary


def get_segmented_lungs_plus(im):
    binary = im < -400  # Step 1: 转换为二值化图像 <-400
    cleared = clear_border(binary)  # Step 2: 清除图像边界的小块区域
    label_image = label(cleared)  # Step 3: 分割图像
    regions = regionprops(label_image)
    areas = [r.area for r in regions]  # Step 4: 保留两个最大的连通区域
    areas.sort()
    region_types = []
    # the max number of connected region is 2
    lung_region_num = 0
    if len(areas) <= 2:
        for region in regions:

            x_coord = region.coords[:, 0]
            y_coord = region.coords[:, 1]
            x_max = np.max(x_coord)
            x_min = np.min(x_coord)
            y_max = np.max(y_coord)
            y_min = np.min(y_coord)
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range < 350 and y_range < 350:
                # right lung region
                region_types.append(1)
                lung_region_num += 1
                # print('region_type now: lung')
            else:
                # noise region
                region_types.append(0)
                # print('region_type now: noise')
        for i, region in enumerate(regions):
            if region_types[i]:
                continue
            else:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    else:

        lung_regions = []
        for region in regions:

            x_coord = region.coords[:, 0]
            y_coord = region.coords[:, 1]
            x_max = np.max(x_coord)
            x_min = np.min(x_coord)
            y_max = np.max(y_coord)
            y_min = np.min(y_coord)
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range < 350 and y_range < 350:
                region_types.append(1)
                lung_regions.append(region)
                lung_region_num += 1
            else:
                # noise region
                region_types.append(0)
        # Eliminate the noise region
        for i, region in enumerate(regions):
            if region_types[i]:
                continue
            else:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
        # The two largest areas of pulmonary connectivity are retained

        if lung_region_num > 2:
            lung_region_areas = [r.area for r in lung_regions]
            lung_region_areas.sort()
            for lung_region in lung_regions:
                if lung_region.area < lung_region_areas[-2]:
                    for coordinates in lung_region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0

    # print('lung_region_num: ', lung_region_num)
    binary = label_image > 0

    selem = disk(2)  # Step 5: 图像腐蚀操作，将结节与血管剥离
    binary = binary_erosion(binary, selem)
    selem = disk(10)  # Step 6: 图像闭环操作，保留贴近肺壁的结节
    binary = binary_closing(binary, selem)
    edges = roberts(binary)  # Step 7: 进一步将肺部残余小孔部位填充
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0  # Step 8: 将二值逻辑图像叠加到输入图像上
    im[get_high_vals] = 0
    # print('lung segmentation complete.')
    return im, binary


def mask_lungs_together(image, xmin, xmax, ymin, ymax):
    # 取出-1200到-250的像素
    mask = (image > -1200) & (image < -250)
    # mask后图像
    processed_image = np.where(mask, image, 0)
    # 使用label精分图像
    labels, nlabels = ndi.label(processed_image)
    coms = ndi.center_of_mass(processed_image, labels, index=np.array(range(1, nlabels, 1)))
    labelIndex = []
    # 判断条件是面积大于1000，小于120000，x在100-400之间，y在50-260、250-500之间
    for i in range(1, nlabels, 1):
        if (ndi.sum(1, labels, index=i) > 1000) & (ndi.sum(1, labels, index=i) < 120000):
            if (coms[i - 1][0] > 100) & (coms[i - 1][0] < 400):
                if ((coms[i - 1][1] > 50) & (coms[i - 1][1] < 260)) | ((coms[i - 1][1] > 250) & (coms[i - 1][1] < 500)):
                    labelIndex.append(i)
    if len(labelIndex) < 1:  # 如果为空，输出空图
        return None
    else:
        # 取出原始图像对应肺叶区域，判断if labels in labelIndex
        # 函数np.inld
        lungs_image = np.where(np.in1d(labels, labelIndex).reshape(512, 512), image, 0)
        lungs = lungs_image[(slice(xmin, xmax, None), slice(ymin, ymax, None))]
        # 同时输出肺叶图像，坐标对应按照传递过来的坐标参数
        return lungs


def mhdraw2npy(filename, savepath):
    numpyimage, offset, spacing, isflip = load_itk_image_plus(filename)
    # lung_mask = np.zeros(numpyimage.shape)
    # 取出肺部mask
    for i in range(numpyimage.shape[0]):
        # lung_mask[i] = mask_lungs_together(numpyimage[i],0,512,0,512)
        get_segmented_lungs(numpyimage[i])
    patient_id = filename.split('/')[-1].replace('.mhd', '')
    # 保存npy信息
    np.save(savepath + patient_id + '_image.npy', numpyimage)
    np.save(savepath + patient_id + '_offset.npy', offset)
    np.save(savepath + patient_id + '_spacing.npy', spacing)
    np.save(savepath + patient_id + '_isflip.npy', isflip)


def nii2npy(filename, savepath):
    # load nii data
    image = sitk.ReadImage(filename)

    numpy_image = sitk.GetArrayFromImage(image)

    numpy_origin = np.array(image.GetOrigin())

    numpy_spacing = np.array(image.GetSpacing())

    isflip = np.all(numpy_spacing < 0)
    for i in range(numpy_image.shape[0]):
        get_segmented_lungs_plus(numpy_image[i])

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    # patient_id = filename.split('/')[-2]
    # draw image sample to check the seg result
    slice_num = numpy_image.shape[0]
    fig_number = 10
    fig, ax = plt.subplots(fig_number, 1, figsize=(20, fig_number * 10))
    for j in range(fig_number):
        ax[j].imshow(numpy_image[int(slice_num / 2) + j, :, :], cmap='gray', vmin=-1200, vmax=600)
    plt.savefig(savepath + '/image_sample.png')
    plt.close(fig)

    print(f'image shape: {numpy_image.shape}')

    np.savez_compressed(savepath + '/image.npz', data=numpy_image)
    np.savez_compressed(savepath + '/offset.npz', data=numpy_origin)
    np.savez_compressed(savepath + '/spacing.npz', data=numpy_spacing)
    np.savez_compressed(savepath + '/isflip.npz', data=isflip)


def generate_virtual_progression_image_plus(patient_id, annotation, compact_change=0.1, diameter_change_rate=0.1,
                                            sigma=10,
                                            luna16_npy_path='/teams/Thymoma_1685081756/luna16_dataset/processed_luna_npy_data/subset0/',
                                            savepath='/teams/Thymoma_1685081756/luna16_dataset/generated_luna_npy_data/subset0/',
                                            show_image=False,
                                            save_image_sample=True,
                                            mode='sort'):
    # 本函数将利用结节像素点个数作为参照，逐渐改变结节附近的肺部组织为结节，直到结节像素点个数是原来的(1+dcr)^2 倍
    image = np.load(luna16_npy_path + patient_id + '_image.npy')
    offset = np.load(luna16_npy_path + patient_id + '_offset.npy')
    spacing = np.load(luna16_npy_path + patient_id + '_spacing.npy')
    isflip = np.load(luna16_npy_path + patient_id + '_isflip.npy')

    cand_anno_coords = annotation[annotation['seriesuid'] == patient_id]
    anno_id = 0
    lung_issue_HU = -800
    nodule_issue_HU = -100
    if len(cand_anno_coords) == 0:
        print('There is no annotation for this patient.')
    for info in cand_anno_coords.values:
        coord_x = info[1]
        coord_y = info[2]
        coord_z = info[3]
        anno_coord = np.array((coord_x, coord_y, coord_z))
        voxel_coord = world_2_voxel(anno_coord, offset, spacing)
        diameter = info[4]

        if isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]
        # 得到当前结节中心的HU值，作为后续生成图像的参照
        # center_voxel_value = image[int(voxel_coord[2]), int(voxel_coord[1]), int(voxel_coord[0])]

        diameter_x = diameter / spacing[0]
        diameter_y = diameter / spacing[1]
        diameter_z = diameter / spacing[2]

        z_start = int(voxel_coord[2] - diameter_z / 2)
        z_end = int(voxel_coord[2] + diameter_z / 2)

        sub_image = image[z_start:z_end, :, :]

        vir_pro_image = copy.deepcopy(sub_image)

        acr_list = np.zeros(vir_pro_image.shape[0])
        if mode == 'sort':
            operate_c = 1.0
        else:
            operate_c = 0.5
        for j in range(vir_pro_image.shape[0]):

            operate_x_start = max(int(voxel_coord[0] - operate_c * diameter_x), 0)
            operate_y_start = max(int(voxel_coord[1] - operate_c * diameter_y), 0)

            operate_x_end = min(int(voxel_coord[0] + operate_c * diameter_x), 511)
            operate_y_end = min(int(voxel_coord[1] + operate_c * diameter_y), 511)

            center_voxel_value = vir_pro_image[j, int(voxel_coord[1]), int(voxel_coord[0])]
            if center_voxel_value < -400:
                center_voxel_value = np.mean(
                    vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end])
                if center_voxel_value < -400:
                    center_voxel_value = nodule_issue_HU

            print('center_voxel_value : ', center_voxel_value)

            lung_pixel_change_num = 0
            # bound the operation region
            operate_region = vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]

            # nodule_mask = np.less(abs(operate_region - center_voxel_value), abs(operate_region - lung_issue_HU)) \
            #               & operate_region != 0
            nodule_mask = (operate_region > -400) & (operate_region < 200) & (operate_region != 0)
            # if nodule has no progression on diameter, do nodule compact change operation and disturb it by normal
            if diameter_change_rate == 0:
                operate_region[nodule_mask] = np.maximum(np.minimum(np.random.normal(operate_region[nodule_mask] +
                                                                                     compact_change, sigma), 500), -800)
                continue

            nodule_pixel_num = np.sum(nodule_mask)
            if nodule_pixel_num == 0:
                print('This CT slice may have no nodule! Please check.')
            print('nodule_pixel_num:', nodule_pixel_num)

            lung_pixel_to_change = int(
                (2 * diameter_change_rate + diameter_change_rate * diameter_change_rate) * nodule_pixel_num)
            print('lung_pixel_to_change:', lung_pixel_to_change)

            # while operate_x_start > max_operate_x_start:
            break_flag = 0
            if mode == 'sort':
                # operate compact change
                if lung_pixel_to_change == 0:
                    continue
                if np.sum(~nodule_mask) < lung_pixel_to_change:
                    print('The annotation of this nodule may be false, or the CT slice may be fuzzy, please check.')
                    continue
                change_flag_matrix = np.full_like(operate_region, False, dtype=bool)
                lung_loc = np.argwhere((~nodule_mask) & (operate_region != 0))
                kernel = np.ones((3, 3))
                kernel[1, 1] = 0
                # utilize 2d conv to extract change trend feature of each pixel in operate region
                change_eval_matrix = convolve2d(operate_region, kernel, mode='same', boundary='fill',
                                                fillvalue=lung_issue_HU)
                change_eval_matrix /= 8.0
                # select the top nodule_pixel_to_change pixels
                max_indices = lung_loc[
                    np.argpartition(change_eval_matrix[(~nodule_mask) & (operate_region != 0)], -lung_pixel_to_change)
                    [-lung_pixel_to_change:]]
                change_flag_matrix[max_indices[:, 0], max_indices[:, 1]] = True
                print('np.sum(change_flag_matrix):', np.sum(change_flag_matrix))

                # operate nodule to lung change
                # operate_region[change_flag_matrix] = np.random.normal(center_voxel_value, abs(0.1 * center_voxel_value))
                comb_coeff = np.random.rand()
                operate_region[change_flag_matrix] = comb_coeff * np.random.normal(center_voxel_value,
                                                                                   abs(0.05 * center_voxel_value)) \
                                                     + (1.0 - comb_coeff) * np.random.uniform(-400, 200)
                nodule_mask[change_flag_matrix] = True
                if compact_change != 0:
                    operate_region[nodule_mask] = np.maximum(np.minimum(np.random.normal(operate_region[nodule_mask] +
                                                                                         compact_change, sigma), 200),
                                                             -800)
                vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region
            if mode == 'traverse':

                while lung_pixel_change_num < lung_pixel_to_change and operate_x_end < vir_pro_image.shape[
                    2] and operate_y_end < vir_pro_image.shape[1]:
                    lung_pixel_change_num_now = 0
                    for k in range(1, operate_region.shape[1] - 1):
                        if lung_pixel_change_num + lung_pixel_change_num_now > lung_pixel_to_change:
                            break
                        for l in range(1, operate_region.shape[0] - 1):
                            # 如果当前像素点本身为结节像素点 或 肺部以外区域 不进行转换
                            if nodule_mask[l, k]:
                                continue
                            if operate_region[l, k] == 0:
                                continue

                            # 检查当前肺部组织像素点周围一圈的像素点，如果有2个及以上结节像素点则进行转换
                            change_count = 0
                            change_flag = False
                            for m in range(-1, 2):
                                for n in range(-1, 2):
                                    if n == 0 or m == 0:
                                        continue
                                    if nodule_mask[l + m, k + n] and operate_region[l + m, k + n] != 0:
                                        change_count = change_count + 1
                                        if change_count > 0:
                                            change_flag = True
                                            break
                                if change_flag:
                                    break

                            if change_flag:
                                # print('vir_pro_image1[j, l, k] to change: ', vir_pro_image1[j, l, k])
                                if np.random.uniform(0, 1) > 0.2:
                                    # print('change one pixel!')
                                    lung_pixel_change_num_now = lung_pixel_change_num_now + 1
                                    nodule_mask[l, k] = True
                                    operate_region[l, k] = np.random.normal(center_voxel_value,
                                                                            abs(0.1 * center_voxel_value))

                            if lung_pixel_change_num + lung_pixel_change_num_now > lung_pixel_to_change:
                                break
                    # update the change to virtual CT image
                    vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region
                    # update operate region
                    operate_x_start = int(operate_x_start - 0.1 * diameter_x)
                    operate_x_end = int(operate_x_end + 0.1 * diameter_x)
                    operate_y_start = int(operate_y_start - 0.1 * diameter_y)
                    operate_y_end = int(operate_y_end + 0.1 * diameter_y)
                    operate_region = vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]

                    # update nodule mask corresponding to operate region
                    nodule_mask = np.less(abs(operate_region - center_voxel_value), abs(operate_region - lung_issue_HU)) \
                                  & operate_region != 0

                    lung_pixel_change_num = lung_pixel_change_num + lung_pixel_change_num_now

                    if lung_pixel_change_num_now == 0:
                        break_flag = break_flag + 1

                    if break_flag > 2:
                        print('There is no lung pixel to change any more.')
                        break

                # nodule compact change operation
                operate_region[nodule_mask] = np.random.normal(
                    operate_region[nodule_mask] + compact_change,
                    0.1 * abs(operate_region[nodule_mask] + compact_change))
                # update the compact change
                vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region

                if nodule_pixel_num > 0:
                    acr_list[j] = float(lung_pixel_change_num) / float(nodule_pixel_num)
                else:
                    acr_list[j] = 0.0

        # save npy and png format generated virtual image

        gene_save_path = savepath + patient_id + '/' + 'dcr_' + str(diameter_change_rate) + '_cc_' \
                         + str(compact_change) + '_sigma_' + str(sigma)
        if not os.path.exists(gene_save_path):
            os.makedirs(gene_save_path)
        if save_image_sample:
            fig_number = min(sub_image.shape[0], 15)
            fig, ax = plt.subplots(fig_number, 2, figsize=(20, 10 * fig_number))
            for j in range(fig_number):
                ax[j, 0].imshow(sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter_x,
                                          voxel_coord[1] - diameter_y),
                                         2 * diameter_x,
                                         2 * diameter_y, linewidth=2.5,
                                         edgecolor='red',
                                         facecolor='none')
                #ax[j, 0].add_patch(rect)
                ax[j, 1].imshow(vir_pro_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter_x,
                                          voxel_coord[1] - diameter_y),
                                         2 * diameter_x,
                                         2 * diameter_y, linewidth=2.5,
                                         edgecolor='red',
                                         facecolor='none')
                #ax[j, 1].add_patch(rect)
            if show_image:
                plt.tight_layout()
                plt.show()
            # if not os.path.exists(gene_save_path + '/generated_image_sample/'):
            #     os.mkdir(gene_save_path + '/generated_image_sample/')
            plt.savefig(gene_save_path + '/' + patient_id +
                        '_anno' + str(anno_id) + '_image.png')
            plt.clf()
            plt.close()

        acr = np.max(acr_list)
        # print('gene_save_path:', gene_save_path)
        np.save(gene_save_path + '/' + patient_id + '_anno' + str(anno_id) + '.npy', vir_pro_image)
        # np.save(gene_save_path + '/' + patient_id + '_anno' + str(anno_id) + '_acr.npy', acr)
        anno_id = anno_id + 1


def generate_virtual_recession_image_plus(patient_id, annotation, compact_change=0.1, diameter_change_rate=-0.2,
                                          sigma=10,
                                          luna16_npy_path='/teams/Thymoma_1685081756/luna16_dataset/processed_luna_npy_data/subset0/',
                                          savepath='/teams/Thymoma_1685081756/luna16_dataset/processed_luna_npy_data/subset0/',
                                          show_image=False,
                                          save_image_sample=True,
                                          mode='sort'):
    image = np.load(luna16_npy_path + patient_id + '_image.npy')
    offset = np.load(luna16_npy_path + patient_id + '_offset.npy')
    spacing = np.load(luna16_npy_path + patient_id + '_spacing.npy')
    isflip = np.load(luna16_npy_path + patient_id + '_isflip.npy')

    cand_anno_coords = annotation[annotation['seriesuid'] == patient_id]
    # HU reference value
    lung_issue_HU = -800
    nodule_issue_HU = 50

    anno_id = 0
    for info in cand_anno_coords.values:
        coord_x = info[1]
        coord_y = info[2]
        coord_z = info[3]
        anno_coord = np.array((coord_x, coord_y, coord_z))
        voxel_coord = world_2_voxel(anno_coord, offset, spacing)

        diameter = info[4]

        if isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]

        diameter_x = diameter / spacing[0]
        diameter_y = diameter / spacing[1]
        diameter_z = diameter / spacing[2]

        z_start = int(voxel_coord[2] - diameter_z / 2)
        z_end = int(voxel_coord[2] + diameter_z / 2)

        sub_image = image[z_start:z_end, :, :]

        vir_pro_image = copy.deepcopy(sub_image)

        acr_list = np.zeros(vir_pro_image.shape[0])
        for j in range(vir_pro_image.shape[0]):

            operate_x_start = int(voxel_coord[0] - 0.7 * diameter_x)
            operate_y_start = int(voxel_coord[1] - 0.7 * diameter_y)

            operate_x_end = int(voxel_coord[0] + 0.7 * diameter_x)
            operate_y_end = int(voxel_coord[1] + 0.7 * diameter_y)

            center_voxel_value = vir_pro_image[j, int(voxel_coord[1]), int(voxel_coord[0])]

            if center_voxel_value < -400:
                center_voxel_value = np.mean(
                    vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end])
                if center_voxel_value < -400:
                    center_voxel_value = nodule_issue_HU

            print('center_voxel_value : ', center_voxel_value)
            nodule_change_num = 0

            operate_region = vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]

            # nodule_mask = np.less(abs(operate_region - center_voxel_value), abs(operate_region - lung_issue_HU)) \
            #               & operate_region != 0
            nodule_mask = (operate_region > -400) & (operate_region < 200) & (operate_region != 0)
            nodule_pixel_num = np.sum(nodule_mask)
            if nodule_pixel_num == 0:
                print('This CT slice may have no nodule! Please check.')
            print('nodule_pixel_num:', nodule_pixel_num)

            nodule_pixel_to_change = int(
                (-2 * diameter_change_rate - diameter_change_rate * diameter_change_rate) * nodule_pixel_num)
            print('nodule_pixel_to_change:', nodule_pixel_to_change)

            if mode == 'sort':
                if nodule_pixel_to_change == 0:
                    continue
                change_flag_matrix = np.full_like(operate_region, False, dtype=bool)
                nodule_loc = np.argwhere(nodule_mask)
                kernel = np.ones((3, 3))
                kernel[1, 1] = 0
                # utilize 2d conv to extract change trend feature of each pixel in operate region
                change_eval_matrix = convolve2d(operate_region, kernel, mode='same', boundary='fill',
                                                fillvalue=lung_issue_HU)
                change_eval_matrix /= 8.0
                # select the top nodule_pixel_to_change pixels
                min_indices = nodule_loc[np.argpartition(change_eval_matrix[nodule_mask], nodule_pixel_to_change)
                [: nodule_pixel_to_change]]
                change_flag_matrix[min_indices[:, 0], min_indices[:, 1]] = True
                # operate compact change
                if compact_change != 0:
                    operate_region[nodule_mask] = np.random.normal(operate_region[nodule_mask] + compact_change,
                                                                   sigma)
                # operate nodule to lung change
                operate_region[change_flag_matrix] = np.random.normal(lung_issue_HU, abs(0.1 * lung_issue_HU))
                vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region
            if mode == 'traverse':
                break_flag = 0
                while nodule_change_num < nodule_pixel_to_change and operate_x_start < operate_x_end \
                        and operate_y_start < operate_y_end:
                    nodule_change_num_now = 0
                    change_flag_matrix = np.full((operate_region.shape[0], operate_region.shape[1]), False, dtype=bool)
                    for k in range(1, operate_region.shape[1] - 1):
                        if nodule_change_num + nodule_change_num_now > nodule_pixel_to_change:
                            break
                        for l in range(1, operate_region.shape[0] - 1):
                            # if the pixel now is lung issue, no need to change
                            if ~nodule_mask[l, k] or operate_region[l, k] == 0:
                                continue

                            change_count = 0
                            change_flag = False
                            for m in range(-1, 2):
                                for n in range(-1, 2):
                                    if m == 0 or n == 0:
                                        continue
                                    if ~nodule_mask[l + m, k + n] and operate_region[l + m, k + n] != 0:
                                        change_count = change_count + 1
                                        change_flag = True
                                        break
                                if change_flag:
                                    break
                            if change_flag:
                                nodule_change_num_now = nodule_change_num_now + 1
                                operate_region[l, k] = np.random.normal(lung_issue_HU, abs(0.1 * lung_issue_HU))
                                nodule_mask[l, k] = False
                            if nodule_change_num + nodule_change_num_now > nodule_pixel_to_change:
                                break

                    nodule_change_num = nodule_change_num + nodule_change_num_now
                    print('nodule_change_num:', nodule_change_num)
                    if nodule_change_num_now == 0:
                        break_flag = break_flag + 1
                        if break_flag > 2:
                            print('Nodule pixel cannot change any longer.')
                            break
                    if nodule_change_num == 0:
                        print('This image possibly has no nodule or the nodule is too small, please check.')
                        break
                    # update change to virtual CT image
                    vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region
                    # update operate region
                    operate_x_start = int(operate_x_start + 0.1 * diameter_x)
                    operate_x_end = int(operate_x_end - 0.1 * diameter_x)
                    operate_y_start = int(operate_y_start + 0.1 * diameter_y)
                    operate_y_end = int(operate_y_end - 0.1 * diameter_y)
                    operate_region = vir_pro_image[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]
                    # update nodule mask corresponding to the operate region
                    nodule_mask = np.less(abs(operate_region - center_voxel_value), abs(operate_region - lung_issue_HU)) \
                                  & operate_region != 0
                if nodule_pixel_num > 0:
                    acr_list[j] = float(nodule_change_num) / float(nodule_pixel_num)
                else:
                    acr_list[j] = 0.0

        gene_save_path = savepath + patient_id + '/' + 'dcr_' + str(diameter_change_rate) + '_cc_' \
                         + str(compact_change) + '_sigma_' + str(sigma)
        if not os.path.exists(gene_save_path):
            os.makedirs(gene_save_path)
        if save_image_sample:
            fig_number = min(sub_image.shape[0], 15)
            fig, ax = plt.subplots(fig_number, 2, figsize=(20, 10 * fig_number))
            for j in range(fig_number):
                ax[j, 0].imshow(sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter_x,
                                          voxel_coord[1] - diameter_y),
                                         2 * diameter_x,
                                         2 * diameter_y, linewidth=2.5,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 0].add_patch(rect)
                ax[j, 1].imshow(vir_pro_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter_x,
                                          voxel_coord[1] - diameter_y),
                                         2 * diameter_x,
                                         2 * diameter_y, linewidth=2.5,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 1].add_patch(rect)
            # if not os.path.exists(gene_save_path + '/generated_image_sample/'):
            #     os.mkdir(gene_save_path + '/generated_image_sample/')
            if show_image:
                plt.tight_layout()
                plt.show()
            plt.savefig(gene_save_path + '/' + patient_id +
                        '_anno' + str(anno_id) + '_image.png')
            plt.clf()
            plt.close()
        acr = -np.max(acr_list)
        np.save(gene_save_path + '/' + patient_id + '_anno' + str(anno_id) + '.npy', vir_pro_image)
        anno_id = anno_id + 1


def generate_virtual_disturb_image_plus(patient_id, annotation,
                                        luna16_npy_path='/teams/Thymoma_1685081756/luna16_dataset/processed_luna_npy_data/subset0/',
                                        savepath='/teams/Thymoma_1685081756/luna16_dataset/generated_luna_npy_data/subset0/',
                                        show_image=False,
                                        save_image_sample=True,
                                        disturb=10):
    image = np.load(luna16_npy_path + patient_id + '_image.npy')
    offset = np.load(luna16_npy_path + patient_id + '_offset.npy')
    spacing = np.load(luna16_npy_path + patient_id + '_spacing.npy')
    isflip = np.load(luna16_npy_path + patient_id + '_isflip.npy')

    cand_anno_coords = annotation[annotation['seriesuid'] == patient_id]

    anno_id = 0
    for info in cand_anno_coords.values:
        coord_x = info[1]
        coord_y = info[2]
        coord_z = info[3]
        anno_coord = np.array((coord_x, coord_y, coord_z))
        voxel_coord = world_2_voxel(anno_coord, offset, spacing)

        diameter = info[4]

        if isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]

        diameter_x = diameter / spacing[0]
        diameter_y = diameter / spacing[1]
        diameter_z = diameter / spacing[2]

        z_start = int(voxel_coord[2] - diameter_z / 2)
        z_end = int(voxel_coord[2] + diameter_z / 2)

        sub_image = image[z_start:z_end, :, :]

        vir_pro_image1 = copy.deepcopy(sub_image)
        vir_pro_image2 = copy.deepcopy(sub_image)
        for j in range(vir_pro_image1.shape[0]):
            operate_x_start = int(voxel_coord[0] - 0.7 * diameter_x)
            operate_y_start = int(voxel_coord[1] - 0.7 * diameter_y)

            operate_x_end = int(voxel_coord[0] + 0.7 * diameter_x)
            operate_y_end = int(voxel_coord[1] + 0.7 * diameter_y)
            # Generate No.1 random disturb image
            operate_region1 = vir_pro_image1[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]

            nodule_mask1 = (operate_region1 > -400) & (operate_region1 < 200) & (operate_region1 != 0)

            operate_region1[nodule_mask1] = np.random.normal(operate_region1[nodule_mask1], disturb)

            vir_pro_image1[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region1
            # Generate No.2 random disturb image
            operate_region2 = vir_pro_image2[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end]

            nodule_mask2 = (operate_region2 > -400) & (operate_region2 < 200) & (operate_region2 != 0)

            operate_region2[nodule_mask2] = np.random.normal(operate_region2[nodule_mask2], disturb)

            vir_pro_image2[j, operate_y_start:operate_y_end, operate_x_start:operate_x_end] = operate_region2

        gene_save_path1 = savepath + 'disturb_' + str(disturb) + '_1/'
        gene_save_path2 = savepath + 'disturb_' + str(disturb) + '_2/'

        if not os.path.exists(gene_save_path1):
            os.makedirs(gene_save_path1)
        if not os.path.exists(gene_save_path2):
            os.mkdir(gene_save_path2)

        np.save(gene_save_path1 + '/' + patient_id + '_anno' + str(anno_id) + '.npy', vir_pro_image1)
        np.save(gene_save_path2 + '/' + patient_id + '_anno' + str(anno_id) + '.npy', vir_pro_image2)

        if save_image_sample:
            fig, ax = plt.subplots(sub_image.shape[0], 2, figsize=(20, 10 * sub_image.shape[0]))
            for j in range(sub_image.shape[0]):
                ax[j, 0].imshow(sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - 1.0 / 2 * diameter_x,
                                          voxel_coord[1] - 1.0 / 2 * diameter_y),
                                         diameter_x,
                                         diameter_y, linewidth=1.0,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 0].add_patch(rect)
                ax[j, 1].imshow(vir_pro_image1[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - 1.0 / 2 * diameter_x,
                                          voxel_coord[1] - 1.0 / 2 * diameter_y),
                                         diameter_x,
                                         diameter_y, linewidth=1.0,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 1].add_patch(rect)
            # if not os.path.exists(gene_save_path1 + '/generated_image_sample/'):
            #     os.mkdir(gene_save_path1 + '/generated_image_sample/')
            if show_image:
                plt.tight_layout()
                plt.show()
            plt.savefig(gene_save_path1 + '/' + patient_id +
                        '_anno' + str(anno_id) + '_image.png')
            plt.clf()
            plt.close()
        anno_id = anno_id + 1


def generate_A_matrix(image, split_size=32, image_size=512):
    # 将M*image_size*image_size的图像分割成M*(（image_size/split_size）^2)*(split_size^2)的矩阵，便于后续的特征分析
    split_list = []
    for m in range(image.shape[0]):
        split_list.append([])
        for i in range(int(image_size / split_size)):
            split_list[m].append([])
            for j in range(int(image_size / split_size)):
                split_list[m][i].append(
                    image[m, i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size])
    A = np.zeros((image.shape[0], int(image_size / split_size) * int(image_size / split_size), split_size * split_size))
    for m in range(image.shape[0]):
        for i in range(int(image_size / split_size)):
            for j in range(int(image_size / split_size)):
                split_index = i * int(image_size / split_size) + j
                for k in range(split_size):
                    for l in range(split_size):
                        pixel_index = k * split_size + l
                        A[m, split_index, pixel_index] = split_list[m][i][j][k][l]
    return split_list, A


def generate_nodule_block_list(image, image_reg, X, Y, split_size=32, image_size=512, show_iamge=False):
    # 从image里面根据结节XY标注坐标提取出所有slice中结节所在小块，最终返回矩阵M*2*(split_size^2)
    nodule_block_list = np.zeros((image.shape[0], 2, split_size * split_size))
    x_start = int(X - split_size / 2)
    x_end = int(X + split_size / 2)
    y_start = int(Y - split_size / 2)
    y_end = int(Y + split_size / 2)
    for m in range(image.shape[0]):
        nodule_block_now1 = image[m][y_start:y_end, x_start:x_end]
        nodule_block_now2 = image_reg[m][y_start:y_end, x_start:x_end]
        # if m==int(image.shape[0]/2): #画图检查结节
        if show_iamge:
            fig, ax = plt.subplots(1, 2)
            fig.set_figwidth(20)
            fig.set_figheight(20)
            ax = ax.flatten()
            ax[0].imshow(nodule_block_now1, cmap='gray')
            ax[1].imshow(nodule_block_now2, cmap='gray')
        for k in range(split_size):
            for l in range(split_size):
                pixel_index = k * split_size + l
                nodule_block_list[m, 0, pixel_index] = nodule_block_now1[k, l]
                nodule_block_list[m, 1, pixel_index] = nodule_block_now2[k, l]
    return nodule_block_list


def HU_ratio_nodule_progression_detection(A1, A2, anno_i=None, anno_j=None, nodule_block_list=None, p=0.02,
                                          split_size=32, image_size=512, detection_threshold=[0.1]):
    M = A1.shape[0]  # slices 数量
    R = len(detection_threshold)
    split_num = int(image_size / split_size)  # 每行（列）有多少小块
    block_num = split_num ** 2  # 小块总数

    change_ratio_matrix = np.zeros([M, block_num])
    detection_matrix = np.zeros([R, M, block_num])
    detection_list = np.zeros([M, R])
    for m in range(M):
        if anno_i and anno_j:
            nodule_block_1 = A1[m, anno_i * split_num + anno_j, :]
            nodule_block_2 = A2[m, anno_i * split_num + anno_j, :]
        else:
            nodule_block_1 = nodule_block_list[m, 0, :]
            nodule_block_2 = nodule_block_list[m, 1, :]
        for i in range(split_num):
            for j in range(split_num):
                block_1d_index = i * split_num + j
                block_1_now = A1[m, block_1d_index, :]
                block_2_now = A2[m, block_1d_index, :]
                # block_1_now = block_2_now  # trick
                ratio_1 = nodule_block_1 / abs(block_1_now + 0.1)  # 算前一套CT中当前小块与结节小块的HU比值
                ratio_2 = nodule_block_2 / abs(block_2_now + 0.1)  # 算后一套CT中当前小块与结节小块的HU比值

                # change = np.abs(np.mean(ratio_1) - np.mean(ratio_2)) / np.abs(np.mean(ratio_1)) # 计算前后两套CT HU比值的相对改变比率
                change = (np.mean(ratio_2) - np.mean(ratio_1)) / np.abs(np.mean(ratio_1))
                change_ratio_matrix[m, block_1d_index] = change
                for r in range(R):
                    # detection_matrix[r, m, block_1d_index] = 1.0 if change>detection_threshold[r] else 0.0 #如果这一改变比率大于threshold，我们认为结节小块相对于这一小块有改变
                    if change > detection_threshold[r]:
                        detection_matrix[r, m, block_1d_index] = 1.0
                    elif change < -detection_threshold[r]:
                        detection_matrix[r, m, block_1d_index] = -1.0
                    else:
                        detection_matrix[r, m, block_1d_index] = 0.0
        for r in range(R):
            detection_list[m, r] = np.sum(detection_matrix[
                                              r, m]) / block_num  # 将结节所在小块与其他小块之间的HU比值变化示性矩阵求和，如果这一数值接近所有小块的数目，我们认为在这张slice上，结节有明显的改变
    return detection_matrix, detection_list

def HU_ratio_nodule_progression_detection_plus(A1, A2, anno_i=None, anno_j=None, nodule_block_list=None, p=0.02,
                                          split_size=32, image_size=512, detection_threshold=[0.1]):
    # log version AUC
    M = A1.shape[0]  # slices 数量
    R = len(detection_threshold)
    split_num = int(image_size / split_size)  # 每行（列）有多少小块
    block_num = split_num ** 2  # 小块总数

    change_ratio_matrix = np.zeros([M, block_num])
    detection_matrix = np.zeros([R, M, block_num])
    detection_list = np.zeros([M, R])
    for m in range(M):
        if anno_i and anno_j:
            nodule_block_1 = A1[m, anno_i * split_num + anno_j, :]
            nodule_block_2 = A2[m, anno_i * split_num + anno_j, :]
        else:
            nodule_block_1 = nodule_block_list[m, 0, :]
            nodule_block_2 = nodule_block_list[m, 1, :]
        for i in range(split_num):
            for j in range(split_num):
                block_1d_index = i * split_num + j
                block_1_now = A1[m, block_1d_index, :]
                block_2_now = A2[m, block_1d_index, :]
                # block_1_now = block_2_now  # trick
                ratio_1 = nodule_block_1 / abs(block_1_now + 0.1)  # 算前一套CT中当前小块与结节小块的HU比值
                ratio_2 = nodule_block_2 / abs(block_2_now + 0.1)  # 算后一套CT中当前小块与结节小块的HU比值

                # change = np.abs(np.mean(ratio_1) - np.mean(ratio_2)) / np.abs(np.mean(ratio_1)) # 计算前后两套CT HU比值的相对改变比率
                change = (np.mean(ratio_2) - np.mean(ratio_1)) / np.abs(np.mean(ratio_1))
                change_ratio_matrix[m, block_1d_index] = change
                for r in range(R):
                    # detection_matrix[r, m, block_1d_index] = 1.0 if change>detection_threshold[r] else 0.0 #如果这一改变比率大于threshold，我们认为结节小块相对于这一小块有改变
                    if change > detection_threshold[r]:
                        detection_matrix[r, m, block_1d_index] = 1.0
                    elif change < -detection_threshold[r]:
                        detection_matrix[r, m, block_1d_index] = -1.0
                    else:
                        detection_matrix[r, m, block_1d_index] = 0.0
        for r in range(R):
            detection_list[m, r] = np.sum(detection_matrix[
                                              r, m]) / block_num  # 将结节所在小块与其他小块之间的HU比值变化示性矩阵求和，如果这一数值接近所有小块的数目，我们认为在这张slice上，结节有明显的改变
    return detection_matrix, detection_list

def visualize_threshold_detection_value(detection_list, detection_threshold=[i / 100.0 for i in range(1, 101, 1)],
                                        Z_compress=False, Z_compress_method='mean', save_image=False, save_path=''):
    # detection_list 每一行对应一个slice结节变化检测结果，每行有R列，代表不同阈值下的结节变化检测结果
    # detection_list(m,r)代表在前后两套CT的第m个slice中，在第r个阈值下，全部小块中相对于结节小块中有明显hu比值变化的小块比例，是否有变化由阈值决定
    M = detection_list.shape[0]  # slice数目
    R = detection_list.shape[1]  # threshold数目
    if Z_compress:
        if Z_compress_method == 'mean':
            detection_list_compressed = np.mean(detection_list, axis=0)

        elif Z_compress_method == 'max':
            detection_list_compressed = np.max(detection_list, axis=0)

        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(20)
        fig.set_figheight(20)
        ax.plot(detection_threshold, detection_list_compressed, linewidth=5.0)
        ax.set_xlabel('threshold \u03BB', size=20)
        ax.set_ylabel('detection curve W(\u03BB)', size=20)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title('detection plot Z compressed by ' + Z_compress_method, size=20)
        if save_image:
            plt.savefig(save_path)
    else:
        fig, ax = plt.subplots(M, 1)
        fig.set_figwidth(20)
        fig.set_figheight(20 * M)
        ax = ax.flatten()
        for m in range(M):
            ax[m].plot(detection_threshold, detection_list[m], linewidth=5.0)
            ax[m].set_xlabel('threshold \u03BB', fontsize=150)
            ax[m].set_ylabel(r'detection curve $\mathcal{W}$' + '(\u03BB)', fontsize=150)
            ax[m].set_xlim(0, 1.0)
            ax[m].set_ylim(-1.0, 1.0)
            ax[m].tick_params(axis='both', labelsize=150)
            # ax[m].set_title('detection curve in slice' + str(m + 1), size=20)
        if save_image:
            plt.savefig(save_path)
        plt.clf()
        plt.close()


def Luna16_simulation(patient_id, annotation, compact_change=0.1, diameter_change_rate=-0.2, disturb=0.1,
                      split_size=32,
                      luna16_npy_path='/teams/Thymoma_1685081756/luna16_dataset/processed_luna_npy_data/subset0/',
                      luna16_gene_path='/teams/Thymoma_1685081756/luna16_dataset/generated_luna_npy_data/subset0/',
                      plot_save_path='/teams/Thymoma_1685081756/luna16_dataset/luna16_simulation_plots/subset0/',
                      save_plot=False,
                      mode='dcrcc'):
    # 读取预先处理和存储好的Luna16 npy数据
    if mode == 'disturb':
        disturb1_image_path = luna16_gene_path + 'disturb_' + str(disturb) + '_1/' + patient_id + '_anno0.npy'
        if not os.path.exists(disturb1_image_path):
            print('Target image has not been generated, now start generating.')
            print('patient_id:', patient_id)
            print('disturb:', disturb)
            generate_virtual_disturb_image_plus(patient_id, annotation,
                                                luna16_npy_path=luna16_npy_path,
                                                savepath=luna16_gene_path,
                                                show_image=False,
                                                save_image_sample=False,
                                                disturb=disturb)
            print('Generation Complete!')
        image = np.load(disturb1_image_path)
    else:
        image = np.load(luna16_npy_path + patient_id + '_image.npy')
    offset = np.load(luna16_npy_path + patient_id + '_offset.npy')
    spacing = np.load(luna16_npy_path + patient_id + '_spacing.npy')
    isflip = np.load(luna16_npy_path + patient_id + '_isflip.npy')
    # 根据id找到病人的标注信息
    cand_anno_coords = annotation[annotation['seriesuid'] == patient_id]
    anno_id = 0
    S_list = []
    ip_list = []
    c_list = []
    acr_list = []
    for info in cand_anno_coords.values:
        # 读取结节坐标
        coord_x = info[1]
        coord_y = info[2]
        coord_z = info[3]
        anno_coord = np.array((coord_x, coord_y, coord_z))
        # 转化为体素坐标
        voxel_coord = world_2_voxel(anno_coord, offset, spacing)
        diameter = info[4]

        if isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]

        diameter_x = diameter / spacing[0]
        diameter_y = diameter / spacing[1]
        diameter_z = diameter / spacing[2]
        if diameter_x > 32 or diameter_y > 32:
            split_size = 64
        # 计算结节Z轴范围
        z_start = int(voxel_coord[2] - diameter_z / 2)
        z_end = int(voxel_coord[2] + diameter_z / 2)
        # 截取图像
        if mode == 'disturb':
            # If in a disturb mode, the image before treatment has been cropped in generation process.
            sub_image = image
        else:
            sub_image = image[z_start:z_end, :, :]
            c = image[int(voxel_coord[2]), int(voxel_coord[1]), int(voxel_coord[0])]
            c_list.append(c)

        # 读取预先生成的虚拟结节进展图像
        if mode == 'disturb':
            vir_image_exist = True
            vir_image_path = luna16_gene_path + 'disturb_' + str(disturb) + '_2/' + patient_id + '_anno0.npy'
        else:
            vir_image_path = luna16_gene_path + patient_id + '/' + 'dcr_' + str(diameter_change_rate) + \
                             '_cc_' + str(compact_change) + '_sigma_' + str(disturb) + '/' + patient_id + '_anno' + str(
                anno_id) + '.npy'
            # print('vir_image_path:', vir_image_path)
            if not os.path.exists(vir_image_path):
                # print('Target image has not been generated, now start generating.')
                # print('patient_id:', patient_id)
                # print('dcr:', diameter_change_rate)
                # print('cc:', compact_change)
                vir_image_exist = False
                if diameter_change_rate >= 0:
                    generate_virtual_progression_image_plus(patient_id, compact_change=compact_change,
                                                            diameter_change_rate=diameter_change_rate,
                                                            sigma=disturb,
                                                            annotation=annotation, luna16_npy_path=luna16_npy_path,
                                                            savepath=luna16_gene_path,
                                                            show_image=False,
                                                            save_image_sample=False,
                                                            mode='sort')
                elif diameter_change_rate < 0:
                    generate_virtual_recession_image_plus(patient_id, compact_change=compact_change,
                                                          diameter_change_rate=diameter_change_rate,
                                                          sigma=disturb,
                                                          annotation=annotation,
                                                          luna16_npy_path=luna16_npy_path,
                                                          savepath=luna16_gene_path,
                                                          show_image=False,
                                                          save_image_sample=False,
                                                          mode='sort')
                print('Generation Complete!')
            else:
                vir_image_exist = True
                # return [], [], [], [], vir_image_exist
                print('This image has already been generated.')

        vir_image = np.load(vir_image_path)
        detection_threshold = [i / 100.0 for i in range(1, 101, 1)]
        _, A1 = generate_A_matrix(sub_image, split_size=split_size)
        _, A2 = generate_A_matrix(vir_image, split_size=split_size)
        nodule_block_list = generate_nodule_block_list(sub_image, vir_image, voxel_coord[0], voxel_coord[1],
                                                       split_size=split_size)
        _, detection_list = HU_ratio_nodule_progression_detection(A1, A2, nodule_block_list=nodule_block_list,
                                                                  split_size=split_size,
                                                                  detection_threshold=detection_threshold)
        dtdf = pd.DataFrame(detection_list)
        dtdf.to_csv(luna16_gene_path + patient_id + '/detection_list.csv', index=False, header=False)
        if save_plot:
            if mode == 'dcrcc':
                plotSavePath = plot_save_path + 'dcr_' + str(diameter_change_rate) + '_cc_' + str(compact_change) + '/'
            else:
                plotSavePath = plot_save_path + 'disturb_' + str(disturb) + '/'
            if not os.path.exists(plotSavePath):
                os.makedirs(plotSavePath)
            plotSavePath = plotSavePath + patient_id + '_anno' + str(anno_id) + '_plot.png'
            visualize_threshold_detection_value(detection_list,
                                                detection_threshold=detection_threshold,
                                                Z_compress=False,
                                                save_image=False,
                                                save_path=plotSavePath)
        S = np.zeros(detection_list.shape[0])
        ip = np.zeros(detection_list.shape[0])
        i = 0
        for detection_value in detection_list:
            S[i] = np.trapz(detection_value, detection_threshold)
            slopes = np.diff(detection_value) / np.diff(detection_threshold)
            max_slope_index = np.argmax(np.abs(slopes))
            ip[i] = detection_threshold[max_slope_index]
            i = i + 1
        # max |S|
        abs_S = np.abs(S)
        max_abs_index = np.argmax(abs_S)
        S = S[max_abs_index]
        '''
        if diameter_change_rate >= 0:
            S = np.max(S)
        else:
            S = np.min(S)
        '''
        ip = np.max(ip)
        S_list.append(S)
        ip_list.append(ip)
        os.remove(vir_image_path)  # avoid OOM
    # del A1, A2, nodule_block_list, sub_image, vir_image, detection_list, S, ip, abs_S, slopes

    return S_list, ip_list, c_list, acr_list, vir_image_exist


def Luna16_data_generate(subsetname, patient_id, annotation, dcr_list, cc_list, disturb_list,
                         split_size=32,
                         luna16_npy_path=ROOT_PATH + LUNA16_PROCESSED_DATA_PATH,
                         luna16_gene_path=ROOT_PATH + LUNA16_GENERATED_DATA_PATH,
                         data_save_path=ROOT_PATH + LUNA16_CSV_PATH,
                         mode='dcrcc'):
    # 此函数用于对luna16中的 单结节 图像进行模拟生长消退，然后绘制分块HU比值变化率曲线，输出csv文件:“subset,patientid,d,c,ddr,cc,S,ip”
    spacing = np.load(luna16_npy_path + subsetname + '/' + patient_id + '_spacing.npy')
    # 根据id找到病人的标注信息
    cand_anno_coords = annotation[annotation['seriesuid'] == patient_id]
    if len(cand_anno_coords) > 1:
        print('This is a multi-nodule CT image!')
        return None
    elif len(cand_anno_coords) == 0:
        print('This patient has no nodule!')
        return None
    info = cand_anno_coords.values[0]
    diameter = info[4]

    diameter_x = diameter / spacing[0]
    diameter_y = diameter / spacing[1]
    diameter_z = diameter / spacing[2]

    if mode == 'dcrcc':
        csv_path = data_save_path + '有变结节测试样本（带有mm直径）_32_sample.csv'
        columns = ['patientid', 'luna16_subset', 'dx', 'dy', 'dz', 'c', 'dcr', 'cc', 'sigma', 'S', 'ip', 'diameter_mm']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(','.join(columns) + '\n')
        for dcr in dcr_list:
            for cc in cc_list:
                for disturb in disturb_list:
                    try:
                        S, ip, c, acr, vir_exist = Luna16_simulation(patient_id, compact_change=cc,
                                                                     diameter_change_rate=dcr,
                                                                     disturb=disturb,
                                                                     annotation=annotation, split_size=split_size,
                                                                     luna16_npy_path=luna16_npy_path + subsetname + '/',
                                                                     luna16_gene_path=luna16_gene_path + subsetname + '/')

                        if vir_exist:
                            print('This virtual image has been tested.')
                            continue
                        else:
                            S = S[0]
                            ip = ip[0]
                            c = c[0]
                            data_row = [patient_id, subsetname, diameter_x, diameter_y, diameter_z, c, dcr, cc, disturb,
                                        S,
                                        ip,
                                        diameter]

                            df_csv = pd.read_csv(csv_path)
                            if ((df_csv['patientid'] == data_row[0]) & (df_csv['dcr'] == data_row[6]) & (
                                    df_csv['cc'] == data_row[7]) & (df_csv['sigma'] == data_row[8])).any():
                                print("Data already exists, continue.")
                                continue
                            else:
                                with open(csv_path, 'a', newline='\n') as f:
                                    csv_writer = csv.writer(f)
                                    csv_writer.writerow(data_row)
                    except:
                        print("Generation Accident Comes Up.")
                        print("pid:", patient_id)
                        print("dcr:", dcr)
                        print("cc:", cc)
                        print("sigma:", disturb)
                        continue
                    # acr_list.append(acr)

    else:
        if not os.path.exists(data_save_path):
            os.mkdir(data_save_path)
        csv_path = data_save_path + 'test_record.csv'
        columns = ['patientid', 'luna16_subset', 'dx', 'dy', 'dz', 'c', 'dcr', 'cc', 'sigma', 'S', 'ip']
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write(','.join(columns) + '\n')
        for disturb in disturb_list:
            S, ip, _, _, _ = Luna16_simulation(patient_id,
                                               annotation=annotation, split_size=split_size,
                                               luna16_npy_path=luna16_npy_path + subsetname + '/',
                                               luna16_gene_path=luna16_gene_path + subsetname + '/',
                                               disturb=disturb,
                                               mode='disturb')

            S = S[0]
            ip = ip[0]

            data_row = [patient_id, subsetname, diameter_x, diameter_y, diameter_z, 0, 0, 0, disturb, S, ip]

            with open(csv_path, 'a', newline='\n') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(data_row)
    # del spacing, cand_anno_coords, info, dx_list, dy_list, dz_list, c_list, dcr_csv_list, cc_csv_list
    # del S_list, ip_list, acr_list, disturb_csv_list, subset_list


def compute_S_by_follow_up_CT_slices(CT1, CT2, anno_center, split_size=32):
    '''
    :param CT1: origin lung CT contains nodule, M*N*N image, M>=1
    :param CT2: reg follow-up CT contains nodule, M*N*N image, M>=1
    :param anno_center: the pixel position of bodule center
    :param split_size: block size
    :return: S-area under detection curve, ip-inflection point of detection curve
    '''
    detection_threshold = [i / 100.0 for i in range(1, 101, 1)]
    _, A1 = generate_A_matrix(CT1, split_size=split_size)
    _, A2 = generate_A_matrix(CT2, split_size=split_size)
    nodule_block_list = generate_nodule_block_list(CT1, CT2, anno_center[0], anno_center[1],
                                                   split_size=split_size)
    _, detection_list = HU_ratio_nodule_progression_detection(A1, A2, nodule_block_list=nodule_block_list,
                                                              split_size=split_size,
                                                              detection_threshold=detection_threshold)
    # visualize_threshold_detection_value(detection_list, detection_threshold = detection_threshold, Z_compress = False)
    S = np.zeros(detection_list.shape[0])
    ip = np.zeros(detection_list.shape[0])

    i = 0
    for detection_value in detection_list:
        S[i] = np.trapz(detection_value, detection_threshold)
        slopes = np.diff(detection_value) / np.diff(detection_threshold)
        max_slope_index = np.argmax(np.abs(slopes))
        ip[i] = detection_threshold[max_slope_index]
        i = i + 1

    return S, ip


def predict_dcr_by_S(CT1, CT2, anno_center, anno_diameter, csv_path=LUNA16_CSV_PATH, pre_mode=1):
    # compute S and ip on each CT slice which contains nodule
    S_list, ip_list = compute_S_by_follow_up_CT_slices(CT1, CT2, anno_center)
    M = S_list.shape[0]
    dcr_pre_vector = np.zeros(M)

    if anno_diameter <= 10:
        dx_group_index = 0
    elif anno_diameter <= 15:
        dx_group_index = 1
    elif anno_diameter <= 20:
        dx_group_index = 2
    else:
        dx_group_index = 3

    # load csv data which contains the posterior distribution of S under different dx and different dcr
    posterior_data = pd.read_csv(csv_path + 'subset0.csv')

    for i in range(1, 10):
        filename = csv_path + f'subset{i}.csv'
        metadata = pd.read_csv(filename)
        posterior_data = pd.concat([posterior_data, metadata])

    posterior_data.reset_index(drop=True, inplace=True)

    # group S by dx
    bins_dx = [0, 10, 15, 20, float('inf')]
    labels_dx = ['dx<10', '10<=dx<=15', '15<dx<=20', 'dx>20']
    posterior_data['dx_group'] = pd.cut(posterior_data['dx'], bins=bins_dx, labels=labels_dx, right=False)

    # screen out S data by dx and group S again by dcr
    dcr_groups_S = posterior_data[posterior_data['dx_group'] == labels_dx[dx_group_index]].groupby('dcr')['S']
    num_dcr = dcr_groups_S.ngroups
    if pre_mode == 1:
        prob_matrix = np.zeros(M, num_dcr)
        for m in range(M):
            i = 0
            for dcr_group, group_S in dcr_groups_S:
                ecdf = ECDF(group_S)
                prob_matrix[m, i] = ecdf(S_list[m])
                dcr_pre_vector[m] += prob_matrix[m, i] * dcr_group
                i += 1

    return dcr_pre_vector


# The function of generate theoretical invariant nodule CT image sample. (normal disturb twice, no diameter and
# compact change.)

def generate_sim_test_nodule_sample(subset_pid_list, dcr_list, cc_list, sigma_list, annotation,
                                    luna16_gene_path=ROOT_PATH + LUNA16_GENERATED_DATA_PATH,
                                    luna16_npy_path=ROOT_PATH + LUNA16_PROCESSED_DATA_PATH,
                                    save_folder_name='test_nodule/',
                                    csv_save_path=ROOT_PATH + LUNA16_THER_CSV_PATH):
    for subset_pid in subset_pid_list:
        pid = subset_pid[1]
        subset_name = subset_pid[0]

        Luna16_data_generate(subset_name, pid, annotation, dcr_list, cc_list, sigma_list,
                             luna16_npy_path=luna16_npy_path,
                             luna16_gene_path=luna16_gene_path + save_folder_name,
                             data_save_path=csv_save_path,
                             mode='dcrcc')


def predict_nodule_progression_by_ther_invariant_hypothesis_not_grouped(csv_file_path, a, b):
    # read the csv
    data = pd.read_csv(csv_file_path)
    total_samples_number = len(data)

    # hypothesis test function
    def judge(S, a, b):
        if a < S < b:
            return 0
        else:
            return 1

    # not grouped prediction
    data['predict_positive'] = data.apply(lambda row: judge(row['S'], a, b), axis=1)
    # the ground truth positive label according to the
    data['ground_positive'] = data.apply(lambda row: 0 if row['dcr'] == 0 and row['cc'] == 0 else 1, axis=1)
    # save result
    new_csv_file_path = csv_file_path.replace('.csv', '_predict_result_not_grouped.csv')
    data.to_csv(new_csv_file_path, index=False)

    accuracy_group = {}
    precision_group = {}
    recall_group = {}
    f1_group = {}

    for group_name, group_data in data.groupby(pd.cut(data['dx'], [0, 10, 15, 20, float('inf')])):
        accuracy = accuracy_score(group_data['ground_positive'], group_data['predict_positive'])
        precision = precision_score(group_data['ground_positive'], group_data['predict_positive'])
        recall = recall_score(group_data['ground_positive'], group_data['predict_positive'])
        f1 = f1_score(group_data['ground_positive'], group_data['predict_positive'])
        sample_count = group_data.shape[0]

        print("Group:", group_name)
        print("Sample Count:", sample_count)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("")
        accuracy_group[group_name] = accuracy
        precision_group[group_name] = precision
        recall_group[group_name] = recall
        f1_group[group_name] = f1

    # calculate the accuracy
    accuracy = accuracy_score(data['ground_positive'], data['predict_positive'])
    precision = precision_score(data['ground_positive'], data['predict_positive'])
    recall = recall_score(data['ground_positive'], data['predict_positive'])
    f1 = f1_score(data['ground_positive'], data['predict_positive'])

    print("Overall Prediction Result:")
    print("Total Samples Number:", total_samples_number)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1


def predict_sim_nodule_progression_by_invariant_hypothesis_grouped(test_csv_file_path, q1q2_invar,
                                                                   group_mode='mm',
                                                                   group_threshold = [],
                                                                   save_csv=False):
    data = pd.read_csv(test_csv_file_path).copy()
    # & ((data['dcr'] >= 0.1) | (data['dcr'] <= -0.4)) PN 8:3
    # & ((data['dcr'] >= 0.45) | (data['dcr'] <= 0.1)) PN 3:7
    data = data[(((data['dcr'] >= 0) & (data['cc'] >= 0)) | ((data['dcr'] <= 0) & (data['cc'] <= 0))) & (
                data['sigma'] <= 100)& ((data['dcr'] >= 0.1) | (data['dcr'] <= -0.4))]
    total_samples_number = len(data)
    pids_num = len(unique(data['patientid']))
    print('Total patient number:', pids_num)
    print('Tatal nodule pair sample number:', total_samples_number)
    def two_tailed_judge(S, a, b):
        if a < S < b:
            return 0
        else:
            return 1

    def one_tailed_judge(S, b):
        if S <= b:
            return 0
        else:
            return 1

    if group_mode == 'pix':
        data['predict_positive'] = data.apply(
            lambda row: one_tailed_judge(row['S'], q1q2_invar[0][1]) if 0 < row['dx'] <= 10
            else (one_tailed_judge(row['S'], q1q2_invar[1][1]) if 10 < row['dx'] <= 15
                  else (one_tailed_judge(row['S'], q1q2_invar[2][1]) if 15 < row['dx'] <= 20
                        else one_tailed_judge(row['S'], q1q2_invar[3][1]))), axis=1)
    else:
        data['predict_positive'] = data.apply(
            lambda row: one_tailed_judge(row['S'], q1q2_invar[0][1]) if group_threshold[0] < row['diameter_mm'] <= group_threshold[1]
            else (one_tailed_judge(row['S'], q1q2_invar[1][1]) if group_threshold[1] < row['diameter_mm'] <= group_threshold[2]
                  else (one_tailed_judge(row['S'], q1q2_invar[2][1]) if group_threshold[2] < row['diameter_mm'] <= group_threshold[3]
                        else one_tailed_judge(row['S'], q1q2_invar[3][1]))), axis=1)


    data['ground_positive'] = data.apply(lambda row: 1 if row['dcr'] > 0.15 or
                                                                  row['cc'] > 150 else 0, axis=1)
    print('ground pos sample number:', len(data[data['ground_positive'] == 1]))
    print('ground neg sample number:', len(data[data['ground_positive'] == 0]))

    # save predict result
    if save_csv:
        new_csv_file_path = test_csv_file_path.replace('.csv',
                                                       group_mode + '_group_predict_result_0913_PN_11_All95_predict.csv')
        data.to_csv(new_csv_file_path, index=False)


    results = []
    # overall
    overall_accuracy = accuracy_score(data['ground_positive'], data['predict_positive'])
    overall_PPV = precision_score(data['ground_positive'], data['predict_positive'])
    overall_NPV = precision_score(data['ground_positive'], data['predict_positive'], pos_label=0)
    overall_sensitivity = recall_score(data['ground_positive'], data['predict_positive'])
    overall_specificity = recall_score(data['ground_positive'], data['predict_positive'], pos_label=0)
    overall_co_ka_coeff = cohen_kappa_score(data['ground_positive'], data['predict_positive'])

    results.append(
        ['all', total_samples_number, overall_accuracy, overall_PPV, overall_NPV, overall_sensitivity,
         overall_specificity, overall_co_ka_coeff])
    for dxgroup_name, dxgroup_data in data.groupby(pd.cut(data['diameter_mm'], group_threshold + [float('inf')])):
        accuracy = accuracy_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'])
        PPV = precision_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'])
        NPV = precision_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'], pos_label=0)
        sensitivity = recall_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'])
        specificity = recall_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'], pos_label=0)
        co_ka_coeff = cohen_kappa_score(dxgroup_data['ground_positive'], dxgroup_data['predict_positive'])
        sample_count = dxgroup_data.shape[0]
        print(f'Group {dxgroup_name}/ Sample Num {sample_count} / accuracy {accuracy} / PPV {PPV} / NPV {NPV} / sensitivity {sensitivity}/ specificity {specificity}')
        results.append(
            [dxgroup_name, sample_count, accuracy, PPV, NPV, sensitivity, specificity, co_ka_coeff])

    if save_csv:
        result_path = test_csv_file_path.replace('.csv',
                                                 group_mode + '_group_accuracy_result_0913_PN_11_All95_predict.csv')
        result_df = pd.DataFrame(results,
                                 columns=['dx_subgroup_name', 'sample_count', 'accuracy',
                                          'PPV',
                                          'NPV',
                                          'sensitivity',
                                          'specificity',
                                          'co_ka_coeff'])
        result_df.to_csv(result_path, index=False)

    print("Overall Prediction Result:")
    print("Total Samples Number:", total_samples_number)
    print("Accuracy:", overall_accuracy)
    print("PPV:", overall_PPV)
    print("NPV:", overall_NPV)
    print("Sensitivity:", overall_sensitivity)
    print("Specificity:", overall_specificity)
    print("co_ka_coeff:", overall_co_ka_coeff)
    print("")

    return None


def predict_real_nodule_progression_by_invariant_hypothesis_grouped(realdata_S_csv_file_path,
                                                                    CT_report_csv_path=ROOT_PATH + REAL_CSV_SAVE_PATH + 'S_method_1_maxabs/realdata_predict_accuracy_result.csv',
                                                                    q1q2_cli_invar=[],
                                                                    save_csv=False):
    data = pd.read_csv(realdata_S_csv_file_path, encoding='gbk').copy()
    report_data = pd.read_csv(CT_report_csv_path, usecols=['patient_id', 'nodule_id', 'positive_from_CT_report'],
                              encoding='gbk').copy()

    total_samples_number = len(data)

    def judge_two_tailed(S, a, b):
        if a < S < b:
            return 0
        else:
            return 1

    def judge_one_tailed(S, b):
        if S < b:
            return 0
        else:
            return 1

    # four group predict

    data['predict_positive_clini_one_tailed'] = data.apply(
        lambda row: judge_one_tailed(row['S'], q1q2_cli_invar[0][1]) if 0 < row['dx'] <= 10
        else (judge_one_tailed(row['S'], q1q2_cli_invar[1][1]) if 10 < row['dx'] <= 15
              else (judge_one_tailed(row['S'], q1q2_cli_invar[1][1]) if 15 < row['dx'] <= 20
                    else judge_one_tailed(row['S'], q1q2_cli_invar[2][1]))), axis=1)

    merged_data = pd.merge(data, report_data, on=['patient_id', 'nodule_id'], how='inner')
    if save_csv:
        new_path = realdata_S_csv_file_path.replace('record', 'predict_result_bayesian_update')
        merged_data.to_csv(new_path, index=False)

    true_pos_sample_number = len(merged_data[merged_data['positive_from_CT_report'] == 1])
    true_neg_sample_number = len(merged_data[merged_data['positive_from_CT_report'] == 0])

    print("Overall Prediction Result:")
    print("Total Samples Number:", total_samples_number)
    print('tp sample_number:', true_pos_sample_number)
    print('tn sample_number:', true_neg_sample_number)

    overall_accuracy = accuracy_score(merged_data['positive_from_CT_report'],
                                      merged_data['predict_positive_clini_one_tailed'])
    overall_precision = precision_score(merged_data['positive_from_CT_report'],
                                      merged_data['predict_positive_clini_one_tailed'])
    overall_sensitivity = recall_score(merged_data['positive_from_CT_report'],
                                      merged_data['predict_positive_clini_one_tailed'])
    overall_specificity = recall_score(merged_data['positive_from_CT_report'],
                                      merged_data['predict_positive_clini_one_tailed'], pos_label=0)
    overall_co_ka_coeff = cohen_kappa_score(merged_data['positive_from_CT_report'],
                                      merged_data['predict_positive_clini_one_tailed'])
    print("Accuracy:", overall_accuracy)
    print("Precision:", overall_precision)
    print("Sensitivity:", overall_sensitivity)
    print("Specificity:", overall_specificity)
    print("co_ka_coeff:", overall_co_ka_coeff)
    print("")


def RealData_S_Compute(pid, real_annotation,
                       bf_npy_path=ROOT_PATH + BF_REAL_PROCESSED_DATA_PATH,
                       af_npy_path=ROOT_PATH + AF_REAL_PROCESSED_DATA_PATH,
                       save_image=True,
                       image_save_path=ROOT_PATH + REAL_IMAGE_SAVE_PATH,
                       csv_save_path=ROOT_PATH + REAL_CSV_SAVE_PATH):
    bf_image = np.load(bf_npy_path + pid + '/image.npy')
    bf_offset = np.load(bf_npy_path + pid + '/offset.npy')
    bf_spacing = np.load(bf_npy_path + pid + '/spacing.npy')
    bf_isflip = np.load(bf_npy_path + pid + '/isflip.npy')

    af_image = np.load(af_npy_path + pid + '/image.npy')
    af_offset = np.load(af_npy_path + pid + '/offset.npy')
    af_spacing = np.load(af_npy_path + pid + '/spacing.npy')
    af_isflip = np.load(af_npy_path + pid + '/isflip.npy')

    print('bf_image.shape:', bf_image.shape)
    print('af_image.shape:', af_image.shape)
    cand_anno_info = real_annotation[real_annotation['ID'] == pid]
    anno_id = 0

    for info in cand_anno_info.values:
        print(info)
        coord_x = info[1]
        coord_y = info[2]
        range_z = info[3]
        range_z = range_z.split('-')
        z_end = af_image.shape[0] - int(range_z[0])
        z_start = af_image.shape[0] - int(range_z[1])
        coord_z = int((z_start + z_end) / 2.0)
        voxel_coord = np.array((coord_x, coord_y, coord_z))
        diameter = info[8] * 10.0 / bf_spacing[0]  # unit:mm
        LUAD_subtype = info[11]
        if diameter > 32:
            split_size = 64
        else:
            split_size = 32
        if bf_isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]
        bf_sub_image = bf_image[z_start:(z_end + 1), :, :]
        af_sub_image = af_image[z_start:(z_end + 1), :, :]
        print("bf_sub_image.shape:", bf_sub_image.shape)

        if save_image:
            fig_number = min(bf_sub_image.shape[0], 10)
            fig, ax = plt.subplots(fig_number, 2, figsize=(20, 10 * fig_number))
            for j in range(fig_number):
                ax[j, 0].imshow(bf_sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - 0.5 * diameter,
                                          voxel_coord[1] - 0.5 * diameter),
                                         diameter,
                                         diameter, linewidth=0.5,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 0].add_patch(rect)
                ax[j, 1].imshow(af_sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - 0.5 * diameter,
                                          voxel_coord[1] - 0.5 * diameter),
                                         diameter,
                                         diameter, linewidth=0.5,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 1].add_patch(rect)
            if not os.path.exists(image_save_path + '/' + pid):
                os.mkdir(image_save_path + '/' + pid)
            plt.savefig(image_save_path + '/' + pid +
                        '/anno' + str(anno_id) + '_image.png')
            plt.clf()
            plt.close()

        detection_threshold = [i / 100.0 for i in range(1, 101, 1)]
        _, A1 = generate_A_matrix(bf_sub_image, split_size=split_size)
        _, A2 = generate_A_matrix(af_sub_image, split_size=split_size)
        nodule_block_list = generate_nodule_block_list(bf_sub_image, af_sub_image, voxel_coord[0], voxel_coord[1],
                                                       split_size=split_size)
        _, detection_list = HU_ratio_nodule_progression_detection(A1, A2, nodule_block_list=nodule_block_list,
                                                                  split_size=split_size,
                                                                  detection_threshold=detection_threshold)
        S = np.zeros(detection_list.shape[0])
        ip = np.zeros(detection_list.shape[0])
        i = 0
        for detection_value in detection_list:
            S[i] = np.trapz(detection_value, detection_threshold)
            slopes = np.diff(detection_value) / np.diff(detection_threshold)
            max_slope_index = np.argmax(np.abs(slopes))
            ip[i] = detection_threshold[max_slope_index]
            i = i + 1
        max_S = np.max(S)
        min_S = np.min(S)
        '''
        S_num_pos = sum([1 for s in S if s > 0])
        S_num_neg = sum([1 for s in S if s <= 0])
        if S_num_pos >= S_num_neg:
            S = max_S
        else:
            S = min_S
        '''

        pos_S_values = [s for s in S if s > 0]
        neg_S_values = [s for s in S if s <= 0]

        if len(pos_S_values) > 0:
            mean_pos_S = mean(pos_S_values)
        else:
            mean_pos_S = 0

        if len(neg_S_values) > 0:
            mean_neg_S = mean(neg_S_values)
        else:
            mean_neg_S = 0

        if abs(mean_pos_S) > abs(mean_neg_S):
            S = max_S
        else:
            S = min_S

        columns = ['patient_id', 'nodule_id', 'dx', 'LUAD_subtype', 'S']
        if not os.path.exists(csv_save_path):
            with open(csv_save_path, 'w') as f:
                f.write(','.join(columns) + '\n')
        data_row = [pid, anno_id, diameter, LUAD_subtype, S]
        with open(csv_save_path, 'a', newline='\n') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data_row)
        anno_id += 1
    return 0


def RealData_S_Compute_0603(pid, real_annotation,
                            bf_npy_path,
                            af_npy_path,
                            bf_date,
                            af_date,
                            csv_save_path,
                            save_image=True):
    bf_image = np.load(bf_npy_path + '/image.npz')
    bf_offset = np.load(bf_npy_path + '/offset.npz')
    bf_spacing = np.load(bf_npy_path + '/spacing.npz')
    bf_isflip = np.load(bf_npy_path + '/isflip.npz')

    af_image = np.load(af_npy_path + '/image.npz')
    af_offset = np.load(af_npy_path + '/offset.npz')
    af_spacing = np.load(af_npy_path + '/spacing.npz')
    af_isflip = np.load(af_npy_path + '/isflip.npz')

    bf_image = bf_image[bf_image.files[0]]
    af_image = af_image[af_image.files[0]]
    af_spacing = af_spacing[af_spacing.files[0]]
    af_isflip = af_isflip[af_isflip.files[0]]
    print('bf_image.shape:', bf_image.shape)
    print('af_image.shape:', af_image.shape)
    m = 0
    for k, real_annotation_row in real_annotation.iterrows():
        coord_x = real_annotation_row['X']
        coord_y = real_annotation_row['Y']
        range_z = real_annotation_row['Z']
        progress = real_annotation_row['是否进展']

        range_z = range_z.split('-')
        z_end = af_image.shape[0] - int(range_z[0])
        z_start = af_image.shape[0] - int(range_z[1])
        coord_z = int((z_start + z_end) / 2.0)
        voxel_coord = np.array((coord_x, coord_y, coord_z))
        diameter = int(int(range_z[1]) - int(range_z[0]))  # unit:mm

        if diameter > 32:
            split_size = 64
        else:
            split_size = 32
        if af_isflip:
            voxel_coord = [512 - voxel_coord[0], 512 - voxel_coord[1], voxel_coord[2]]
        bf_sub_image = bf_image[z_start:(z_end + 1), :, :]
        af_sub_image = af_image[z_start:(z_end + 1), :, :]
        print("bf_sub_image.shape:", bf_sub_image.shape)
        print("af_sub_image.shape:", af_sub_image.shape)
        if save_image:
            fig_number = min(bf_sub_image.shape[0], 10)
            fig, ax = plt.subplots(fig_number, 2, figsize=(20, 10 * fig_number))
            for j in range(fig_number):
                ax[j, 0].imshow(bf_sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter,
                                          voxel_coord[1] - diameter),
                                         2 * diameter,
                                         2 * diameter, linewidth=2.0,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 0].add_patch(rect)
                ax[j, 1].imshow(af_sub_image[j, :, :], cmap='gray', vmin=-1200, vmax=600)
                rect = patches.Rectangle((voxel_coord[0] - diameter,
                                          voxel_coord[1] - diameter),
                                         2 * diameter,
                                         2 * diameter, linewidth=2.0,
                                         edgecolor='red',
                                         facecolor='none')
                ax[j, 1].add_patch(rect)
            plt.savefig(af_npy_path + '/anno' + str(m) + '_image_sample.png')
            plt.clf()
            plt.close()

        detection_threshold = [i / 100.0 for i in range(1, 101, 1)]
        _, A1 = generate_A_matrix(bf_sub_image, split_size=split_size)
        _, A2 = generate_A_matrix(af_sub_image, split_size=split_size)
        nodule_block_list = generate_nodule_block_list(bf_sub_image, af_sub_image, voxel_coord[0], voxel_coord[1],
                                                       split_size=split_size)
        detection_matrix, detection_list = HU_ratio_nodule_progression_detection(A1, A2, nodule_block_list=nodule_block_list,
                                                                  split_size=split_size,
                                                                  detection_threshold=detection_threshold)
        S = np.zeros(detection_list.shape[0])
        ip = np.zeros(detection_list.shape[0])
        i = 0
        for detection_value in detection_list:
            S[i] = np.trapz(detection_value, detection_threshold)
            slopes = np.diff(detection_value) / np.diff(detection_threshold)
            max_slope_index = np.argmax(np.abs(slopes))
            ip[i] = detection_threshold[max_slope_index]
            i = i + 1

        max_S = np.max(S)
        min_S = np.min(S)

        pos_S_values = [s for s in S if s > 0]
        neg_S_values = [s for s in S if s <= 0]

        if len(pos_S_values) > 0:
            mean_pos_S = mean(pos_S_values)
        else:
            mean_pos_S = 0

        if len(neg_S_values) > 0:
            mean_neg_S = mean(neg_S_values)
        else:
            mean_neg_S = 0

        if abs(mean_pos_S) > abs(mean_neg_S):
            S = max_S
        else:
            S = min_S

        columns = ['patient_id', '结节编号', '检查日期', '比较检查日期', '是否进展', 'diameter_z (unit: pixel)', 'diameter_z (unit: mm)', 'S']
        if not os.path.exists(csv_save_path):
            with open(csv_save_path, 'w') as f:
                f.write(','.join(columns) + '\n')
        data_row = [pid, m, af_date, bf_date, progress, diameter, diameter * af_spacing[2], S]
        with open(csv_save_path, 'a', newline='\n') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data_row)
        m += 1

    return 0


def Eva_method_display(result_csv_path, result_save_path=ROOT_PATH + EVA_SAVE_PATH):
    result_data = pd.read_csv(result_csv_path, encoding='gbk')
    pid_list = result_data['patient_id']
    nid_list = result_data['nodule_id']
    ground_positive = result_data['positive_from_CT_report']
    predict_positive = result_data['predict_positive_clini_one_tailed']

    LUAD_subtype = result_data['LUAD_subtype']
    results = []
    clinical_ground_positive = copy.deepcopy(ground_positive)
    clinical_ground_positive[LUAD_subtype == 'IAC'] = 1
    clinical_ground_positive[LUAD_subtype == 'Pre-IA'] = 0
    clinical_ground_positive = clinical_ground_positive[(LUAD_subtype == 'IAC') | (LUAD_subtype == 'Pre-IA')]
    clinical_predict_positive = predict_positive[(LUAD_subtype == 'IAC') | (LUAD_subtype == 'Pre-IA')]

    fp_pid_list = pid_list[(ground_positive == 0) & (predict_positive == 1)]
    fp_iac_pid_list = pid_list[(ground_positive == 0) & (predict_positive == 1) & (LUAD_subtype == 'IAC')]
    fn_pid_list = pid_list[(ground_positive == 1) & (predict_positive == 0)]
    fn_preia_pid_list = pid_list[(ground_positive == 1) & (predict_positive == 0) & (LUAD_subtype == 'Pre-IA')]

    fp_nid_list = nid_list[(ground_positive == 0) & (predict_positive == 1)]
    fp_iac_nid_list = nid_list[(ground_positive == 0) & (predict_positive == 1) & (LUAD_subtype == 'IAC')]
    fn_nid_list = nid_list[(ground_positive == 1) & (predict_positive == 0)]
    fn_preia_nid_list = nid_list[(ground_positive == 1) & (predict_positive == 0) & (LUAD_subtype == 'Pre-IA')]

    print("fp_pid_list:", fp_pid_list)
    print("fp_iac_pid_list:", fp_iac_pid_list)
    print("fn_pid_list:", fn_pid_list)
    print("fn_preia_pid_list:", fn_preia_pid_list)

    print("fp_nid_list:", fp_nid_list)
    print("fp_iac_nid_list:", fp_iac_nid_list)
    print("fn_nid_list:", fn_nid_list)
    print("fn_preia_nid_list:", fn_preia_nid_list)

    dx = result_data['dx']
    ground_positive_iac = ground_positive[LUAD_subtype == 'IAC']
    predict_positive_iac = predict_positive[LUAD_subtype == 'IAC']

    ground_positive_preia = ground_positive[LUAD_subtype == 'Pre-IA']
    predict_positive_preia = predict_positive[LUAD_subtype == 'Pre-IA']

    con_mat = confusion_matrix(ground_positive, predict_positive)
    con_mat_clinical = confusion_matrix(clinical_ground_positive, clinical_predict_positive)
    con_mat_iac = confusion_matrix(ground_positive_iac, predict_positive_iac)
    con_mat_preia = confusion_matrix(ground_positive_preia, predict_positive_preia)
    con_mat_dx1 = confusion_matrix(ground_positive[dx <= 10], predict_positive[dx <= 10])
    con_mat_dx2 = confusion_matrix(ground_positive[(dx > 10) & (dx <= 15)], predict_positive[(dx > 10) & (dx <= 15)])
    con_mat_dx3 = confusion_matrix(ground_positive[(dx > 15) & (dx <= 20)], predict_positive[(dx > 15) & (dx <= 20)])
    con_mat_dx4 = confusion_matrix(ground_positive[dx > 20], predict_positive[dx > 20])
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
    # con_mat_norm = np.around(con_mat_norm, decimals=2)

    # === plot ===
    plt.figure(figsize=(2, 2))
    sns.heatmap(con_mat, annot=True, cmap='Blues')
    plt.title('All Test Sample')
    plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # === plot ===
    plt.figure(figsize=(2, 2))
    sns.heatmap(con_mat_clinical, annot=True, cmap='Blues')
    plt.title('IAC=1 Pre-IA=0')
    plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # === plot ===
    plt.figure(figsize=(2, 2))
    sns.heatmap(con_mat_iac, annot=True, cmap='Blues')
    plt.title('IAC test sample')
    plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # === plot ===
    plt.figure(figsize=(2, 2))
    sns.heatmap(con_mat_preia, annot=True, cmap='Blues')
    plt.title('Pre-IA test sample')
    plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    accuracy = accuracy_score(ground_positive, predict_positive)
    precision = precision_score(ground_positive, predict_positive)
    sensitivity = recall_score(ground_positive, predict_positive)
    specificity = recall_score(ground_positive, predict_positive, pos_label=0)
    co_ka_coeff = cohen_kappa_score(ground_positive, predict_positive)
    results.append(["all", len(ground_positive), accuracy, precision, sensitivity, specificity, co_ka_coeff])
    print("Overall Accuracy:", accuracy)
    print("Overall Precision:", precision)
    print("Overall Sensitivity:", sensitivity)
    print("Overall Specificity:", specificity)
    print("Overall Cohen's Kappa coefficient:", co_ka_coeff)

    clinical_accuracy = accuracy_score(clinical_ground_positive, clinical_predict_positive)
    clinical_precision = precision_score(clinical_ground_positive, clinical_predict_positive)
    clinical_sensitivity = recall_score(clinical_ground_positive, clinical_predict_positive)
    clinical_specificity = recall_score(clinical_ground_positive, clinical_predict_positive, pos_label=0)
    clinical_co_ka_coeff = cohen_kappa_score(clinical_ground_positive, clinical_predict_positive)

    print("IAC Pre-IA Accuracy:", clinical_accuracy)
    print("IAC Pre-IA Precision:", clinical_precision)
    print("IAC Pre-IA Sensitivity:", clinical_sensitivity)
    print("IAC Pre-IA Specificity:", clinical_specificity)
    print("IAC Pre-IA Cohen's Kappa coefficient:", clinical_co_ka_coeff)

    accuracy_dx1 = accuracy_score(ground_positive[dx <= 10], predict_positive[dx <= 10])
    precision_dx1 = precision_score(ground_positive[dx <= 10], predict_positive[dx <= 10])
    sensitivity_dx1 = recall_score(ground_positive[dx <= 10], predict_positive[dx <= 10])
    specificity_dx1 = recall_score(ground_positive[dx <= 10], predict_positive[dx <= 10], pos_label=0)
    co_ka_coeff_dx1 = cohen_kappa_score(ground_positive[dx <= 10], predict_positive[dx <= 10])
    results.append(
        ["(0.0, 10.0]", len(ground_positive[dx <= 10]), accuracy_dx1, precision_dx1, sensitivity_dx1, specificity_dx1,
         co_ka_coeff_dx1])
    print("Diameter Group 1 Num:", sum(dx <= 10))
    print("Diameter Group 1 Accuracy:", accuracy_dx1)
    print("Diameter Group 1 Precision:", precision_dx1)
    print("Diameter Group 1 Sensitivity:", sensitivity_dx1)
    print("Diameter Group 1 Specificity:", specificity_dx1)
    print("Diameter Group 1 Cohen's Kappa coefficient:", co_ka_coeff_dx1)

    accuracy_dx2 = accuracy_score(ground_positive[(dx > 10) & (dx <= 15)], predict_positive[(dx > 10) & (dx <= 15)])
    precision_dx2 = precision_score(ground_positive[(dx > 10) & (dx <= 15)], predict_positive[(dx > 10) & (dx <= 15)])
    sensitivity_dx2 = recall_score(ground_positive[(dx > 10) & (dx <= 15)], predict_positive[(dx > 10) & (dx <= 15)])
    specificity_dx2 = recall_score(ground_positive[(dx > 10) & (dx <= 15)], predict_positive[(dx > 10) & (dx <= 15)],
                                   pos_label=0)
    co_ka_coeff_dx2 = cohen_kappa_score(ground_positive[(dx > 10) & (dx <= 15)],
                                        predict_positive[(dx > 10) & (dx <= 15)])
    results.append(
        ["(10.0, 15.0]", len(ground_positive[(dx > 10) & (dx <= 15)]), accuracy_dx2, precision_dx2, sensitivity_dx2,
         specificity_dx2, co_ka_coeff_dx2])
    print("Diameter Group 2 Num:", sum((dx > 10) & (dx <= 15)))
    print("Diameter Group 2 Accuracy:", accuracy_dx2)
    print("Diameter Group 2 Precision:", precision_dx2)
    print("Diameter Group 2 Sensitivity:", sensitivity_dx2)
    print("Diameter Group 2 Specificity:", specificity_dx2)
    print("Diameter Group 2 Cohen's Kappa coefficient:", co_ka_coeff_dx2)

    accuracy_dx3 = accuracy_score(ground_positive[(dx > 15) & (dx <= 20)], predict_positive[(dx > 15) & (dx <= 20)])
    precision_dx3 = precision_score(ground_positive[(dx > 15) & (dx <= 20)], predict_positive[(dx > 15) & (dx <= 20)])
    sensitivity_dx3 = recall_score(ground_positive[(dx > 15) & (dx <= 20)], predict_positive[(dx > 15) & (dx <= 20)])
    specificity_dx3 = recall_score(ground_positive[(dx > 15) & (dx <= 20)], predict_positive[(dx > 15) & (dx <= 20)],
                                   pos_label=0)
    co_ka_coeff_dx3 = cohen_kappa_score(ground_positive[(dx > 15) & (dx <= 20)],
                                        predict_positive[(dx > 15) & (dx <= 20)])
    results.append(
        ["(15.0, 20.0]", len(ground_positive[(dx > 15) & (dx <= 20)]), accuracy_dx3, precision_dx3, sensitivity_dx3,
         specificity_dx3, co_ka_coeff_dx3])
    print("Diameter Group 3 Num:", sum((dx > 15) & (dx <= 20)))
    print("Diameter Group 3 Accuracy:", accuracy_dx3)
    print("Diameter Group 3 Precision:", precision_dx3)
    print("Diameter Group 3 Sensitivity:", sensitivity_dx3)
    print("Diameter Group 3 Specificity:", specificity_dx3)
    print("Diameter Group 3 Cohen's Kappa coefficient:", co_ka_coeff_dx3)

    accuracy_dx4 = accuracy_score(ground_positive[dx > 20], predict_positive[dx > 20])
    precision_dx4 = precision_score(ground_positive[dx > 20], predict_positive[dx > 20])
    sensitivity_dx4 = recall_score(ground_positive[dx > 20], predict_positive[dx > 20])
    specificity_dx4 = recall_score(ground_positive[dx > 20], predict_positive[dx > 20],
                                   pos_label=0)
    co_ka_coeff_dx4 = cohen_kappa_score(ground_positive[dx > 20],
                                        predict_positive[dx > 20])
    results.append(
        ["(20.0, inf]", len(ground_positive[dx > 20]), accuracy_dx4, precision_dx4, sensitivity_dx4, specificity_dx4,
         co_ka_coeff_dx4])
    print("Diameter Group 4 Num:", sum(dx > 20))
    print("Diameter Group 4 Accuracy:", accuracy_dx4)
    print("Diameter Group 4 Precision:", precision_dx4)
    print("Diameter Group 4 Sensitivity:", sensitivity_dx4)
    print("Diameter Group 4 Specificity:", specificity_dx4)
    print("Diameter Group 4 Cohen's Kappa coefficient:", co_ka_coeff_dx4)

    result_df = pd.DataFrame(results, columns=['dx_subgroup_name', 'sample_count', 'accuracy',
                                               'precision',
                                               'sensitivity',
                                               'specificity',
                                               'co_ka_coeff'])
    result_path = result_csv_path.replace('result', 'accuracy_result_0517')
    result_df.to_csv(result_path, index=False)


# Auto Registration by itk-elastix
def reg_elastix(FixedImagePath, MovingImagePath, OutputImagefilename):
    OutputPath = os.path.dirname(OutputImagefilename)

    fixed_image = itk.imread(FixedImagePath, itk.F)
    moving_image = itk.imread(MovingImagePath, itk.F)

    if not os.path.exists(OutputPath):
        os.mkdir(OutputPath)

    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)
    # parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    # parameter_object.AddParameterMap(parameter_map_affine)
    # parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
    # parameter_object.AddParameterMap(parameter_map_bspline)
    parameter_object.SetParameter("AutomaticTransformInitialization", ["true"])
    parameter_object.SetParameter("UseCUDA", ["true"])
    # print(parameter_object)

    result_registered_image, _ = itk.elastix_registration_method(fixed_image, moving_image,
                                                                 parameter_object=parameter_object)
    itk.imwrite(result_registered_image, OutputImagefilename, compression=True)