"""
Filename: NPDSlib.py
Description: Python codes to realize the NPDS method proposed in "A Novel Statistic Guided by Clinical Experience for Assessing the Progression of Lung Nodules".

Author: Hang Yu
Institution: School of Statistics, Renmin University of China
Email: hangyustat@ruc.edu.cn
Date: 2024-09-23
"""

from numpy import *
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
import itk
from plotnine import *
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
from IPython.display import display, Image as IPImage

class NPDS_Hypothesis_Testing:
    def __init__(self, coord_x, coord_y, range_z, diameter, baseline_nii_path, followup_nii_path):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.range_z = range_z.split('-')

        self.bf_CT_nii = itk.imread(baseline_nii_path, itk.F)
        self.af_CT_nii = itk.imread(followup_nii_path, itk.F)
        self.bf_CT_npy = itk.GetArrayViewFromImage(self.bf_CT_nii)
        self.af_CT_npy = itk.GetArrayViewFromImage(self.af_CT_nii)
        self.af_spacing = np.array(self.af_CT_nii.GetSpacing())
        self.diameter_pixel = diameter / self.af_spacing[0]
        self.diameter_mm = diameter
        self.isflip = np.all(self.af_spacing < 0)
        self.ClinvNod_NPDS_95th_percentiles = [0.0011799599609374932, 0.005169005859374974, 0.0505342207031249, 0.10536974414062492]

        self.z_end = self.af_CT_npy.shape[0] - int(self.range_z[0])
        self.z_start = self.af_CT_npy.shape[0] - int(self.range_z[1])
        self.coord_z = int((self.z_start + self.z_end) / 2.0)
        self.voxel_coord = np.array((self.coord_x, self.coord_y, self.coord_z))
        self.diameter_z = int(int(self.range_z[1]) - int(self.range_z[0]))
        if self.diameter_z > 32 or self.diameter_pixel > 32:
            self.split_size = 64
        else:
            self.split_size = 32
        if self.isflip:
            self.voxel_coord = [512 - self.voxel_coord[0], 512 - self.voxel_coord[1], self.voxel_coord[2]]
        self.af_sub_image = self.af_CT_npy[self.z_start:(self.z_end + 1), :, :]
        self.image_size = self.af_sub_image.shape[1]
        print(f'baseline CT size : {self.bf_CT_npy.shape}')
        print(f'follow-up CT size : {self.af_CT_npy.shape}')
        print('Initialization complete.')

    def registration_by_elastix(self):
        parameter_object = itk.ParameterObject.New()
        parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
        parameter_object.AddParameterMap(parameter_map_rigid)
        parameter_object.SetParameter("AutomaticTransformInitialization", ["true"])
        parameter_object.SetParameter("UseCUDA", ["true"])
        print('Running registration algorithm...')
        result_registered_image, _ = itk.elastix_registration_method(self.af_CT_nii, self.bf_CT_nii,
                                                                     parameter_object=parameter_object)
        print('Registration complete.')
        self.bf_CT_nii = result_registered_image
        self.bf_CT_npy = itk.GetArrayViewFromImage(self.bf_CT_nii)
        self.bf_sub_image = self.bf_CT_npy[self.z_start:(self.z_end + 1), :, :]

        print(f'baseline CT size : {self.bf_CT_npy.shape}')
        print(f'follow-up CT size : {self.af_CT_npy.shape}')

    def get_segmented_lungs_in_CT_slice(self, im):
        binary = im < -400  # Step 1: Transform CT slice to binary image
        cleared = clear_border(binary)  # Step 2: Remove small areas near the edges of the image
        label_image = label(cleared)  # Step 3: Segment the image to regions
        regions = regionprops(label_image)
        areas = [r.area for r in regions]
        areas.sort()
        region_types = []
        # Step 4: Retain the one or two largest connected regions if len(areas) <= 2; else, retain the largest one
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
                else:
                    # noise region
                    region_types.append(0)
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
        binary = label_image > 0

        selem = disk(2)  # Step 5: Perform image erosion to separate the nodule from the blood vessels.
        binary = binary_erosion(binary, selem)
        selem = disk(10)  # Step 6: Perform a closing operation on the image to retain nodules that are close to the lung wall.
        binary = binary_closing(binary, selem)
        edges = roberts(binary)  # Step 7: Further fill in small cavities in the lungs.
        binary = ndi.binary_fill_holes(edges)
        get_high_vals = binary == 0  # Step 8: Overlay the binary mask onto the output image.
        im[get_high_vals] = 0
        return im, binary

    def get_segmented_lungs(self):
        print("Running lung mask extraction ...")
        for i in range(self.bf_sub_image.shape[0]):
            self.get_segmented_lungs_in_CT_slice(self.bf_sub_image[i])
        for i in range(self.bf_sub_image.shape[0]):
            self.get_segmented_lungs_in_CT_slice(self.af_sub_image[i])
        print("Lung mask extraction complete.")

    def generate_lung_tissue_blocks(self, image, split_size=32, image_size=512):
        split_list = []
        for m in range(image.shape[0]):
            split_list.append([])
            for i in range(int(image_size / split_size)):
                split_list[m].append([])
                for j in range(int(image_size / split_size)):
                    split_list[m][i].append(
                        image[m, i * split_size:(i + 1) * split_size, j * split_size:(j + 1) * split_size])
        A = np.zeros(
            (image.shape[0], int(image_size / split_size) * int(image_size / split_size), split_size * split_size))
        for m in range(image.shape[0]):
            for i in range(int(image_size / split_size)):
                for j in range(int(image_size / split_size)):
                    split_index = i * int(image_size / split_size) + j
                    for k in range(split_size):
                        for l in range(split_size):
                            pixel_index = k * split_size + l
                            A[m, split_index, pixel_index] = split_list[m][i][j][k][l]
        return split_list, A

    def generate_nodule_block_list(self, image, image_reg, X, Y, split_size=32):
        nodule_block_list = np.zeros((image.shape[0], 2, split_size * split_size))
        x_start = int(X - split_size / 2)
        x_end = int(X + split_size / 2)
        y_start = int(Y - split_size / 2)
        y_end = int(Y + split_size / 2)
        for m in range(image.shape[0]):
            nodule_block_now1 = image[m][y_start:y_end, x_start:x_end]
            nodule_block_now2 = image_reg[m][y_start:y_end, x_start:x_end]
            for k in range(split_size):
                for l in range(split_size):
                    pixel_index = k * split_size + l
                    nodule_block_list[m, 0, pixel_index] = nodule_block_now1[k, l]
                    nodule_block_list[m, 1, pixel_index] = nodule_block_now2[k, l]
        return nodule_block_list

    def HU_ratio_nodule_progression_detection(self, A1, A2, anno_i=None, anno_j=None, nodule_block_list=None,
                                                   split_size=32, image_size=512, detection_threshold=[0.1]):
        M = A1.shape[0]  # slices num
        R = len(detection_threshold)
        split_num = int(image_size / split_size)  # block num in a single row
        block_num = split_num ** 2  # total block num

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
                    ratio_1 = nodule_block_1 / abs(block_1_now + 0.1)
                    ratio_2 = nodule_block_2 / abs(block_2_now + 0.1)
                    change = (np.mean(ratio_2) - np.mean(ratio_1)) / np.abs(np.mean(ratio_1))
                    change_ratio_matrix[m, block_1d_index] = change
                    for r in range(R):
                        if change > detection_threshold[r]:
                            detection_matrix[r, m, block_1d_index] = 1.0
                        elif change < -detection_threshold[r]:
                            detection_matrix[r, m, block_1d_index] = -1.0
                        else:
                            detection_matrix[r, m, block_1d_index] = 0.0
            for r in range(R):
                detection_list[m, r] = np.sum(detection_matrix[
                                                  r, m]) / block_num
        return detection_matrix, detection_list

    def NPDS_calculate(self):
        detection_lambda = [i / 100.0 for i in range(1, 101, 1)]
        _, A1 = self.generate_lung_tissue_blocks(self.bf_sub_image, split_size=self.split_size, image_size=self.image_size)
        _, A2 = self.generate_lung_tissue_blocks(self.af_sub_image, split_size=self.split_size, image_size=self.image_size)
        nodule_block_list = self.generate_nodule_block_list(self.bf_sub_image, self.af_sub_image, self.voxel_coord[0], self.voxel_coord[1],
                                                       split_size=self.split_size)
        discrete_stat_matrix, NPDSt_lambda_list = self.HU_ratio_nodule_progression_detection(A1, A2,
                                                                                 nodule_block_list=nodule_block_list,
                                                                                 split_size=self.split_size,
                                                                                 detection_threshold=detection_lambda)
        NPDSt = np.zeros(NPDSt_lambda_list.shape[0])
        i = 0
        for detection_value in NPDSt_lambda_list:
            NPDSt[i] = np.trapz(detection_value, detection_lambda)
            i = i + 1

        max_NPDSt = np.max(NPDSt)
        min_NPDSt = np.min(NPDSt)

        pos_NPDSt_values = [s for s in NPDSt if s > 0]
        neg_NPDSt_values = [s for s in NPDSt if s <= 0]

        if len(pos_NPDSt_values) > 0:
            mean_pos_NPDSt = mean(pos_NPDSt_values)
        else:
            mean_pos_NPDSt = 0

        if len(neg_NPDSt_values) > 0:
            mean_neg_NPDSt = mean(neg_NPDSt_values)
        else:
            mean_neg_NPDSt = 0

        if abs(mean_pos_NPDSt) > abs(mean_neg_NPDSt):
            self.NPDS = max_NPDSt
        else:
            self.NPDS = min_NPDSt

    def hypothesis_test_by_ClinvNod_sample(self):
        if self.diameter_mm <= 5:
            group_reference_num = 1
            self.progress = self.NPDS > self.ClinvNod_NPDS_95th_percentiles[0]
        elif self.diameter_mm <= 10:
            group_reference_num = 2
            self.progress = self.NPDS > self.ClinvNod_NPDS_95th_percentiles[1]
        elif self.diameter_mm <= 15:
            group_reference_num = 3
            self.progress = self.NPDS > self.ClinvNod_NPDS_95th_percentiles[2]
        else:
            group_reference_num = 4
            self.progress = self.NPDS > self.ClinvNod_NPDS_95th_percentiles[3]

        self.ClinvNod_NPDSs = pd.read_csv('./Data/ClinvSample_NPDS_G' + str(group_reference_num) + '.csv').copy()['S']
        ClinvNod_sig = self.ClinvNod_NPDSs.apply(lambda x: 1 if x > self.NPDS else 0)
        self.p_value = ClinvNod_sig.sum() / len(ClinvNod_sig)
        print(f' NPDS : {self.NPDS:.10f}\n Progression Prediction Result : {self.progress}\n p_value : {self.p_value:.10f}')

    def visualize_ct_slices(self, ct_array, figsize=(6, 6), cmap='gray'):
        num_slices = ct_array.shape[0]

        def plot_slice(slice_idx):
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(ct_array[slice_idx, :, :], cmap=cmap)
            ax.set_title(f"CT Slice {slice_idx + 1}/{num_slices}")
            ax.axis('off')
            rect = patches.Rectangle((self.voxel_coord[0] - self.diameter_pixel,
                                      self.voxel_coord[1] - self.diameter_pixel),
                                     2 * self.diameter_pixel,
                                     2 * self.diameter_pixel, linewidth=1.0,
                                     edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)
            plt.show()
        interact(plot_slice, slice_idx=widgets.IntSlider(min=0, max=num_slices - 1, step=1, value=0))

    def visualize_ct_slices_as_gif(self, ct_array, gif_path='./Data/CT_slice.gif', figsize=(6, 6), cmap='gray', duration=300):
        num_slices = ct_array.shape[0]
        images = []
        for slice_idx in range(num_slices):
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(ct_array[slice_idx, :, :], cmap=cmap)
            # ax.set_title(f"CT Slice {slice_idx + 1}/{num_slices}")
            ax.axis('off')
            rect = patches.Rectangle((self.voxel_coord[0] - self.diameter_pixel,
                                      self.voxel_coord[1] - self.diameter_pixel),
                                     2 * self.diameter_pixel,
                                     2 * self.diameter_pixel, linewidth=1.0,
                                     edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)

            # Save figure to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Convert the buffer to an image
            img = PILImage.open(buf)
            images.append(img)

            plt.close(fig)  # Close the figure to free memory

        # Save all images as a GIF
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

        # Display the GIF in Jupyter Notebook
        with open(gif_path, 'rb') as f:
            display(IPImage(data=f.read()))

    def visualize_HT_result(self):
        p = (ggplot(self.ClinvNod_NPDSs.to_frame(), aes(x='S'))
             + geom_histogram(aes(y='..density..'), binwidth=0.01, fill='skyblue', color='black', alpha=0.7)
             + geom_vline(xintercept=self.NPDS, color='red', linetype='dashed', size=1)
             + labs(title='NPDS distribution of ClinvNod sample & the tested NPDS line',
                    x='NPDS Values', y='Density')
             + theme_minimal()
             )
        print(p)

