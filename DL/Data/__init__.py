import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import random
from utils.util import worker_init_fn
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


class NPDSDataset(Dataset):
    def __init__(self, images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates):
        self.images0 = images0
        self.images1 = images1
        self.coords = coords
        self.progs = progs
        self.pids = pids
        self.anno_ids = anno_ids
        self.before_dates = before_dates
        self.after_dates = after_dates
        self.block_size = 40

    def __len__(self):
        return len(self.images0)

    def __getitem__(self, idx):
        image0 = self.images0[idx]
        image1 = self.images1[idx]
        coord = self.coords[idx]
        prog = self.progs[idx]
        pid = self.pids[idx]
        anno_id = self.anno_ids[idx]
        before_date = self.before_dates[idx]
        after_date = self.after_dates[idx]

        image0 = torch.tensor(image0, dtype=torch.float32)
        image1 = torch.tensor(image1, dtype=torch.float32)
        coord = torch.tensor(coord, dtype=torch.float32)
        prog = torch.tensor(prog, dtype=torch.float32)

        coord_x = int(coord[0].item())
        coord_y = int(coord[1].item())

        H, W = image0.shape[1], image0.shape[2]
        half_block = self.block_size // 2

        pad_left = max(0, half_block - coord_x)
        pad_right = max(0, coord_x + half_block - W)
        pad_top = max(0, half_block - coord_y)
        pad_bottom = max(0, coord_y + half_block - H)

        image0_pad = F.pad(image0, (pad_left, pad_right, pad_top, pad_bottom))  # (left, right, top, bottom)
        image1_pad = F.pad(image1, (pad_left, pad_right, pad_top, pad_bottom))

        coord_x_pad = coord_x + pad_left
        coord_y_pad = coord_y + pad_top

        image0_n = image0_pad[:, coord_y_pad - half_block:coord_y_pad + half_block,
                   coord_x_pad - half_block:coord_x_pad + half_block]
        image1_n = image1_pad[:, coord_y_pad - half_block:coord_y_pad + half_block,
                   coord_x_pad - half_block:coord_x_pad + half_block]
        # image0_n = image0[:, (coord_y - self.block_size // 2):(coord_y + self.block_size // 2), (coord_x - self.block_size // 2):(coord_x + self.block_size // 2)]
        # image1_n = image1[:, (coord_y - self.block_size // 2):(coord_y + self.block_size // 2), (coord_x - self.block_size // 2):(coord_x + self.block_size // 2)]

        return image0, image1, image0_n, image1_n, prog, pid, anno_id, before_date, after_date


def show_all_images(image_data):
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()
    num_images = image_data.shape[0]
    rows, cols = 2, 5
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image_data[i], cmap="gray")
        plt.axis("off")
        plt.title(f"Img {i}")

    plt.tight_layout()
    plt.show()


def load_process_images_coords_progress(image_folders, anno_csv_paths, real_result_csv_paths):
    images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates = [], [], [], [], [], [], [], []

    ids_part1 = os.listdir(image_folders[0] + 'before/')

    annos_df_part1 = pd.read_csv(anno_csv_paths[0], encoding='gbk').copy()
    annos_df_part1['ID'] = annos_df_part1['ID'].astype(str)
    '''
    progress_df_part1 = pd.read_excel(progress_csv_path_part1).copy()
    progress_df_part1['patient_id'] = progress_df_part1['patient_id'].astype(str)
    '''
    real_result_df_part1 = pd.read_excel(real_result_csv_paths[0]).copy()
    real_result_df_part1['patient_id'] = real_result_df_part1['patient_id'].astype(str)

    sample_num = 0
    for pid in ids_part1:
        cand_anno_info = annos_df_part1[annos_df_part1['ID'] == pid]
        if cand_anno_info.empty:
            print(f'Patient {pid} in RelData60 has not anno info, continue.')
            continue
        image0 = np.load(image_folders[0] + 'before/' + pid + '/image.npy')
        image1 = np.load(image_folders[0] + 'after/' + pid + '/image.npy')

        resized_image0 = np.zeros((image0.shape[0], 256, 256), dtype=np.float32)
        for j in range(image0.shape[0]):
            resized_image0[j] = cv2.resize(image0[j].astype(np.float32), (256, 256))
        resized_image0 = resized_image0.astype(np.float16)
        resized_image1 = np.zeros((image1.shape[0], 256, 256), dtype=np.float32)
        for j in range(image1.shape[0]):
            resized_image1[j] = cv2.resize(image1[j].astype(np.float32), (256, 256))
        resized_image1 = resized_image1.astype(np.float16)
        af_isflip = np.load(image_folders[0] + 'after/' + pid + '/isflip.npy')

        anno_id = 0
        for info in cand_anno_info.values:
            coord_x = info[1]
            coord_y = info[2]
            range_z = info[3]
            range_z = range_z.split('-')
            z_end = image1.shape[0] - int(range_z[0])
            z_start = image1.shape[0] - int(range_z[1])
            coord_z = int((z_start + z_end) / 2.0)
            if af_isflip:
                coord_x = 512 - coord_x
                coord_y = 512 - coord_y
            coord_x = int(coord_x / 2)
            coord_y = int(coord_y / 2)
            nodule_coord = np.array((coord_x, coord_y), dtype=np.float32)
            '''
            progress_row = progress_df_part1[(progress_df_part1['patient_id'] == pid) & (progress_df_part1['nodule_id'] == anno_id)]
            if progress_row.empty:
                continue
            else:
                progress_row = progress_row.iloc[0]
            '''
            real_result_row = real_result_df_part1[(real_result_df_part1['patient_id'] == pid) & (real_result_df_part1['nodule_id'] == anno_id)]
            if real_result_row.empty:
                print(f'Patient {pid} annot_id {anno_id} in RealData60 is not included in real result.')
                anno_id += 1
                continue
            else:
                real_result_row = real_result_row.iloc[0]
            progress = real_result_row['是否进展']

            '''
            half_block = 20
            show_all_images(resized_image0[(coord_z - 5):(coord_z + 5), coord_y - half_block:coord_y + half_block,
                   coord_x - half_block:coord_x + half_block])
            show_all_images(resized_image1[(coord_z - 5):(coord_z + 5), coord_y - half_block:coord_y + half_block,
                   coord_x - half_block:coord_x + half_block])
            '''

            images0.append(resized_image0[(coord_z - 5):(coord_z + 5), :, :])
            images1.append(resized_image1[(coord_z - 5):(coord_z + 5), :, :])
            coords.append(nodule_coord)
            progs.append(progress)
            pids.append(pid)
            before_dates.append('')
            after_dates.append('')
            anno_ids.append(anno_id)
            anno_id += 1
            sample_num += 1
            print(f'sample_num:{sample_num}')

    annos_df = pd.read_excel(anno_csv_paths[1]).copy()

    real_result_df_part2 = pd.read_excel(real_result_csv_paths[1]).copy()
    real_result_df_part2['patient_id'] = real_result_df_part2['patient_id'].astype(str).str.zfill(10)
    real_result_df_part2['检查日期'] = real_result_df_part2['检查日期'].astype(str)
    real_result_df_part2['比较检查日期'] = real_result_df_part2['比较检查日期'].astype(str)
    for image_folder in image_folders[1:]:
        ids = os.listdir(image_folder)
        for id in ids:
            dates = os.listdir(image_folder + id + '/')
            bf_dates = [date for date in dates if 'regto' in date]
            af_dates = [bd.split('regto')[1] for bd in bf_dates]
            if len(bf_dates) == 0:
                print(f'Patient {id} in {image_folder}, before img npy folder is empty, continue.')
                continue
            for i, bf_date in enumerate(bf_dates):
                af_date = af_dates[i]

                af_id_date = '-'.join([id, af_date])
                anno_info_rows = annos_df[annos_df['ID'] == af_id_date]
                if anno_info_rows.empty:
                    print(f'Patient {id} in {image_folder} has not anno info, continue.')
                    continue

                bfDate = bf_date.split('regto')[0]
                print(f'bf_date now:{bf_date}')
                image0 = np.load(image_folder + id + '/' + bf_date + '/image.npz')
                image0 = image0[image0.files[0]]
                print(f'af_date now:{af_date}')
                image1 = np.load(image_folder + id + '/' + af_date + '/image.npz')
                image1 = image1[image1.files[0]]

                resized_image0 = np.zeros((image0.shape[0], 256, 256), dtype=np.float32)
                for j in range(image0.shape[0]):
                    resized_image0[j] = cv2.resize(image0[j].astype(np.float32), (256, 256))
                resized_image0 = resized_image0.astype(np.float16)
                resized_image1 = np.zeros((image1.shape[0], 256, 256), dtype=np.float32)
                for j in range(image1.shape[0]):
                    resized_image1[j] = cv2.resize(image1[j].astype(np.float32), (256, 256))
                resized_image1 = resized_image1.astype(np.float16)
                af_isflip = np.load(image_folder + id + '/' + af_date + '/isflip.npz')
                af_isflip = af_isflip[af_isflip.files[0]]

                anno_id = 0
                for k, anno_info_row in anno_info_rows.iterrows():
                    real_result_row = real_result_df_part2[(real_result_df_part2['patient_id']==id) & (real_result_df_part2['结节编号']==anno_id) & (real_result_df_part2['检查日期']==af_date) & (real_result_df_part2['比较检查日期']==bfDate)]
                    if real_result_row.empty:
                        print(f'Patient {id} annot_id {anno_id} bf_date {bfDate} af_date {af_date} in part2 is not included in real result.')
                        anno_id += 1
                        continue
                    coord_x = anno_info_row['X']
                    coord_y = anno_info_row['Y']
                    range_z = anno_info_row['Z']
                    range_z = range_z.split('-')
                    z_end = image1.shape[0] - int(range_z[0])
                    z_start = image1.shape[0] - int(range_z[1])
                    coord_z = int((z_start + z_end) / 2.0)
                    if af_isflip:
                        coord_x = 512 - coord_x
                        coord_y = 512 - coord_y
                    coord_x = int(coord_x / 2)
                    coord_y = int(coord_y / 2)
                    nodule_coord = np.array((coord_x, coord_y), dtype=np.float32)

                    progress = anno_info_row['是否进展']

                    images0.append(resized_image0[(coord_z - 5):(coord_z + 5), :, :])
                    images1.append(resized_image1[(coord_z - 5):(coord_z + 5), :, :])
                    coords.append(nodule_coord)
                    progs.append(progress)
                    pids.append(id)
                    before_dates.append(bfDate)
                    after_dates.append(af_date)
                    anno_ids.append(anno_id)

                    anno_id += 1
                    sample_num += 1
                    print(f'sample_num:{sample_num}')
    print('Job Done.')
    # return np.array(images0, dtype=np.float32), np.array(images1, dtype=np.float32), np.array(coords, dtype=np.float32), np.array(progs, dtype=np.float32)
    return images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates


def load_process_images_coords_progress_result_domain(image_folders, anno_csv_paths, real_result_csv_paths):
    images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates = [], [], [], [], [], [], [], []

    annos_df_part1 = pd.read_csv(anno_csv_paths[0], encoding='gbk').copy()
    annos_df_part1['ID'] = annos_df_part1['ID'].astype(str)

    real_result_df_part1 = pd.read_excel(real_result_csv_paths[0]).copy()
    real_result_df_part1['patient_id'] = real_result_df_part1['patient_id'].astype(str)

    sample_num = 0
    for _, real_result_row in real_result_df_part1.iterrows():
        pid = real_result_row['patient_id']
        anno_id = real_result_row['nodule_id']
        progress = real_result_row['是否进展']
        cand_anno_info = annos_df_part1[(annos_df_part1['ID'] == pid)]
        image0_path = image_folders[0] + 'before/' + pid + '/image.npy'
        image1_path = image_folders[0] + 'before/' + pid + '/image.npy'
        if cand_anno_info.empty:
            print(f'Patient {pid} in RelData60 has no anno info, continue.')
            continue
        else:
            info = cand_anno_info.iloc[anno_id]
        if not os.path.exists(image0_path) or not os.path.exists(image1_path):
            print(f'Patient {pid} in RelData60 has no image data, continue.')
            continue
        image0 = np.load(image0_path)
        image1 = np.load(image1_path)

        resized_image0 = np.zeros((image0.shape[0], 256, 256), dtype=np.float32)
        for j in range(image0.shape[0]):
            resized_image0[j] = cv2.resize(image0[j].astype(np.float32), (256, 256))
        resized_image0 = resized_image0.astype(np.float16)
        resized_image1 = np.zeros((image1.shape[0], 256, 256), dtype=np.float32)
        for j in range(image1.shape[0]):
            resized_image1[j] = cv2.resize(image1[j].astype(np.float32), (256, 256))
        resized_image1 = resized_image1.astype(np.float16)
        af_isflip = np.load(image_folders[0] + 'after/' + pid + '/isflip.npy')

        coord_x = info[1]
        coord_y = info[2]
        range_z = info[3]
        range_z = range_z.split('-')
        z_end = image1.shape[0] - int(range_z[0])
        z_start = image1.shape[0] - int(range_z[1])
        coord_z = int((z_start + z_end) / 2.0)
        if af_isflip:
            coord_x = 512 - coord_x
            coord_y = 512 - coord_y
        coord_x = int(coord_x / 2)
        coord_y = int(coord_y / 2)
        nodule_coord = np.array((coord_x, coord_y), dtype=np.float32)

        images0.append(resized_image0[(coord_z - 5):(coord_z + 5), :, :])
        images1.append(resized_image1[(coord_z - 5):(coord_z + 5), :, :])
        coords.append(nodule_coord)
        progs.append(progress)
        pids.append(pid)
        before_dates.append('')
        after_dates.append('')
        anno_ids.append(anno_id)
        sample_num += 1
        print(f'sample_num:{sample_num}')

    annos_df_part2 = pd.read_excel(anno_csv_paths[1]).copy()

    real_result_df_part2 = pd.read_excel(real_result_csv_paths[1]).copy()
    real_result_df_part2['patient_id'] = real_result_df_part2['patient_id'].astype(str).str.zfill(10)
    real_result_df_part2['检查日期'] = real_result_df_part2['检查日期'].astype(str)
    real_result_df_part2['比较检查日期'] = real_result_df_part2['比较检查日期'].astype(str)

    for _, real_result_row in real_result_df_part2.iterrows():
        id = real_result_row['patient_id']
        anno_id = real_result_row['结节编号']
        progress = real_result_row['是否进展']
        af_date = real_result_row['检查日期']
        bf_date = real_result_row['比较检查日期']
        id_afdate = '-'.join([id, af_date])
        cand_anno_info = annos_df_part2[(annos_df_part2['ID'] == id_afdate)]
        if cand_anno_info.empty:
            print(f'Patient {id_afdate} in Part2 has no anno info, continue.')
            continue
        else:
            info = cand_anno_info.iloc[anno_id]
        image_exist = False
        image_exist_folder = ''
        for image_folder in image_folders[1:]:
            image0_path = image_folder + id + '/' + bf_date + 'regto' + af_date + '/image.npz'
            image1_path = image_folder + id + '/' + af_date + '/image.npz'
            if os.path.exists(image0_path) and os.path.exists(image1_path):
                image_exist = True
                image_exist_folder = image_folder
                break
        if not image_exist:
            print(f'Patient {id} in Part2 has no anno info, continue.')
            continue
        print(f'bf_date now:{bf_date}')
        image0 = np.load(image_exist_folder + id + '/' + bf_date + 'regto' + af_date + '/image.npz')
        image0 = image0[image0.files[0]]
        print(f'af_date now:{af_date}')
        image1 = np.load(image_exist_folder + id + '/' + af_date + '/image.npz')
        image1 = image1[image1.files[0]]

        resized_image0 = np.zeros((image0.shape[0], 256, 256), dtype=np.float32)
        for j in range(image0.shape[0]):
            resized_image0[j] = cv2.resize(image0[j].astype(np.float32), (256, 256))
        resized_image0 = resized_image0.astype(np.float16)
        resized_image1 = np.zeros((image1.shape[0], 256, 256), dtype=np.float32)
        for j in range(image1.shape[0]):
            resized_image1[j] = cv2.resize(image1[j].astype(np.float32), (256, 256))
        resized_image1 = resized_image1.astype(np.float16)
        af_isflip = np.load(image_exist_folder + id + '/' + af_date + '/isflip.npz')
        af_isflip = af_isflip[af_isflip.files[0]]

        coord_x = info['X']
        coord_y = info['Y']
        range_z = info['Z']
        range_z = range_z.split('-')
        z_end = image1.shape[0] - int(range_z[0])
        z_start = image1.shape[0] - int(range_z[1])
        coord_z = int((z_start + z_end) / 2.0)
        if af_isflip:
            coord_x = 512 - coord_x
            coord_y = 512 - coord_y
        coord_x = int(coord_x / 2)
        coord_y = int(coord_y / 2)
        nodule_coord = np.array((coord_x, coord_y), dtype=np.float32)

        images0.append(resized_image0[(coord_z - 5):(coord_z + 5), :, :])
        images1.append(resized_image1[(coord_z - 5):(coord_z + 5), :, :])
        coords.append(nodule_coord)
        progs.append(progress)
        pids.append(id)
        before_dates.append(bf_date)
        after_dates.append(af_date)
        anno_ids.append(anno_id)
        sample_num += 1
        print(f'sample_num:{sample_num}')

    print('Job Done.')
    return images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates


