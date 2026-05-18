import numpy as np
import math
import cv2
import random
import torch.distributed as dist
import os
import glob
import pandas as pd
import torch
from torch import nn
import torchvision.transforms.functional as F
from torchvision import transforms
import torchvision


def get_all_input(base_dir, file_ends='.png'):
    """
    :param base_dir:
    :return:
    """
    g = os.walk(base_dir)
    img_list = []
    for path, _, filelist in g:
        for filename in filelist:
            im_name = os.path.join(path, filename)
            if im_name.endswith(file_ends):
                img_list.append(im_name)
    img_list.sort()
    return img_list


def get_data_list(dataset_top_path, data_path_dict, get_data_list_method, logging=None):
    """
    z00467161, 2025-02-17
    config datapath in yml
    sub dir format: "ISO_2400_realRAW_label_with_xymap_f15_256": {"DataType": "h5", "DataProportion": 6.8}
    """

    train_data_list = []
    for k, info in data_path_dict.items():
        datatype = info["DataType"]
        value = info["DataProportion"]
        if value == 0:
            continue

        if datatype is None:
            t = [k]
        else:
            if "glob" == get_data_list_method:
                t = glob.glob(f"{dataset_top_path}/{k}/**/*.{datatype}", recursive=True)
            else:
                t = []
                with open(f"{dataset_top_path}/{k}/filelist.txt", "r") as txt_list:
                    for line in txt_list:
                        t.append(f"{dataset_top_path}/{k}/{line.strip()}")

        data_num = len(t)
        t = t * math.ceil(value)
        use_num = int(data_num * value)
        train_data_list.extend(t[:use_num])

        assert use_num > 0, f">>>>>>>>>>>>>>>> NO data in {dataset_top_path}/{k} !!!!"
        if logging is not None:
            logging.info("%s : %d * %.2f = %d", k, data_num, value, use_num)

    return train_data_list.copy()



def get_data_list_from_cfg_csv_new(data_cfg_csv, local_path = '', Train_mode = 0, logging=None):
    try:
        df = pd.read_csv(os.path.join(local_path, data_cfg_csv))
    except ValueError:
        print("there is no dataset cfg csv files, please cheack!!!")
    if (logging==None):
        print(f"\n===> Load Dataset Files" )
    else:
        logging.info(f"\n===> Load Dataset Files" )
    filenames = []
    if Train_mode == 0:
        for idx, _data in df.iterrows():
            if 'gen_color_checker' in _data['DataCategoryLocal']:
                _temp_filenames = ['gen_color_checker'] * int(_data['DataProportion'])
            elif 'color_checker_hr' in _data['DataCategoryLocal']:
                _temp_filenames = ['color_checker_hr'] * int(_data['DataProportion'])
            else:
                path = os.path.join(_data['DatasetVersionLocal'], _data['DataCategoryLocal'] )
                _temp_filenames =  get_all_input(path, _data['DataType']) # * _data['DataProportion']

                _need_len = np.ceil(_data['DataProportion'] * len(_temp_filenames)).astype(np.int32)
                if _data['DataProportion'] > 1:
                    _temp_filenames *= np.ceil(_data['DataProportion']).astype(np.int32)
                np.random.shuffle(_temp_filenames)
                _temp_filenames = _temp_filenames[:_need_len]

            filenames += _temp_filenames
            if logging==None:
                print(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}" )
            else:
                logging.info(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}")

    elif Train_mode == 2:
        for idx, _data in df.iterrows():
            if 'gen_color_checker' in _data['DataCategoryLocal']:
                _temp_filenames = ['gen_color_checker'] * int(_data['DataProportion'])
            else:
                _temp_filenames =  get_all_input(os.path.join(_data['DatasetVersionFlexml'], _data['DataCategoryLocal'] ), \
                    _data['DataType'])  # * _data['DataProportion']

                _need_len = np.ceil(_data['DataProportion'] * len(_temp_filenames)).astype(np.int32)
                if _data['DataProportion'] > 1:
                    _temp_filenames *= np.ceil(_data['DataProportion']).astype(np.int32)
                np.random.shuffle(_temp_filenames)
                _temp_filenames = _temp_filenames[:_need_len]

            filenames += _temp_filenames
            if 0 == get_dist_info() and logging==None:
                print(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}" )
            elif 0 == get_dist_info():
                logging.info(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}")

    elif Train_mode==1 or Train_mode==3:
        for idx, _data in df.iterrows():
            if 'gen_color_checker' in _data['DataCategoryLocal']:
                _temp_filenames = ['gen_color_checker'] * int(_data['DataProportion'])
            else:
                _temp_filenames =  [l.strip() for l in open(local_path + _data['DataCategoryRoma'])]  # * _data['DataProportion']

                _need_len = np.ceil(_data['DataProportion'] * len(_temp_filenames)).astype(np.int32)
                if _data['DataProportion'] > 1:
                    _temp_filenames *= np.ceil(_data['DataProportion']).astype(np.int32)
                np.random.shuffle(_temp_filenames)
                _temp_filenames = _temp_filenames[:_need_len]

            filenames += _temp_filenames
            if 0 == get_dist_info() and logging==None:
                print(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}" )
            elif 0 == get_dist_info():
                logging.info(f"{_data['DataCategoryLocal']}: {len(_temp_filenames)}")
    else:
        raise KeyError("Do not support Train mode: {}".format(Train_mode))
    return filenames


def image_mirror_for_pattern(image, crop_size):
    ## image FCHW
    dim = len(image.shape)
    if dim == 3:
        image = image.unsqueeze(0)

    if image.shape[2] < crop_size:
        image = torch.cat([image, image.flip(dims=[2])], axis = 2)
    if image.shape[3] < crop_size:
        image = torch.cat([image, image.flip(dims=[3])], axis = 3)
    if image.shape[2] < crop_size or image.shape[3] < crop_size:
        image = image_mirror_for_pattern(image, crop_size)

    if dim == 3:
        return image[:, :, :crop_size, :crop_size].squeeze(0)
    else:
        return image[:, :, :crop_size, :crop_size]


def image_shift_for_pattern_raw(image, crop_size):
    ## image FCHW
    dim = len(image.shape)
    if dim == 3:
        image = image.unsqueeze(0)

    if image.shape[2] < crop_size:
        img2 = image.clone()
        image = torch.cat([image, img2], axis = 2)
    if image.shape[3] < crop_size:
        img2 = image.clone()
        image = torch.cat([image, img2], axis = 3)
    if image.shape[2] < crop_size or image.shape[3] < crop_size:
        image = image_shift_for_pattern_raw(image, crop_size)

    if dim == 3:
        return image[:, :, :crop_size, :crop_size].squeeze(0)
    else:
        return image[:, :, :crop_size, :crop_size]


class SuitPatchPrepare(nn.Module):
    def __init__(self,
                 need_frame_num = 5,
                 crop_patch_size = 256,
                 crop_type = 'random',          # random, left_up, centre
                 proc_patch_level = 'single',   # single, all
                 this_patch_num = 5,
                 random_step = 4,               # 1,  2 , 4
                 enlarge_type = 'shift',
                 ) -> None:
        super().__init__()
        self.num_frames          = need_frame_num
        self.proc_patch_level   = proc_patch_level
        if self.proc_patch_level == 'single':
            if self.num_frames < this_patch_num:
                start_frame = np.random.randint(0, this_patch_num - self.num_frames)
            else:
                start_frame = 0
            self.frame_list = [x + start_frame for x in range(self.num_frames)]
        elif self.proc_patch_level == 'all':
            self.frame_list = self.get_patch_frame_idx(this_patch_num)

        self.crop_patch_size    = crop_patch_size
        self.crop_type          = crop_type
        self.random_step        = random_step
        self.enlarge_type       = enlarge_type

    def get_patch_frame_idx(self, this_frame):
        if this_frame > self.num_frames :
            start_frame = np.random.randint(0, this_frame - self.num_frames)
            return  [x + start_frame for x in range(self.num_frames)]
        
        order_list = [x for x in range(this_frame)]
        inv_list = [this_frame - 1 - x for x in range(this_frame)]

        if len(order_list) > 1:
            temp = order_list.copy()
            cout = 0
            while len(temp) < self.num_frames:
                cout += 1
                if cout % 2 == 0:
                    temp += order_list[1:]
                else:
                    temp += inv_list[1:]
            return temp[:self.num_frames]
        else:
            return [0 for x in range(self.num_frames)]

    def get_crop_patch_suit_size_all(self, patches, grid=None, crop_size = 128, crop_type = 'random'):
        start_h, start_w = 0, 0
        if self.fix_idx:
            start_h, start_w = self.start_h_last, self.start_w_last
        else:
            if patches.shape[-2] > crop_size:
                if crop_type == 'random':
                    start_h = np.random.randint(0, patches.shape[-2] - crop_size) // self.random_step * self.random_step
                elif crop_type == 'centre':
                    start_h = (patches.shape[-2] - crop_size) // 2
                elif crop_type == 'left_up':
                    start_h = 0
            if patches.shape[-1] > crop_size:
                if crop_type == 'random':
                    start_w = np.random.randint(0, patches.shape[-1] - crop_size) // self.random_step * self.random_step
                elif crop_type == 'centre':
                    start_w = (patches.shape[-1] - crop_size) // 2
                elif crop_type == 'left_up':
                    start_w = 0
            self.start_h_last = start_h
            self.start_w_last = start_w

        output = patches[..., :, start_h:start_h + crop_size,
                                start_w:start_w + crop_size]

        if grid is None:
            return output
        else:
            outgrid = grid[..., :, start_h:start_h + crop_size,
                                start_w:start_w + crop_size]
            #outgrid[..., ::2, :, :] = outgrid[..., ::2, :, :] - start_w
            #outgrid[..., 1::2, :, :] = outgrid[..., 1::2, :, :] - start_h
            return output, outgrid

    def get_crop_patch_suit_size_single(self, patches, grid=None, crop_size = 128, crop_type = 'random'):
        # output = torch.zeros([self.])
        if crop_type == 'random':
            output = []
            for _ in self.frame_list:
                output.append(self.get_crop_patch_suit_size_all(patches, grid = None, crop_size=crop_size, crop_type=crop_type))
            output = torch.cat(output, dim=0)
        else:
            _patches = patches[[0 for x in range(self.num_frames)], ...]
            # _grid = grid[[0 for x in range(self.num_frames)], ...] if grid is not None else None
            output = self.get_crop_patch_suit_size_all(_patches, grid = None, crop_size=crop_size, crop_type=crop_type)
        return output

    def get_enlarged_patch(self, patches, grid=None, size = 128):
        enlarge_func = image_shift_for_pattern_raw
        if self.enlarge_type == 'mirror':
            enlarge_func = image_mirror_for_pattern

        patches = enlarge_func(patches, size)
        if grid is not None:
            grid = enlarge_func(grid, size)
        
        return patches, grid

    @torch.no_grad()
    def forward(self, patches: torch.Tensor, grid = None, fix_idx = False) -> torch.Tensor:
        if patches.shape[-1] < self.crop_patch_size or patches.shape[-2] < self.crop_patch_size:
            patches, grid = self.get_enlarged_patch(patches, grid, size=self.crop_patch_size)

        self.fix_idx = fix_idx
        if self.proc_patch_level == 'single':
            patches_ok = self.get_crop_patch_suit_size_single(patches, grid = None, crop_size=self.crop_patch_size, crop_type=self.crop_type)
        elif self.proc_patch_level == 'all':
            patches_all = patches[self.frame_list, ...]
            grid = grid[self.frame_list, ...] if grid is not None else None
            patches_ok = self.get_crop_patch_suit_size_all(patches_all, grid, crop_size=self.crop_patch_size, crop_type=self.crop_type)

        if isinstance(patches_ok, tuple):
            return patches_ok[0], patches_ok[1]
        else:
            return patches_ok, None


class SuitPatchPrepareInterMotion(nn.Module):
    def __init__(self,
                 need_frame_num = 5,
                 crop_patch_size = 256,
                 crop_type = 'motion',          # random, left_up, centre
                 proc_patch_level = 'single',   # single, all
                 this_patch_num = 5,
                 random_step = 4,               # 1,  2 , 4
                 enlarge_type = 'shift',
                 motion_interval = 16,
                 return_grid = False
                 ) -> None:
        super().__init__()
        self.num_frames          = need_frame_num
        self.proc_patch_level   = proc_patch_level
        if self.proc_patch_level == 'single':
            if self.num_frames < this_patch_num:
                start_frame = np.random.randint(0, this_patch_num - self.num_frames)
            else:
                start_frame = 0
            self.frame_list = [x + start_frame for x in range(self.num_frames)]
        elif self.proc_patch_level == 'all':
            self.frame_list = self.get_patch_frame_idx(this_patch_num)

        self.crop_patch_size    = crop_patch_size
        self.crop_type          = crop_type
        self.random_step        = random_step
        self.enlarge_type       = enlarge_type
        self.motion_interval = motion_interval
        self.return_grid = return_grid

    def get_patch_frame_idx(self, this_frame):
        if this_frame > self.num_frames :
            start_frame = np.random.randint(0, this_frame - self.num_frames)
            return  [x + start_frame for x in range(self.num_frames)]
        
        order_list = [x for x in range(this_frame)]
        inv_list = [this_frame - 1 - x for x in range(this_frame)]

        if len(order_list) > 1:
            temp = order_list.copy()
            cout = 0
            while len(temp) < self.num_frames:
                cout += 1
                if cout % 2 == 0:
                    temp += order_list[1:]
                else:
                    temp += inv_list[1:]
            return temp[:self.num_frames]
        else:
            return [0 for x in range(self.num_frames)]

    def get_crop_patch_suit_size_all(self, patches, grid=None, crop_size = 128, crop_type = 'random', frame_id=0):
        start_h, start_w = 0, 0
        if self.fix_idx:
            start_h, start_w = self.start_h_last, self.start_w_last
        else:
            if patches.shape[-2] > crop_size:
                if crop_type == 'random':
                    start_h = np.random.randint(0, patches.shape[-2] - crop_size) // self.random_step * self.random_step
                elif crop_type == 'centre':
                    start_h = (patches.shape[-2] - crop_size) // 2
                elif crop_type == 'motion':
                    if frame_id == 0:
                        start_h = np.random.randint(0, patches.shape[-2] - crop_size) // self.random_step * self.random_step
                        if self.fix_speed:
                            self.h_speed = np.random.randint(0-self.motion_interval, self.motion_interval)  // self.random_step * self.random_step
                    else:
                        if self.fix_speed:
                            start_h = self.start_h_last + self.h_speed + np.random.randint(-2, 2)
                        else:
                            start_h = self.start_h_last + np.random.randint(0-self.motion_interval, self.motion_interval)  // self.random_step * self.random_step
                        start_h = np.clip(start_h, 0, patches.shape[-2] - crop_size)
                        
                elif crop_type == 'left_up':
                    start_h = 0
            if patches.shape[-1] > crop_size:
                if crop_type == 'random':
                    start_w = np.random.randint(0, patches.shape[-1] - crop_size) // self.random_step * self.random_step
                elif crop_type == 'motion':
                    if frame_id == 0:
                        start_w = np.random.randint(0, patches.shape[-1] - crop_size) // self.random_step * self.random_step
                        if self.fix_speed:
                            self.w_speed = np.random.randint(0-self.motion_interval, self.motion_interval)  // self.random_step * self.random_step
                    else:
                        if self.fix_speed:
                            start_w = self.start_w_last + self.w_speed + np.random.randint(-2, 2)
                        else:
                            start_w = self.start_w_last + np.random.randint(0-self.motion_interval, self.motion_interval)  // self.random_step * self.random_step
                        start_w = np.clip(start_w, 0, patches.shape[-1] - crop_size)
                elif crop_type == 'centre':
                    start_w = (patches.shape[-1] - crop_size) // 2
                elif crop_type == 'left_up':
                    start_w = 0
            if self.return_grid:
                if frame_id > 0:
                    d_h = self.start_h_last - start_h
                    d_w = self.start_w_last - start_w
                    cal_grid = torch.cat([d_h*torch.ones_like(patches[:,:1,:crop_size,:crop_size]),
                                        d_w*torch.ones_like(patches[:,:1,:crop_size,:crop_size])],
                                        dim=1)
                else:
                    cal_grid = torch.zeros_like(patches[:,:2,:crop_size,:crop_size])
            self.start_h_last = start_h
            self.start_w_last = start_w

        output = patches[..., :, start_h:start_h + crop_size,
                                start_w:start_w + crop_size]
        if self.return_grid:
            return output, cal_grid

        if grid is None:
            return output
        else:
            outgrid = grid[..., :, start_h:start_h + crop_size,
                                start_w:start_w + crop_size]
            #outgrid[..., ::2, :, :] = outgrid[..., ::2, :, :] - start_w
            #outgrid[..., 1::2, :, :] = outgrid[..., 1::2, :, :] - start_h
            return output, outgrid

    def get_crop_patch_suit_size_single(self, patches, grid=None, crop_size = 128, crop_type = 'random'):
        # output = torch.zeros([self.])
        if crop_type == 'random' or crop_type == 'motion':
            output = []
            grid = []
            i_frame = 0
            for _ in self.frame_list:
                if self.return_grid:
                    out1, grid1 = self.get_crop_patch_suit_size_all(patches, grid = None, crop_size=crop_size, crop_type=crop_type, frame_id=i_frame)
                    output.append(out1)
                    grid.append(grid1)
                else:
                    output.append(self.get_crop_patch_suit_size_all(patches, grid = None, crop_size=crop_size, crop_type=crop_type, frame_id=i_frame))
                i_frame += 1
            output = torch.cat(output, dim=0)
            if self.return_grid:
                grid = torch.cat(grid[1:] + [grid[-1]], dim=0)
                
        else:
            _patches = patches[[0 for x in range(self.num_frames)], ...]
            # _grid = grid[[0 for x in range(self.num_frames)], ...] if grid is not None else None
            output = self.get_crop_patch_suit_size_all(_patches, grid = None, crop_size=crop_size, crop_type=crop_type)
        if self.return_grid:
            return output, grid
        return output

    def get_enlarged_patch(self, patches, grid=None, size = 128):
        enlarge_func = image_shift_for_pattern_raw
        if self.enlarge_type == 'mirror':
            enlarge_func = image_mirror_for_pattern

        patches = enlarge_func(patches, size)
        if grid is not None:
            grid = enlarge_func(grid, size)
        
        return patches, grid

    @torch.no_grad()
    def forward(self, patches: torch.Tensor, grid = None, fix_idx = False, fix_speed = False) -> torch.Tensor:
        if patches.shape[-1] < self.crop_patch_size or patches.shape[-2] < self.crop_patch_size:
            patches, grid = self.get_enlarged_patch(patches, grid, size=self.crop_patch_size)

        self.fix_idx = fix_idx
        self.fix_speed = fix_speed
        if self.proc_patch_level == 'single':
            patches_ok = self.get_crop_patch_suit_size_single(patches, grid = None, crop_size=self.crop_patch_size, crop_type=self.crop_type)
        elif self.proc_patch_level == 'all':
            patches_all = patches[self.frame_list, ...]
            grid = grid[self.frame_list, ...] if grid is not None else None
            patches_ok = self.get_crop_patch_suit_size_all(patches_all, grid, crop_size=self.crop_patch_size, crop_type=self.crop_type)

        if isinstance(patches_ok, tuple):
            return patches_ok[0], patches_ok[1]
        else:
            return patches_ok, None


class SuitPatchPrepareRect(nn.Module):
    """
    Variant of SuitPatchPrepare that supports rectangular crops (non-square).
    Uses crop_height and crop_width instead of single crop_patch_size.
    """

    def __init__(
        self,
        need_frame_num=5,
        crop_height=256,
        crop_width=256,
        crop_type="random",
        proc_patch_level="single",
        this_patch_num=5,
        random_step=4,
        enlarge_type="shift",
    ) -> None:
        super().__init__()
        self.num_frames = need_frame_num
        self.proc_patch_level = proc_patch_level
        if self.proc_patch_level == "single":
            if self.num_frames < this_patch_num:
                start_frame = np.random.randint(0, this_patch_num - self.num_frames)
            else:
                start_frame = 0
            self.frame_list = [x + start_frame for x in range(self.num_frames)]
        elif self.proc_patch_level == "all":
            self.frame_list = self.get_patch_frame_idx(this_patch_num)

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_type = crop_type
        self.random_step = random_step
        self.enlarge_type = enlarge_type

    def get_patch_frame_idx(self, this_frame):
        if this_frame > self.num_frames:
            start_frame = np.random.randint(0, this_frame - self.num_frames)
            return [x + start_frame for x in range(self.num_frames)]

        order_list = [x for x in range(this_frame)]
        inv_list = [this_frame - 1 - x for x in range(this_frame)]

        if len(order_list) > 1:
            temp = order_list.copy()
            cout = 0
            while len(temp) < self.num_frames:
                cout += 1
                if cout % 2 == 0:
                    temp += order_list[1:]
                else:
                    temp += inv_list[1:]
            return temp[:self.num_frames]
        else:
            return [0 for x in range(self.num_frames)]

    def get_crop_patch_rect_suit_size_all(
        self, patches, grid=None, crop_height=128, crop_width=128, crop_type="random"
    ):
        start_h, start_w = 0, 0
        if self.fix_idx:
            start_h, start_w = self.start_h_last, self.start_w_last
        else:
            if patches.shape[-2] > crop_height:
                if crop_type == "random":
                    start_h = (
                        np.random.randint(0, patches.shape[-2] - crop_height)
                        // self.random_step
                        * self.random_step
                    )
                elif crop_type == "centre":
                    start_h = (patches.shape[-2] - crop_height) // 2
                elif crop_type == "left_up":
                    start_h = 0
            if patches.shape[-1] > crop_width:
                if crop_type == "random":
                    start_w = (
                        np.random.randint(0, patches.shape[-1] - crop_width)
                        // self.random_step
                        * self.random_step
                    )
                elif crop_type == "centre":
                    start_w = (patches.shape[-1] - crop_width) // 2
                elif crop_type == "left_up":
                    start_w = 0
            self.start_h_last = start_h
            self.start_w_last = start_w

        output = patches[..., :, start_h : start_h + crop_height, start_w : start_w + crop_width]

        if grid is None:
            return output
        else:
            outgrid = grid[
                ..., :, start_h : start_h + crop_height, start_w : start_w + crop_width
            ]
            return output, outgrid

    def get_crop_patch_rect_suit_size_single(
        self, patches, grid=None, crop_height=128, crop_width=128, crop_type="random"
    ):
        if crop_type == "random":
            output = []
            for _ in self.frame_list:
                output.append(
                    self.get_crop_patch_rect_suit_size_all(
                        patches, grid=None, crop_height=crop_height, crop_width=crop_width, crop_type=crop_type
                    )
                )
            output = torch.cat(output, dim=0)
        else:
            _patches = patches[[0 for x in range(self.num_frames)], ...]
            output = self.get_crop_patch_rect_suit_size_all(
                _patches, grid=None, crop_height=crop_height, crop_width=crop_width, crop_type=crop_type
            )
        return output

    def get_enlarged_patch(self, patches, grid=None, height=128, width=128):
        # For rectangular crops, we use the larger dimension for enlarging
        max_size = max(height, width)
        enlarge_func = image_shift_for_pattern_raw
        if self.enlarge_type == "mirror":
            enlarge_func = image_mirror_for_pattern

        patches = enlarge_func(patches, max_size)
        if grid is not None:
            grid = enlarge_func(grid, max_size)

        return patches, grid

    @torch.no_grad()
    def forward(self, patches: torch.Tensor, grid=None, fix_idx=False) -> torch.Tensor:
        if (
            patches.shape[-2] < self.crop_height
            or patches.shape[-1] < self.crop_width
        ):
            patches, grid = self.get_enlarged_patch(
                patches, grid, height=self.crop_height, width=self.crop_width
            )

        self.fix_idx = fix_idx
        if self.proc_patch_level == "single":
            patches_ok = self.get_crop_patch_rect_suit_size_single(
                patches,
                grid=None,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                crop_type=self.crop_type,
            )
        elif self.proc_patch_level == "all":
            patches_all = patches[self.frame_list, ...]
            grid = grid[self.frame_list, ...] if grid is not None else None
            patches_ok = self.get_crop_patch_rect_suit_size_all(
                patches_all,
                grid,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                crop_type=self.crop_type,
            )

        if isinstance(patches_ok, tuple):
            return patches_ok[0], patches_ok[1]
        else:
            return patches_ok, None


class RandomBrightnessGamma(nn.Module):
    def __init__(
        self,
        gain_range=(0.8, 1.2),
        gamma_range=(0.5, 1.5),
        p_gain=0.3,
        p_gamma=0.3,
        p_dark_gain=0.1,
        dark_gain_range=(0.05, 0.15),
    ):
        super().__init__()
        self.gain_range = gain_range
        self.gamma_range = gamma_range
        self.p_gain = p_gain
        self.p_gamma = p_gamma
        self.p_dark_gain = p_dark_gain
        self.dark_gain_range = dark_gain_range
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, C, H, W] 或 [B, T, C, H, W]
        """
        if x.dim() == 4:
            return self._apply_single(x)
        elif x.dim() == 5:
            # 如果是 batch 的视频：对每个样本独立随机
            B = x.size(0)
            out_list = []
            for b in range(B):
                out_list.append(self._apply_single(x[b]))
            return torch.stack(out_list, dim=0)
        else:
            raise ValueError("Input must be [T, C, H, W] or [B, T, C, H, W]")
    def _apply_single(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        out = x
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            out = torch.clamp(out, min=0.0)
            out = out ** gamma
        if random.random() < self.p_dark_gain:
            gain = random.uniform(*self.dark_gain_range)
            out = out * gain
        elif random.random() < self.p_gain:
            gain = random.uniform(*self.gain_range)
            out = out * gain
        out = torch.clamp(out, 0.0, 1.0)
        return out


class RandomColorize:
    def __init__(self, hue_range=0.1, saturation_range=0.5, brightness_range=0.3):
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
    
    def __call__(self, img):
        """
        img: PIL Image 或 tensor
        """
        # # 如果是灰度图，先转换为RGB
        # if img.mode == 'L':
        #     img = img.convert('RGB')
        
        # 随机颜色调整
        hue_factor = random.uniform(-self.hue_range, self.hue_range)
        saturation_factor = random.uniform(1-self.saturation_range, 1+self.saturation_range)
        brightness_factor = random.uniform(1-self.brightness_range, 1+self.brightness_range)
        
        # 调整色调、饱和度、亮度
        img = F.adjust_hue(img, hue_factor)
        img = F.adjust_saturation(img, saturation_factor)
        img = F.adjust_brightness(img, brightness_factor)
        return img


def image_read(data_file, tif = False, en_ratio = False):
    if tif:
        img_bgr_ori = cv2.imread(data_file, -1)
        ratio = 65535.0
    else:
        img_bgr_ori = cv2.imread(data_file)
        ratio = 255.0
    if en_ratio:
        img_bgr_ori = img_bgr_ori / ratio
    return img_bgr_ori


def StaticAngleAndExchangeRGB(img_bgr_ori, static_angle=False, exchange_rgb=False):
    if static_angle:
        angle = random.choices([0.0, 90.0, 180.0, 270.0], weights=[0.4, 0.2, 0.2, 0.2])
        angle_np = np.array(angle)
        img_bgr_ori, heightNew, widthNew = random_rotate(img_bgr_ori, angle_np[0])
    if exchange_rgb:
        exchange_prob = random.random()
        if exchange_prob < 0.1:
            img_bgr_exc = img_bgr_ori[:, :, [0, 2, 1]]
        elif exchange_prob < 0.2:
            img_bgr_exc = img_bgr_ori[:, :, [1, 0, 2]]
        elif exchange_prob < 0.3:
            img_bgr_exc = img_bgr_ori[:, :, [1, 2, 0]]
        elif exchange_prob < 0.4:
            img_bgr_exc = img_bgr_ori[:, :, [2, 0, 1]]
        elif exchange_prob < 0.5:
            img_bgr_exc = img_bgr_ori[:, :, [2, 1, 0]]
        else:
            img_bgr_exc = img_bgr_ori
        img_bgr_ori = img_bgr_exc

    return img_bgr_ori


def data_resize(img, out_h=128, out_w=128):
    out_img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return out_img


def image_crop(img_bgr_ori, crop_patch_size):
    start_h = random.randint(0, img_bgr_ori.shape[0] - crop_patch_size)
    start_w = random.randint(0, img_bgr_ori.shape[1] - crop_patch_size)
    img_bgr_ori = img_bgr_ori[start_h:start_h + crop_patch_size, start_w:start_w + crop_patch_size, :]
    return img_bgr_ori



def img_aug(img, coin):
    if coin < 0.25:
        return img
    elif coin < 0.5:
        return cv2.flip(img, 1)
    elif coin < 0.75:
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, -1)


def img_color_aug(img_16bit):
    img_gbr = img_16bit[:, :, (1, 0, 2)]
    img_brg = img_16bit[:, :, (0, 2, 1)]
    coin = random.random()
    if coin > 0.8:
        output_img = img_16bit.copy()
        output_img[:, :, 0:1] = img_16bit[:, :, 1:2]  # bgr 2 ggr
    elif coin > 0.6:
        output_img = img_16bit.copy()
        output_img[:, :, 1:2] = img_16bit[:, :, 1:2]  # bgr 2 bgg
    elif coin > 0.4:
        output_img = img_16bit.copy()
        output_img[:, :, 0:1] = img_16bit[:, :, 1:2]
        output_img[:, :, 1:2] = img_16bit[:, :, 1:2]  # bgr 2 ggg
    elif coin > 0.2:
        output_img = img_gbr
    else:
        output_img = img_brg
    return output_img


def img_saturate(image, lightness, saturation):
    """
    :param image:
    :param lightness: -100, 100, ??????10-90
    :param saturation: -100, 100, ??????10-90
    :return:
    """
    # ?????? ?????????????????????????
    # ????????? BGR??HLS
    MAX_VALUE = 100
    image = image.astype(np.float32) / 65535.0
    # lightness = 50
    # saturation = 50
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 1.?????????????��)
    hlsImg[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # ?????
    hlsImg[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 65535.0 + 0.5
    lsImg = lsImg.astype(np.uint16)
    return lsImg


def image_prepare(data_file, crop_patch_size=128, num_frames=5, tif = False, static_angle=False, exchange_rgb=False, img_png_aug_flag=True):
    img_bgr_ori = image_read(data_file, tif, False)
    img_bgr_ori = StaticAngleAndExchangeRGB(img_bgr_ori, static_angle, exchange_rgb)

    if img_bgr_ori.shape[0] < crop_patch_size:
        img_bgr_ori = image_mirror_for_pattern(torch.from_numpy(img_bgr_ori.transpose(2, 0, 1)).float(),crop_patch_size ).numpy().transpose(1, 2, 0)
    else:
        if random.random() > 0.5:
            img_bgr_ori = data_resize(img_bgr_ori, out_h=crop_patch_size, out_w=crop_patch_size)
        else:
            img_bgr_ori = image_crop(img_bgr_ori, crop_patch_size)

    img_hdr_tif = img_bgr_ori
    if img_png_aug_flag:
        coin = random.random()
        img_hdr_tif = img_aug(img_bgr_ori, coin)
    if tif:
        img_hdr_tif = img_color_aug(img_hdr_tif)
        lightness, saturation = random.randint(10, 90), random.randint(10, 90)
        img_hdr_tif = img_saturate(img_hdr_tif, lightness, saturation)

    img_bgr_ori = img_hdr_tif.transpose(2, 0, 1)  # hwc -> chw
    ratio = 65535.0 if tif else 255.0
    img_bgr_ori = torch.from_numpy(img_bgr_ori / ratio).type(torch.float32)
    return img_bgr_ori.unsqueeze(0).repeat(num_frames, 1, 1, 1)


def gen_color_checker_24(step, color_size=150):
    color_24_rgb_list = np.array([[115, 82, 69], [204, 161, 141], [101, 134, 179], [89, 109, 61], [141,137,194],
            [132,228,208], [249,118,35], [80,91,182], [222,91,125], [91,63,123], [173,232,91],
            [255,164,26], [44,56,142], [74,148,81], [179,42,50], [250,226,21], [191,81,160],
            [6,142,172], [252,252,252], [230,230,230], [200,200,200], [143,143,142], [100,100,100], [50,50,50]])
    
    image           = np.zeros((4*color_size+5*step, 6*color_size+7*step, 3))
    shuffle_index   = [x for x in range(24)]
    random.shuffle(shuffle_index)
    image = np.ones((4*color_size+5*step, 6*color_size+7*step, 3)) * np.random.randint(0,256)
    ## bgr
    for i in range(4):
        for j in range(6):
            image[step+i*(color_size+step): step+i*(color_size+step)+color_size, step+j*(color_size+step):step+j*(color_size+step)+color_size, :] = color_24_rgb_list[i*6+j, [2,1,0]] # shuffle_index[i*6+j]
    # cv2.imwrite('color_check.png', image)
    return torch.from_numpy(image.transpose(2, 0, 1) / 255.0).float()
