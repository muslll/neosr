import os
import os.path as osp
import random
import time
import math
import cv2
import numpy as np
import torch
from torch.utils import data

from neosr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from neosr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class otf_paired(data.Dataset):
    """OTF degradation dataset. Originally from Real-ESRGAN

    It loads lq (Low-Quality) images, and augments them.
    It also generates blur kernels and sinc kernels for degrading low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            gt_size (int): Cropped patched size for gt patches.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(otf_paired, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(
                    dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], [
                                                  'lq', 'gt'], self.filename_tmpl)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        # a list for each kernel probability
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        # betag used in generalized Gaussian blur kernels
        self.betag_range = opt['betag_range']
        # betap used in plateau blur kernels
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect

        # Note: this operation must run on cpu, otherwise CUDAPrefetcher will fail
        with torch.device('cpu'):
            self.pulse_tensor = torch.zeros(21, 21, dtype=torch.float32)
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load lq & gt images -------------------------------- #
        # initialize gt images
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(gt_path)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')

        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except AttributeError:
            raise AttributeError(lq_path)

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                if img_bytes is None:
                    raise ValueError(f'No data returned from path: {gt_path}, {lq_path}')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e} in paths {gt_path}, {lq_path}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = gt_path[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1


        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        gt_size = self.opt['gt_size']
        # random crop
        img_gt, img_lq = paired_random_crop(
            img_gt, img_lq, gt_size, scale, gt_path)
        # flip, rotation
        img_gt, img_lq = basic_augment(
            [img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 512
        # TODO: 512 is hard-coded. You may change it accordingly
        gt_h, gt_w = img_gt.shape[0:2]
        lq_h, lq_w = img_lq.shape[0:2]
        crop_pad_size = 128
        # pad
        if lq_h < crop_pad_size or lq_w < crop_pad_size:
            pad_lq_h = max(0, crop_pad_size - lq_h)
            pad_lq_w = max(0, crop_pad_size - lq_w)
            img_lq = cv2.copyMakeBorder(
                img_lq, 0, pad_lq_h, 0, pad_lq_w, cv2.BORDER_REFLECT_101)
        if gt_h < (crop_pad_size * scale) or gt_w < (crop_pad_size * scale):
            pad_gt_h = max(0, (crop_pad_size * scale) - gt_h)
            pad_gt_w = max(0, (crop_pad_size * scale) - gt_w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_gt_h, 0, pad_gt_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_lq.shape[0] > crop_pad_size or img_lq.shape[1] > crop_pad_size:
            lq_h, lq_w = img_lq.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, lq_h - crop_pad_size)
            left = random.randint(0, lq_w - crop_pad_size)
            img_lq = img_lq[top:top + crop_pad_size,
                            left:left + crop_pad_size, ...]
        if img_gt.shape[0] > (crop_pad_size * scale) or img_gt.shape[1] > (crop_pad_size * scale):
            lq_h, lq_w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, lq_h - (crop_pad_size * scale))
            left = random.randint(0, lq_w - (crop_pad_size * scale))
            img_gt = img_gt[top:top + (crop_pad_size * scale),
                            left:left + (crop_pad_size * scale), ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        rng = np.random.default_rng()
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if rng.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = rng.uniform(np.pi / 3, np.pi)
            else:
                omega_c = rng.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if rng.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = rng.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # NOTE: using torch.tensor(device='cuda') won't work.
        # Keeping old constructor for now.
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'lq': img_lq, 'kernel1': kernel, 'kernel2': kernel2,
                    'sinc_kernel': sinc_kernel, 'gt_path': gt_path, 'lq_path': lq_path}
        return return_d

    def __len__(self):
        return len(self.paths)