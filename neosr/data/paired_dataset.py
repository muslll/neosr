import os
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision.transforms.functional import normalize, rgb_to_grayscale

from neosr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.utils import get_root_logger, FileClient, imfrombytes, img2tensor
from neosr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class paired(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(paired, self).__init__()
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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

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

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = basic_augment(
                [img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            # TODO: currently torchvision v2.Grayscale doesn't output one channel
            # Switch to it once fixed, to avoid such conversions

            img_gt = np.round(img_gt * 255.0).astype(np.uint8)
            img_gt = Image.fromarray(img_gt)
            img_gt = rgb_to_grayscale(img_gt)
            img_gt = np.array(img_gt, dtype=np.float32) / 255.
            img_gt = np.expand_dims(img_gt, axis=-1)

            img_lq = np.round(img_lq * 255.0).astype(np.uint8)
            img_lq = Image.fromarray(img_lq)
            img_lq = rgb_to_grayscale(img_lq)
            img_lq = np.array(img_lq, dtype=np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=-1)

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] *
                            scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor(
            [img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)