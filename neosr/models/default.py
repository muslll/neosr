import os
import time
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import pytorch_optimizer
import torch
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

from neosr.archs import build_network
from neosr.losses import build_loss
from neosr.losses.loss_util import get_refined_artifact_map
from neosr.metrics import calculate_metric
from neosr.optimizers import build_optimizer
from neosr.utils import get_root_logger, imwrite, tensor2img
from neosr.utils.dist_util import master_only
from neosr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class default():
    """Default model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')
        self.is_train = opt['is_train']
        self.optimizers = []
        self.schedulers = []

        # define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        if self.opt.get('print_network', False) is True:
            self.print_network(self.net_g)

        # define network net_d
        if self.opt.get('network_d', None) is not None:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            if self.opt.get('print_network', False) is True:
                self.print_network(self.net_d)

        # load pretrained g
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g')
            self.load_network(self.net_g, load_path, param_key, self.opt['path'].get(
                'strict_load_g', True))

        # load pretrained d
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d')
            self.load_network(self.net_d, load_path, param_key, self.opt['path'].get(
                'strict_load_d', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_g.train()
        if self.opt.get('network_d', None) is not None:
            self.net_d.train()

        # for amp
        self.use_amp = self.opt.get('use_amp', False) is True
        self.amp_dtype = torch.bfloat16 if self.opt.get('bfloat16', False) is True else torch.float16

        # initialise GradScale object for Generator and Discriminator
        self.scaler_g = torch.cuda.amp.GradScaler(enabled = self.use_amp, init_scale=2.**5)
        self.scaler_d = torch.cuda.amp.GradScaler(enabled = self.use_amp, init_scale=2.**5)

        # initialise counter of how many gradients has to be accumulated
        self.n_accumulated = 0
        self.accum_iters = self.opt["datasets"]["train"].get("accumulate", 1)
        if self.accum_iters == 0 or self.accum_iters == None:
            self.accum_ters = 1

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(
                train_opt['perceptual_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_perceptual = None

        # GAN loss
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_gan = None

        # LDL loss
        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_ldl = None

        # Color loss
        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_color = None

        # Luma loss
        if train_opt.get('luma_opt'):
            self.cri_luma = build_loss(train_opt['luma_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_luma = None

        # Focal Frequency Loss
        if train_opt.get('ff_opt'):
            self.cri_ff = build_loss(train_opt['ff_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_ff = None

        # Patch Loss
        if train_opt.get('patch_opt'):
            self.cri_patch = build_loss(train_opt['patch_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_patch = None

        if train_opt.get('patch3d_opt'):
            self.cri_patch3d = build_loss(train_opt['patch3d_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_patch3d = None

        if train_opt.get('patch3dxd_opt'):
            self.cri_patch3dxd = build_loss(train_opt['patch3dxd_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_patch3dxd = None

        # Gradient Variance Loss
        if train_opt.get('gradvar_opt'):
            self.cri_gradvar = build_loss(train_opt['gradvar_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_gradvar = None

        # Best Buddy GANs Loss
        if train_opt.get('bbl_opt'):
            self.cri_bbl = build_loss(train_opt['bbl_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_bbl = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_patch is None and self.cri_patch_3d is None and self.cri_patch_3d_xd is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type in {'Adam', 'adam'}:
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type in {'AdamW', 'adamw'}:
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type in {'Adan', 'adan'}:
            optimizer = pytorch_optimizer.Adan(params, lr, **kwargs)
        elif optim_type in {'Lamb', 'lamb'}:
            optimizer = pytorch_optimizer.Lamb(params, lr, **kwargs)
        elif optim_type in {'Lion', 'lion'}:
            optimizer = pytorch_optimizer.Lion(params, lr, **kwargs)
        elif optim_type in {'prodigy', 'Prodigy'}:
            optimizer = pytorch_optimizer.Prodigy(params, lr, **kwargs)
        elif optim_type in {'AdamL', 'adaml'}:
            optimizer = build_optimizer('AdamL', params, lr, **kwargs)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        if self.opt.get('network_d', None) is not None:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(
                optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        self.scheduler_type = scheduler_type

        if scheduler_type not in {'none', 'disabled'}:
            if scheduler_type in {'MultiStepLR', 'multisteplr'}:
                for optimizer in self.optimizers:
                    self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, **train_opt['scheduler']))
            elif scheduler_type in {'CAWR', 'cawr'}:
                for optimizer in self.optimizers:
                    self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, **train_opt['scheduler']))
            elif scheduler_type in {'CosineAnnealing', 'cosineannealing'}:
                for optimizer in self.optimizers:
                    self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, **train_opt['scheduler']))
            else:
                raise NotImplementedError(
                    f'Scheduler {scheduler_type} is not implemented yet.')
        else:
            self.scheduler_type = None
            print("INFO: No scheduler in use.")

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr']
                                    for v in optimizer.param_groups])
        return init_lr_groups_l

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=True):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=True):
                param_group['lr'] = lr

    def optimize_parameters(self, current_iter):

        if self.opt.get('network_d', None) is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        # LQ matching for Color/Luma losses
        self.match_lq = self.opt['train'].get('match_lq', False)
        self.lq_interp = F.interpolate(self.lq, scale_factor=self.opt['scale'], mode='bicubic')

        # increment accumulation counter and check if accumulation limit has been reached
        self.n_accumulated += 1
        apply_gradient = self.n_accumulated >= self.accum_iters

        # reset the counter back to zero
        if apply_gradient:
            self.n_accumulated = 0
        
        # define list of losses, each individual loss needs to have the scaler applied separately
        losses_for_backward_g = []

        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            self.output = self.net_g(self.lq)

            l_g_total = 0
            loss_dict = OrderedDict()

            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix
                    losses_for_backward_g.append(l_g_pix)
                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep,l_g_style,l_resnet,l_vgg = self.cri_perceptual(self.output, self.gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                        losses_for_backward_g.append(l_g_percep)
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style
                        losses_for_backward_g.append(l_g_style)
                    if l_resnet is not None:
                        loss_dict['l_resnet'] = l_resnet
                    if l_vgg is not None:
                        loss_dict['l_vgg'] = l_vgg
                # patch loss
                if self.cri_patch:
                    l_g_patch = self.cri_patch(self.output, self.gt)
                    l_g_total += l_g_patch
                    loss_dict['l_g_patch'] = l_g_patch
                    losses_for_backward_g.append(l_g_patch)
                if self.cri_patch3d:
                    l_g_patch3d = self.cri_patch3d(self.output, self.gt)
                    l_g_total += l_g_patch3d
                    loss_dict['l_g_patch_3d'] = l_g_patch3d
                    losses_for_backward_g.append(l_g_patch3d)
                if self.cri_patch3dxd:
                    l_g_patch3dxd = self.cri_patch3dxd(self.output, self.gt)
                    l_g_total += l_g_patch3dxd
                    loss_dict['l_g_patch3dxd'] = l_g_patch3dxd
                    losses_for_backward_g.append(l_g_patch3dxd)
                # ldl loss
                if self.cri_ldl:
                    pixel_weight = get_refined_artifact_map(self.gt, self.output, 7)
                    l_g_ldl = self.cri_ldl(
                        torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
                    l_g_total += l_g_ldl
                    loss_dict['l_g_ldl'] = l_g_ldl
                    losses_for_backward_g.append(l_g_ldl)
                # color loss
                if self.cri_color:
                    if self.match_lq:
                        l_g_color = self.cri_color(self.output, self.lq_interp)
                    else:
                        l_g_color = self.cri_color(self.output, self.gt)
                    l_g_total += l_g_color
                    loss_dict['l_g_color'] = l_g_color
                    losses_for_backward_g.append(l_g_color)
                # luma loss
                if self.cri_luma:
                    if self.match_lq:
                        l_g_luma = self.cri_luma(self.output, self.lq_interp)
                    else:
                        l_g_luma = self.cri_luma(self.output, self.gt)
                    l_g_total += l_g_luma
                    loss_dict['l_g_luma'] = l_g_luma
                    losses_for_backward_g.append(l_g_luma)
                # Focal Frequency Loss
                if self.cri_ff:
                    l_g_ff = self.cri_ff(self.output, self.gt)
                    l_g_total += l_g_ff
                    loss_dict['l_g_ff'] = l_g_ff
                    losses_for_backward_g.append(l_g_ff)
                # Gradient Variance Loss
                if self.cri_gradvar:
                    l_g_gradvar = self.cri_gradvar(self.output, self.gt)
                    l_g_total += l_g_gradvar
                    loss_dict['l_g_gradvar'] = l_g_gradvar
                    losses_for_backward_g.append(l_g_gradvar)
                # pixel loss
                if self.cri_bbl:
                    l_g_bbl = self.cri_bbl(self.output, self.gt)
                    l_g_total += l_g_bbl
                    loss_dict['l_g_bbl'] = l_g_bbl
                    losses_for_backward_g.append(l_g_bbl)
                # GAN loss
                if self.cri_gan:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    loss_dict['l_g_gan'] = l_g_gan
                    losses_for_backward_g.append(l_g_gan)

        # add total loss to loss_dict for tensorboard tracking
        loss_dict['l_g_total'] = l_g_total

        # iterate through the losses, retaining graph on all but the last
        for loss_idx, loss_g in enumerate(losses_for_backward_g):
            is_last_loss = loss_idx == len(losses_for_backward_g) - 1
            loss_g = loss_g / self.accum_iters
            self.scaler_g.scale(loss_g).backward(retain_graph = not is_last_loss)

        if apply_gradient:
            # gradient clipping on generator
            if self.opt["train"].get("grad_clip", True):
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 1.0, error_if_nonfinite=False)

            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.optimizer_g.zero_grad(set_to_none=True)

        # optimize net_d
        if self.opt.get('network_d', None) is not None:
            for p in self.net_d.parameters():
                p.requires_grad = True

            losses_for_backward_d = []

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                # real
                if self.cri_gan:
                    real_d_pred = self.net_d(self.gt)
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict['l_d_real'] = l_d_real
                    loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                    losses_for_backward_d.append(l_d_real)
                # fake
                    fake_d_pred = self.net_d(self.output.detach().clone())
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict['l_d_fake'] = l_d_fake
                    loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                    losses_for_backward_d.append(l_d_fake)

                # Iterate through the losses, retaining graph on all but the last
                for loss_idx, loss_d in enumerate(losses_for_backward_d):
                    is_last_loss = loss_idx == len(losses_for_backward_d) - 1
                    loss_d = loss_d / self.accum_iters
                    self.scaler_d.scale(loss_d).backward(retain_graph=not is_last_loss)

                if apply_gradient:
                    # gradient clipping on discriminator
                    if self.opt["train"].get("grad_clip", True):
                        self.scaler_d.unscale_(self.optimizer_d)
                        torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 1.0, error_if_nonfinite=False)

                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                    self.optimizer_d.zero_grad(set_to_none=True)

            self.log_dict = self.reduce_loss_dict(loss_dict)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if self.scheduler_type is not None:
            if current_iter > 0 and self.n_accumulated == 0:
                for scheduler in self.schedulers:
                    scheduler.step()
            # set up warm-up learning rate
            if current_iter < warmup_iter:
                # get initial lr for each group
                init_lr_g_l = self._get_init_lr()
                # modify warming-up learning rates
                # currently only support linearly warm up
                warm_up_lr_l = []
                for init_lr_g in init_lr_g_l:
                    warm_up_lr_l.append(
                        [v / warmup_iter * current_iter for v in init_lr_g])
                # set learning rate
                self._set_lr(warm_up_lr_l)

    def test(self):
        # pad to multiplication of window_size
        if self.opt.get('window_size', None) is not None:
            window_size = self.opt.get('window_size') 
        elif self.opt.get('patch_size', None) is not None:
            window_size = self.opt.get('patch_size')
        else:
            window_size = 8

        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        self.net_g.eval()
        with torch.inference_mode():
            self.output = self.net_g(img)
        self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h -
                                  mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device, memory_format=torch.channels_last, non_blocking=True)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device, memory_format=torch.channels_last, non_blocking=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(
                current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def validation(self, dataloader, current_iter, tb_logger, save_img=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        if self.opt['use_amp'] is True:
            net = net.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        else:
            net = net.to(self.device, non_blocking=True)

        if self.opt['compile'] is True:
            net = torch.compile(net)
            # see option fullgraph=True

        if self.opt['dist']:
            find_unused_parameters = self.opt.get(
                'find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key, strict=True):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}.')
            raise IOError(f'Cannot save {save_path}.')

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        self.save_network(self.net_g, 'net_g', current_iter)

        if self.opt.get('network_d', None) is not None:
            self.save_network(self.net_d, 'net_d', current_iter)

        self.save_training_state(epoch, current_iter)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, param_key, strict=True):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: None.
        """
        self.param_key = param_key
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=torch.device('cuda'))

        try:
            if 'params-ema' in load_net:
                param_key = 'params-ema'
            elif 'params' in load_net:
                param_key = 'params'
            elif 'params_ema' in load_net:
                param_key = 'params_ema'
            else:
                param_key = self.param_key
            load_net = load_net[param_key]
        except:
            pass

        if param_key:
            logger.info(
                f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        else:
            logger.info(
                f'Loading {net.__class__.__name__} model from {load_path}.')

        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
        torch.cuda.empty_cache()

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """

        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter,
                    'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{int(current_iter)}.state'
            save_path = os.path.join(
                self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(
                    f'Still cannot save {save_path}. Just ignore it.')
                raise IOError(f'Cannot save, aborting.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.inference_mode():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses, strict=True)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
