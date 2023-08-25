# Configuration Walkthrough

In this document each configuration option for `neosr` will be explained. [Templates]() can be used for convenience.

## Initial Notes

- Avoid special characters in both your paths and on filenames. Although parsing UTF-8 isn't a problem in most cases, it can still potentially cause problems.

- Prefer full paths over of relative paths. Both can be used, but full paths avoid user confusion.

- Make sure to use slashes in the path that match your system. For Windows, backslashes should be used (`\`). For Unix-like systems (OSX and Linux), you should use normal slashes (`/`).

- Use quotation marks for all paths (especially if they contain spaces or special characters in it).

- Pay attention to indentation. The configuration is done in yaml (for now), so if the indentation is wrong, you may find problems.

- **Do not** mix OTF degradations with `paired` and `default`. OTF should always be used with model `otf` and dataloader `otf`.

## Header Options

In this document we will use [train_compact.yml]() as the base configuration file.

---
#### `name`

The `name` option sets the folder name where your training files will be stored. It's a convention to use a prefix based on the scale factor of the model you're training:

```
name: 4x_mymodel
```

---
#### `model_type`

The `model_type` option specifies which model should be used. If you are training with a `paired` or `single` dataset, you should set it to `default`:

```
model_type: default
```

If you want to use on-the-fly degradations, set it to `otf` instead.

---
#### `scale`

The `scale` option sets the scale ratio of your generator. It can be 1x, 2x or 4x:

```
scale: 4
```

---
#### `num_gpu`

The `num_gpu` sets the number of GPUs. Typically, you should set it to `1` unless you're doing distributed training. For distributed training, see below.

```
num_gpu: 1
```

---
#### `use_amp`, `bfloat16`

The `use_amp` option enables [Automatic Mixed Precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) to speed up training. If you are using a GPU with tensor cores (Nvidia Turing or higher), using AMP is recommended. The `bfloat16` option sets the dtype to [BFloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) instead of the default float16. Using this is recommended if your GPU has support (Nvidia Ampere or higher), as it won't cause precision problems.

```
# Turing or above
use_amp: true
# Ampere or above
bfloat16: true
```

---
#### `compile`

The `compile` option is experimental. This option enables pytorch's new [`torch.compile()`](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever), which can speed up training. As of this writing, pytorch 2.0.0 does not have support for `compile` using Python 3.11. Pytorch version 2.1 is expected to fix this, as well as better Windows support.

```
compile: true
```

**Note**: For now, this option won't work if you enable `ema_decay`.

---
#### `manual_seed`

The `manual_seed` option enables deterministic training. If you're not doing precise comparisons, this option *should* be commented out, otherwise training performance will decrease.

```
manual_seed: 1024
```
**Note**: Using high seeds is recommended because of `torch.Generator`.

---
### Dataset options

This section describes the options within:

```
datasets:
    train:
```

---
#### (dataset) `type`

The `type` option specifies the type of dataset loader.
Possible options are `paired`, `single` and `otf`. The `single` type should only be used for inference, not training. The `paired` option is the default one and will only work if you have LQ images set in `dataroot_lq`. The `otf` type is meant to be used together with `model: otf`.
```
datasets:
    train:
        type: paired # For paired datasets
        #type: otf # For on-the-fly degradations
```

---
#### `dataroot_gt`, `dataroot_lq`

The `dataroot_gt` and `dataroot_lq` options are the folder paths to your dataset. This can be either normal images or an LMDB database.
The "gt" (ground-truth) are the ideal images, the ones you want your model to transform your images to. The "lq" (low quality) images are the degraded ones.
For a folder with images, just include the path:
```
dataroot_gt: "C:\My_Images_Folder\gt\"
dataroot_lq: "C:\My_Images_Folder\lq\"
```
If you're using LMDB, both paths should end with the `.lmdb` suffix:
```
dataroot_gt: "/home/user/dataset_gt.lmdb"
dataroot_lq: "/home/user/dataset_lq.lmdb"
```

---
#### `meta_info`

The `meta_info` (*optional*) option is a text file describing the image file names. This is *optional*, but recommended to avoid unexpected training aborts due to dataset errors such as file name mismatches.

To generate the meta_info, you can use the script [generate_meta_info.py](dataset/util/generate_meta_info.py).

**Note**: If you use `create_lmdb.py` to convert all your images into an LMDB database, the `meta_info` option is not necessary, as the script will automatically generate and link it.

```
meta_info: "C:\meta_info.txt"
```

---
#### `io_backend`

The `io_backend` type option has two possible variables: `disk` or `lmdb`. If you're using a folder with images, config should be:
```
datasets:
    train:
        io_backend:
            type: disk
```
Or if you're using LMDB:
```
datasets:
    train:
        io_backend:
            type: lmdb
```

---
#### `gt_size`

The `gt_size` is one of the most important options you have to change. It sets the size that each image will be cropped before being sent to the network.

Notes on `gt_size`:
- gt_size is the crop size being applied to your GT. The LQ pair will be cropped to gt_size divided by your scale ratio. For example, if you set `gt_size: 128` and `scale: 4` that means your GT will be 128px and your LQ will be 32px. This is important, because if you have a different `scale` you have to adapt your gt_size, otherwise you might run out of VRAM due to your LQ crop being bigger than before. For example, using `gt_size: 128` with `scale: 1` will lead to your LQ crop being the same as your GT, which means it will consume a great amount of VRAM.

- Commonly used *constant* values for `gt_size` are:
    - For a *4x* model: `128`, `192`, `256`
    - For a *2x* model: `64`, `96`, `128`
    - For a *1x* model: `32`, `48` and `64`

<br>

- Depending on the arch you're using, you may encounter tensor size mismatches and other problems with some `gt_size` values. In general, multiples of 8 or 16 should work on most networks.

- For transformers, your `gt_size` must be divisible by the window size. Standard values for window size are 8, 12, 16, 24 and 32.

- Increasing `gt_size` will lead to better end results (better model restoration accuracy), but VRAM usage will increase quadratically.

```
gt_size: 128
```

---
#### `batch_size`

The `batch_size` option specifies the number of images to feed the network in each iteration.

Notes on `batch_size`:
- Large batches have normalizing effect, i.e. training becomes more stable.
- Research shows that the batch size not only stabilizes training, but also makes the network learn faster. It may also improve the accuracy of the final restoration, although this depends on the optimizer you're using.
- Common batch_size values are: 4, 8 and 16. Anything higher than 64 can be considered "high batch" (in research).
- `batch_size` sets batches **per gpu**.

---
#### `use_hflip`, `use_rot`

The `use_hflip` and `use_rot` options are augmentations. It will rotate and flip images during training to increase variety. This is a standard basic augmentation that has been shown to improve models.

```
use_hflip: true
use_rot: true
```

---
### `num_worker_per_gpu`

The `num_worker_per_gpu` option is the number of threads used by the [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

```
num_worker_per_gpu: 6
```

---
### `dataset_enlarge_ratio`

The `dataset_enlarge_ratio` option is used to artificially increase the size of the dataset. If your dataset is too small, training will reach an epoch too fast, causing slowdowns. Using this option will duplicate the dataset by N times, so epochs will take longer to reach.

```
dataset_enlarge_ratio: 1
```

---
### Validation

The validation options, when enabled, will automatically run your model in a folder of images every time the `val_freq` iter is reached.
For example:

```
datasets:
    train:
        val:
            name: any_name
            type: single
            dataroot_lq: "/folder/path"
            io_backend:
                type: disk

val:
    val_freq: 1000
    save_img: true
```

The above configuration will perform inference on the `dataroot_lq` folder whenever it reaches 1000 iterations (`val_freq: 1000`).
Alternatively, you can use a paired validation set (both GT and LQ) and calculate metrics such as PSNR and SSIM:

```
datasets:
    train:
        val:
            name: any_name
            type: paired
            dataroot_gt: "/folder/path/gt/"
            dataroot_lq: "/folder/path/lq/"
            io_backend:
                type: disk

val:
    val_freq: 1000
    save_img: true
    metrics:
        psnr:
            type: calculate_psnr
            crop_border 4
        ssim:
            type: calculate_ssim
            crop_border: 4
```
Validation results are saved in `experiments/model_name/visualization/`. The metric value can be seen on the training log file, with `tensorboard` or `wandb` (see Logger options below).

---
### `path`

The `path` options describe the path for pretrained models or resume state.

```
# Generator Pretrain
pretrain_network_g: "/path/to/pretrain.pth"
# Discriminator Pretrain
pretrain_network_d: "/path/to/pretrain.pth"
```
If you have problems loading a pretrain and you have made sure that both your `scale` and `network_g`/`network_d` parameters are the same as the pretrained model, you can try using one of the following options:

```
param_key_g: params_ema
strict: false
```

If you have a `.state` file that you want to load, comment out the `pretrain_network_*` option and use `resume_state` instead:

```
resume_state: "/path/to/pretrain.state"
```

---
### `network_g` and `network_d`

These options describe which network architecture should be used. For a list of supported architectures, see the neosr [`readme.md`](readme.md).
Unless the template files has some network parameter explicitly commented, all network parameters are set to defaults based on their research papers. This means that you don't need to manually type any parameters, just use their names. For example:

```
network_g:
    type: dat_s
    upscale: 2

network_d:
    type: unet
```
The above option will train the DAT-S generator with the U-net discriminator.

**Note**: Some networks have a parameter to specify the upscaling factor. These should be set to the same value as your `scale` option.
The name of this parameter varies for each arch (`upscale`, `upsampling`, etc). By default, it's will always be set to `4`, so if you're training a 2x model make sure this parameter is the same.

---
### `print_network`

This option is for logging only. If set to `true`, it will prince the whole network in the terminal and save to your training log file.

```
print_network: false
```

---
### Train

These options describe the main training options, such as optimizers and losses.

---
### `optim_g` and `optim_d`

The `optim_` options set the optimizers for the `g`enerator and `d`iscriminator, and their options. For the supported optimizers, see the [`readme`](readme.md). For their respective options, see [pytorch documentation](https://pytorch.org/docs/stable/optim.html) and [pytorch-optimizer](https://github.com/kozistr/pytorch_optimizer).

```
train:
    optim_g:
        type: adamw
        lr: !!float 1e-4
        weight_decay: 0
        betas: [0.9, 0.99]
        fused: true
    optim_d:
        type: adamw
        lr: !!float 1e-4
        weight_decay: 0
        betas: [0.9, 0.99]
        fused: true
```

The above option will set [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) to a learning rate of 1e-4 (scientific notation).

**Note**: The `fused: true` option can only be used with Adam and AdamW and is experimental. Some networks may not work properly when set to true.

---
### `scheduler`

This option sets the learning rate scheduler. Supported types are `MultiStepLR` and `CosineAnnealingLR`.

```
train:
    scheduler:
        type: multisteplr
        milestones: [60000, 120000]
        gamma: 0.5
```

The above option sets the MultiStepLR scheduler to reduce the learning by `gamma: 0.5` at iter counts of 60k and 120k.
For more information, see the [pytorch documentation](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html)

---
### `warmup_iter`

This option linearly ramps up the learning rate for the specified iter count. For example:

```
train:
    warmup_iter: 10000
```

If you start training with a learning rate of 1e-4 using the above option, the learning rate will start from 0 and increase to 1e-4 (linearly) when it reaches 10k iterations.

This technique is used to reduce overfitting when fine-tuning models.

---
### `total_iter`

Sets the total number of iterations for training. When the `total_iter` value is reached, the model will stop the training script and save the last models.

```
total_iter: 500000 # end of training will be 500k iter
```

---
### `ema_decay`

This option uses exponential moving averaging on the model weights. This is a common technique supposed to improve generalization.

```
ema_decay: 0.999
```

**Note**: ema_decay may cause conflicts, be advised.

---
### Losses

These options specify which losses to use and their respective parameters like weight.


---
### `pixel_opt`

The `pixel_opt` option defines the pixel loss.

```
train:
    pixel_opt:
        type: L1Loss
        loss_weight: 1.0
        reduction: mean
```

The above option sets pixel loss with L1 criteria and weight of 1.0.
Possible values for `type` are: `L1Loss`, `MSELoss` (also known as L2) and [`HuberLoss`](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss).

---
### `perceptual_opt`

This option sets the perceptual loss. It uses the VGG19 network to extract features from images.

```
train:
    perceptual_opt:
        type: PerceptualLoss
        layer_weights:
            "conv1_2": 0.1
            "conv2_2": 0.1
            "conv3_4": 1
            "conv4_4": 1
            "conv5_4": 1
        vgg_type: vgg19
        use_input_norm: true
        range_norm: false
        perceptual_weight: 1.0
        style_weight: 0
        criterion: l1
```

Possible values for `criterion` are: `l1`, `l2`, `huber` and `fro`.

---
### `gan_opt`

This option enables GAN training.

```
train:
    gan_opt:
        type: GANLoss
        gan_type: vanilla
        real_label_val: 1.0
        fake_label_val: 0.0
        loss_weight: 0.1
```

Possible values for `gan_type` are: `vanilla` or `lsgan`.

---
### `color_opt`

This option sets the color loss. In this loss, images are linearized from sRGB to Linear RGB and then converted to CIE XYZ. The X and Z channels (colors) are used to calculate the distance between GT and LQ.

```
train:
    color__opt:
        type: colorloss
        loss_weight: 1.0
        criterion: l1
```

Possible values for `criterion` are: `l1`, `l2` and `huber`.

---
### `ldl_opt`

This option sets the [LDL loss](https://github.com/csjliang/LDL). See the [research paper](https://arxiv.org/abs/2203.09195) for details.

```
train:
    ldl_opt:
        type: L1Loss
        loss_weight: 1.0
        reduction: mean
```

Possible values for `type` are: `L1Loss`, `MSELoss` and `HuberLoss`.

---
### `ff_opt`

This option sets the [Focal-Frequency Loss](https://github.com/EndlessSora/focal-frequency-loss). See the [research paper](https://arxiv.org/abs/2012.12821) for details.

```
train:
    ff_opt:
        type: focalfrequencyloss
        loss_weight: 1.0
```

---
### `net_d_iters` and `net_d_init_iters`

This option defines when to start and stop the discriminator.

The `net_d_init_iters` sets when the discriminator is enabled:

```
net_d_init_iters: 80000
```

The above option will start the discriminator at 80k iters.

The `net_d_iter` sets the total iter count for the discriminator.

```
net_d_iter: 500000
```

The above option will stop the discriminator at 500k iter count.

---
### Logger

These options describe the logger configuration.

---
### `print_freq`

This sets the terminal and log file printing of training information.

```
logger:
    print_freq: 100
```

The above option will print training information at every 100 iterations.

---
### `save_checkpoint_freq`

This option sets the frequency of saving model files and state file.

```
logger:
    save_checkpoint_freq: 1000
```

The above option will save models and state at every 1k iter count.

---
### `use_tb_logger` and `wandb`

This option enables to use [`tensorboard`](https://www.tensorflow.org/tensorboard/). A folder will be created inside `experiments/` containing files needed to initialize tensorboard.

```
logger:
    use_tb_logger: true
```

For details on using tensorboard, see the [documentation](https://www.tensorflow.org/tensorboard/).

Alternatively, you can use [`wandb`](https://wandb.ai/) by using the following option:

```
logger:
    use_tb_logger: true
    wandb:
        project: "experiments/tb_logger/project/"
        resume_id: 1
```

The option `use_tb_logger: true` is required to use `wandb`.

---
### Distributed Training

These options describe the distributed training configuration.

---
### `backend`

This option specifies the backend to use for distributed training.

```
dist_params:
    backend: nccl
    port: 29500
```

The above option will set up distributed training using the nvidia [`nccl`](https://developer.nvidia.com/nccl) library on port 29500.
You can also launch training with [`slurm`](https://slurm.schedmd.com/overview.html) by passing a command line argument:

```
pytorch train.py --launcher slurm -opt options.yml
```