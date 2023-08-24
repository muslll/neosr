# neosr

**neosr** is a framework for training real-world single-image super-resolution networks.

## installation

Requires Python =>3.11 and CUDA =>11.7
```
git clone https://github.com/muslll/neosr
cd neosr
```
Install [**`poetry`**](https://python-poetry.org/) (*recommended*), then run:
```
poetry install
```
Note: you need to use `poetry shell` to enter the env.

Alternatively, you can install through `pip` (**not recommended**):
```
pip install -e .
```

## quick start

Start training by running:

```
python train.py -opt options.yml
```
Where `options.yml` is a configuration file. You can find templates on [options](options/).

Please read the [Configuration Walkthrough]() for explanation on each option.

## datasets

If you don't already have a dataset, you can either download research datasets like [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or use one of the following:

...

*Note: those are not meant to be used in academic research*.

## features

### Supported archs:

| arch                                                                                              | option                                 |
|---------------------------------------------------------------------------------------------------|----------------------------------------|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)                                             | `esrgan`                               |
| [SRVGGNetCompact](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/srvgg_arch.py) | `compact`                              |
| [SwinIR](https://github.com/JingyunLiang/SwinIR)                                                  | `swinir_small`, `swinir_medium`        |
| [HAT](https://github.com/XPixelGroup/HAT)                                                         | `hat_s`, `hat_m`, `hat_l`              |
| [OmniSR](https://github.com/Francis0625/Omni-SR)                                                  | `omnisr`                               |
| [SRFormer](https://github.com/HVision-NKU/SRFormer)                                               | `srformer_light`, `srformer_medium`    |
| [DAT](https://github.com/zhengchen1999/dat)                                                       | `dat_light`, `dat_small`, `dat_medium` |

### Supported models:

| model                                                                  | option    |
|------------------------------------------------------------------------|-----------|
| Base model, supports both Generator and Discriminator                  | `default` |
| Builds on top of `default`, adding Real-ESRGAN on-the-fly degradations | `otf`     |

### Supported dataset loaders:

| loader                                          | option   |
|-------------------------------------------------|----------|
| Paired datasets                                 | `paired` |
| Single datasets (for inference, no GT required) | `single` |
| Real-ESRGAN on-the-fly degradation              | `otf`    |

### Supported losses:

| loss                                        | option               |
|---------------------------------------------|----------------------|
| L1 Loss                                     | `L1Loss`, `l1`       |
| L2 Loss                                     | `MSELoss`, `l2`      |
| Huber Loss                                  | `HuberLoss`, `huber` |
| Perceptual Loss                             | `PerceptualLoss`     |
| GAN                                         | `GANLoss`            |
| XYZ Color Loss                              | `colorloss`          |
| [LDL Loss](https://github.com/csjliang/LDL) | `ldl_opt`            |

## support

[**KoFi**]()

## license and acknowledgements

Released under [Apache license](license.txt).
This code was originally based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Check other licenses in [readme](license/readme.md).

Thanks to [victorca25/traiNNer](https://github.com/victorca25/traiNNer), [styler00dollar/Colab-traiNNer](https://github.com/styler00dollar/Colab-traiNNer/) and [timm](https://github.com/huggingface/pytorch-image-models) for providing helpful insights into some problems.

