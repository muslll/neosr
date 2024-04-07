import argparse
import os
import random

from os import path as osp
from collections import OrderedDict

import yaml
import torch

from neosr.utils import set_random_seed
from neosr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser(prog='neosr',
                        usage=argparse.SUPPRESS,
                        description='''-------- neosr command-line options --------''')

    parser._optionals.title = 'training and inference'
    
    parser.add_argument('-opt', type=str, required=False,
                        help='Path to option YAML file.')

    parser.add_argument( '--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                        help='job launcher')
    
    parser.add_argument('--auto_resume', action='store_true', default=False)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--force_yml', nargs='+', default=None,
                        help='Force to update yml files. Examples: train:total_iter=200000')

    # Options for convert.py script
   
    group = parser.add_argument_group('model conversion')

    group.add_argument('--input', type=str, required=False,
                        help='Input Pytorch model path.')

    group.add_argument('-onnx', '--onnx', action='store_true',
                        help='Enables ONNX conversion.', default=False)

    group.add_argument('-safetensor', '--safetensor', action='store_true',
                        help='Enables safetensor conversion.', default=False)

    group.add_argument('-net', '--network', type=str,
                        required=False, help='Generator network.')

    group.add_argument('-s', '--scale', type=int,
                        help='Model scale ratio.', default=4)

    group.add_argument('-window', '--window', type=int,
                        help='Model scale ratio.', default=None)

    group.add_argument('-opset', '--opset', type=int,
                        help='ONNX opset. (default: 17)', default=17)

    group.add_argument('-static', '--static', type=int, nargs=3,
                        help='Set static shape for ONNX conversion. Example: -static "3,640,640".', default=None)

    group.add_argument('-nocheck', '--nocheck', action='store_true',
                        help='Disables checking against original pytorch model on ONNX conversion.', default=False)

    group.add_argument('-fp16', '--fp16' , action='store_true',
                        help='Enable half-precision. (default: false)', default=False)

    group.add_argument('-optimize', '--optimize', action='store_true',
                        help='Run ONNX optimizations', default=False)

    group.add_argument('-fulloptimization', '--fulloptimization', action='store_true',
                        help='Run full ONNX optimizations', default=False)

    group.add_argument('--output', type=str, required=False,
                        help='Output ONNX model path.', default=root_path)


    args = parser.parse_args()

    # error if no config file exists
    if args.input is None and not osp.exists(args.opt):
        msg = "Didn't get a config! Please link the config file using -opt /path/to/config.yml"
        raise ValueError(msg) 

    if args.input is None:
        # parse yml to dict
        opt = yaml_load(args.opt)

        # distributed settings
        if args.launcher == 'none':
            opt['dist'] = False
        else:
            opt['dist'] = True
            if args.launcher == 'slurm' and 'dist_params' in opt:
                init_dist(args.launcher, **opt['dist_params'])
            else:
                init_dist(args.launcher)
        opt['rank'], opt['world_size'] = get_dist_info()

        # random seed
        seed = opt.get('manual_seed')
        if seed is None:
            opt["deterministic"] = False
            seed = random.randint(1024, 10000)
            opt['manual_seed'] = seed
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 0
        else:
            # Determinism
            opt["deterministic"] = True
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        set_random_seed(seed + opt['rank'])

        # force to update yml options
        if args.force_yml is not None:
            for entry in args.force_yml:
                # now do not support creating new keys
                keys, value = entry.split('=')
                keys, value = keys.strip(), value.strip()
                value = _postprocess_yml_value(value)
                eval_str = 'opt'
                for key in keys.split(':'):
                    eval_str += f'["{key}"]'
                eval_str += '=value'
                # using exec function
                exec(eval_str)

        opt['auto_resume'] = args.auto_resume
        opt['is_train'] = is_train

        # debug setting
        if args.debug and not opt['name'].startswith('debug'):
            opt['name'] = 'debug_' + opt['name']

        if opt.get('num_gpu', 'auto') == 'auto':
            opt['num_gpu'] = torch.cuda.device_count()

        # datasets
        for phase, dataset in opt['datasets'].items():
            # for multiple datasets, e.g., val_1, val_2; test_1, test_2
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            if 'scale' in opt:
                dataset['scale'] = opt['scale']
            if dataset.get('dataroot_gt') is not None:
                dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
            if dataset.get('dataroot_lq') is not None:
                dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

        # paths
        for key, val in opt['path'].items():
            if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
                opt['path'][key] = osp.expanduser(val)

        if is_train:
            experiments_root = opt['path'].get('experiments_root')
            if experiments_root is None:
                experiments_root = osp.join(root_path, 'experiments')
            experiments_root = osp.join(experiments_root, opt['name'])

            opt['path']['experiments_root'] = experiments_root
            opt['path']['models'] = osp.join(experiments_root, 'models')
            opt['path']['training_states'] = osp.join(
                experiments_root, 'training_states')
            opt['path']['log'] = experiments_root
            opt['path']['visualization'] = osp.join(
                experiments_root, 'visualization')

            # change some options for debug mode
            if 'debug' in opt['name']:
                if 'val' in opt:
                    opt['val']['val_freq'] = 8
                opt['logger']['print_freq'] = 1
                opt['logger']['save_checkpoint_freq'] = 8
        else:  # test
            results_root = opt['path'].get('results_root')
            if results_root is None:
                results_root = osp.join(root_path, 'experiments', 'results')
            results_root = osp.join(results_root, opt['name'])

            opt['path']['results_root'] = results_root
            opt['path']['log'] = results_root
            opt['path']['visualization'] = results_root
    else:
        opt = None

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(
            0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
