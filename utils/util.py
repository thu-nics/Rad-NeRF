import torch
import os
import sys
import logging
import datetime

### ckpt-related ###
def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']


### log-related ###
global_logger = None

def init_global_logger(filename, verbosity=1, name=None):
    global global_logger
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()

    logging.Formatter.converter = beijing

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s]%(message)s", datefmt='%Y/%m/%d-%H:%M:%S')
    
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    logger.propagate = False

    if (logger.hasHandlers()):
        logger.handlers.clear()
    # 写入文件
    fh = logging.FileHandler(filename, "a", encoding='utf-8', delay=True)
    fh.setFormatter(formatter)
    fh.setLevel(level_dict[verbosity])
    logger.addHandler(fh)
    # 终端显示
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setFormatter(formatter)
    # sh.setLevel(level_dict[verbosity])
    # logger.addHandler(sh)

    global_logger = logger
    return global_logger


def get_global_logger():
    DEFAULT_LOG_PATH = '/tmp/nerf_log.txt'

    global global_logger
    if global_logger is not None:
        return global_logger
    else:
        print('The global logger is not initialized. \nInitialize the global logger with default parameters.')
        print('default log path:', DEFAULT_LOG_PATH)
        init_global_logger(DEFAULT_LOG_PATH)
        return global_logger