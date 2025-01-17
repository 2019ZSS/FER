import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np


seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import models
from models import segmentation


def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training
    # from trainers.fer2013_trainer import FER2013Trainer
    from trainers.tta_trainer import FER2013Trainer

    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    image_size = configs['image_size']
    image_size = (image_size, image_size)
    base_path_to_FER_plus = configs['data_path']
    max_loaded_images_per_label = configs['max_loaded_images_per_label'] if 'max_loaded_images_per_label' in configs else 1000
    from utils.datasets.ferplus_dataset import FERPlus
    from torch.utils.data import DataLoader
    data_sets = []
    for i in range(3):
        data_sets.append(FERPlus(idx_set=i, image_size=image_size, 
                        base_path_to_FER_plus=base_path_to_FER_plus, 
                        max_loaded_images_per_label=max_loaded_images_per_label))
    return data_sets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/fer2013_config.json", help='model config path')
    opt = parser.parse_args()
    config_path = opt.config_path
    config_path = config_path.strip('\'').strip('\"')
    main(config_path)