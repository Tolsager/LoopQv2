import random
import numpy as np
import os
import torch
from config import *


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def check_paths(infer=False):
    if infer:
        paths = [CSV_TEST, IMAGE_DIRECTORY_TEST]
    else:
        paths = [CSV_TRAIN, IMAGE_DIRECTORY_TRAIN]
    print("Checking that paths are set correctly\n")

    path_exists = True
    for path in paths:
        if not os.path.exists(path):
            print(f"Path: {path}\ndoes not exist\n")
            path_exists = False
    if not path_exists:
        print("Please update the paths in the 'config.py' file.")
        input("Once the paths have been updated, run the script again.")
        print("Exiting...")
        exit()
