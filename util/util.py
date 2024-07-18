"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os
from data.gdaldiy import imgwrite

def tensor2im(input_image, imtype=np.uint16):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().data.numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 2047.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    imgwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
