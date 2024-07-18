import numpy as np
import os
import sys
import ntpath
import time
from . import util
import cv2

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def stretchImg_Bands(data, resultPath):
    H, W, C = data.shape
    if C > 2:
        out = np.zeros([H, W, C], dtype=np.uint8)
        for i in range(3):
            max = 2047
            t = (data[:, :, i]) / max * 255
            t[t < 0] = 0
            t[t > 255] = 255
            out[:, :, i] = t
    else:
        out = np.zeros([H, W, 1], dtype=np.uint8)
        max = 2047
        t = (data[:, :, 1]) / max * 255
        t[t < 0] = 0
        t[t > 255] = 255
        out[:, :, 0] = t[:, :]
    cv2.imwrite(resultPath, out)

def save_images(webpage, visuals, image_path, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []
    ims_dict = {}
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.tif' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

class Visualizer():
    def __init__(self, opt):
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False
        self.current_epoch = 0
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def display_current_results(self, visuals, epoch):
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = self.img_dir + '/epoch%.3d_%s.png' % (epoch, label)
            stretchImg_Bands(image_numpy, img_path)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
