import tifffile
import glob
import numpy as np
# from PIL import Image
import threading
import os
import cv2
from osgeo import gdal


def stretchImg_RGB(imgPath, resultPath): #low越大越暗，high越小越亮
    data = gdal.Open(imgPath)
    data = data.ReadAsArray()
    data = data.swapaxes(1, 0)
    data = data.swapaxes(2, 1)
    H, W, C = data.shape
    out = np.zeros([H, W, 3], dtype=np.uint8)
    for i in range(3):   # 处理可见光s2，改为for i in range(1, 4),表示选用哨兵2的B2,B3,B4波段
        max = 2047
        t = data[:, :, i] / max * 255
        t[t < 0] = 0
        t[t > 255] = 255
        out[:, :, i] = t   # 可见光填i-1
    cv2.imwrite(resultPath, out)

if __name__ == '__main__':
    path = './results/pix2pix/test_latest/images/'

    ori_imgs = os.listdir(path)

    if not os.path.exists('./results/pix2pix/test_latest/uint8'):
        os.mkdir("./results/pix2pix/test_latest/uint8")

    for ori_img in ori_imgs:
        if 'B' in ori_img:
            save_img = './results/pix2pix/test_latest/uint8/' + ori_img.split('.')[0] + '.png'
            stretchImg_RGB(path + ori_img, save_img)
