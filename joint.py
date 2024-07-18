from PIL import Image
from osgeo import gdal
import numpy as np
from tqdm import *


list1 = ["byte", "uint8", "uint16", "int16", "uint32", "int32", "float32", "float64", "cint16", "cint32", "cfloat32", "cfloat64"]
list2 = [gdal.GDT_Byte, gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32,
         gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]

def imgread(path):
    img = gdal.Open(path)
    c = img.RasterCount
    img_arr = img.ReadAsArray()      # 读取整幅图像
    if c > 1:
        img_arr = img_arr.swapaxes(1, 0)
        img_arr = img_arr.swapaxes(2, 1)   # H, W, C
    else:
        img_arr = img_arr
    del img
    return img_arr.reshape(img_arr.shape[0], img_arr.shape[1], -1)

def imgwrite(path, narray):
    s = narray.shape
    narray = narray.reshape((s[0], s[1], -1))
    s = narray.shape
    dt_name = narray.dtype.name
    name = list(set(list1) & set([dt_name.lower()])) + ["byte"]
    datatype = list2[list1.index(name[0])]

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(path, s[1], s[0], s[2], datatype)

    for i in range(s[2]):
        dataset.GetRasterBand(i + 1).WriteArray(narray[:, :, i])
    del dataset
    # return dataset

def read_img(datapath, scale=2047):
    img = imgread(datapath)
    img = img[:, :, :]
    img[img > scale] = scale
    return img


def joint_uint8():
    images = []
    for i in range(0, 3021):
        image_path = f"./results/pix2pix/test_latest/FID/fake/{i + 1}.png"
        image = Image.open(image_path)
        images.append(image)

    result_image = Image.new("RGB", (13568, 14592))
    # result_image = Image.new("RGB", (17912, 19902))

    with open('./results/pix2pix/test_latest/xys.txt', 'r') as f:
        xywhs = f.readlines()

    for i in range(len(xywhs)):
        x = int(xywhs[i].split(' ')[0])
        y = int(xywhs[i].split(' ')[1])

        result_image.paste(images[i], (x, y))

    result_image.save(r"./results/pix2pix/test_latest/fake.png")

def joint_uint16():
    images = []
    for i in range(0, 3021):
        image_path = f"./results/pix2pix/test_latest/real/{i + 1}.tif"
        image = read_img(image_path)
        images.append(image)

    result_image = np.zeros([14592, 13568, 3], dtype=np.uint16)

    with open('./results/pix2pix/test_latest/xys.txt', 'r') as f:
        xywhs = f.readlines()

    for i in tqdm(range(len(xywhs))):
        x = int(xywhs[i].split(' ')[0])
        y = int(xywhs[i].split(' ')[1])

        result_image[y: y + 256, x: x + 256, :] = images[i]

    imgwrite(r"./results/pix2pix/test_latest/real-uint16.tif", result_image)

if __name__ == '__main__':
    joint_uint16()

