import os
import shutil
from tqdm import *

path = './results/pix2pix/test_latest/images/'

fake_path = [x for x in os.listdir(path) if 'fake' in x]

real_path = [x for x in os.listdir(path) if 'real_B' in x]

fake_copy = './results/pix2pix/test_latest/fake/'
real_copy = './results/pix2pix/test_latest/real/'

if not os.path.exists(fake_copy):
    os.mkdir(fake_copy)

if not os.path.exists(real_copy):
    os.mkdir(real_copy)

for i in tqdm(range(len(fake_path))):
    shutil.copy(path + fake_path[i], fake_copy + fake_path[i].split('_')[0] + '.tif')
    shutil.copy(path + real_path[i], real_copy + fake_path[i].split('_')[0] + '.tif')
