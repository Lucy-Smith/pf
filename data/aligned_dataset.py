import os
import torch
from data.gdaldiy import read_img
import torch.utils.data as data
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod    # 静态
    def modify_commandline_options(parser, is_train):   # 不用实例化，没有self
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'A')  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'B')  # get the image directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = read_img(A_path)
        A = A[:, :, :self.input_nc]
        B = read_img(B_path)
        B = B[:, :, :self.output_nc]

        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)

        torch.set_printoptions(precision=16)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)


