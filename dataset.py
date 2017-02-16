import torch.utils.data as data
import torch
from os import listdir
from os.path import join
import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_txt(index):
    f = open('/media/b3-542/Library/Label-Free Supervision test/data_inter.txt', 'r')
    s = f.readlines()
    a = s[index].split(',')
    for index, item in enumerate(a):
        a[index] = int(item)
    return torch.Tensor(a[1:5])


def find_image(image_dir):
    image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
    image_filenames.sort()
    return image_filenames


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = find_image(image_dir)

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = Image.open(self.image_filenames[index])
        input = input.resize((59, 59))
        target = load_txt(index)
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)