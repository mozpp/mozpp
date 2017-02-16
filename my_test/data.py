from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale

from dataset import DatasetFromFolder

root_dir='/media/b3-542/Library/pytorch'

def input_transform():
    return Compose([ToTensor()])

def target_transform():
    return Compose([ToTensor()])

def get_training_set():
    train_dir = join(root_dir, "train")

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform()
                             #target_transform=target_transform()
                             )