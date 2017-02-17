from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np


input_image='/media/b3-542/Library/dataset_fenghuo/image3/445100000013225524181_start_2016-12-20_12_53_46.avi_000003.561.jpg'
model='/media/b3-542/Library/pytorch/my_test/model_epoch_2000.pth'
cuda=True

img = Image.open(input_image)
img=img.resize((59,59))

model = torch.load(model)
input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])

if cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)

print(out)