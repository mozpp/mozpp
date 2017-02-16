from __future__ import print_function
import os.path
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import Net
from data import get_training_set

cuda=True
cudnn.benchmark=True

#continune trainning?
#resume=False
resume='/media/b3-542/Library/pytorch/my_test/model_epoch_1000.pth'

print('===> Loading datasets')
train_set = get_training_set()
training_data_loader = DataLoader(dataset=train_set)
print('===> Building model')
model = Net()

if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            #start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint.state_dict())
            print("=> loaded checkpoint  {}"
                  .format( resume))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

criterion = nn.MSELoss()

#if cuda:
model = model.cuda()
criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, 1000 + 1):
    train(epoch)
    #test()
    if epoch%100==0:
        checkpoint(epoch)