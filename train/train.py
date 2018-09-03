__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import os
# third-party library
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from data_read import read_datas
from chinese_data_create import get_code_map

# Hyper Parameters
 # 训练整批数据多少次, 为了节约时间, 我们只训练一次
EPOCH = 300              # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # 学习率

class CNN(nn.Module):
    def __init__(self,out_size,width,height):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, out_size)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization

def train_model(out_size,width,height,datas,labels,pkl_name,torch_datas,torch_labels):
    cnn = CNN(out_size,width,height)
    print(cnn)  # net architecture
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    for epoch in range(EPOCH):
        output = cnn(torch_datas)
        loss = loss_func(output,torch_labels)
        print('epoch=%d loss=%0.4f'%(epoch,loss))
        if loss < 0.05:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    torch.save(cnn,pkl_name)
    return cnn

if __name__ == "__main__":
    code_map,width,height = get_code_map()
    datas,labels = read_datas(width,height)
    out_size = max(code_map.keys())+1
    torch_datas = torch.from_numpy(datas)
    torch_labels = torch.from_numpy(labels)

    pkl_name = format("chinese_character_%d.pkl"%out_size)

    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name)
    else:
        cnn = train_model(out_size,width,height,datas,labels,pkl_name,torch_datas,torch_labels)

    output = cnn(torch_datas)
    print("result:")

    def get_max_index(row):
        max_value = -99999999.0
        res = 0;
        i = 0
        for a in row:
            if a > max_value:
                max_value = a
                res = i
            i = i+1
        return res

    count = len(labels)
    for i in range(count):
        max_index=get_max_index(output[i])
        s = format("%s-%s"%(code_map[labels[i]],code_map[max_index]))
        print(s)

