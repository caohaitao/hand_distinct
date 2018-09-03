__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import torch.nn as nn
import os

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
        x = x.view(1, -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization

if __name__ == "__main__":
    number = 20
    pkl_name = format("chinese_character_%d.pkl"%number)
    proto_name = format("hand%d.proto"%number)
    param_name = format("hand%d.param"%number)
    bin_name = format("hand%d.bin"%number)

    if os.path.exists(pkl_name):
        cnn = torch.load(pkl_name)
        dummy_init = Variable(torch.randn(1,1,28,28))
        torch.onnx.export(cnn,dummy_init,proto_name,verbose=True)
        print("export onnx success,file name(%s)"%(proto_name))
        if os.path.exists("onnx2ncnn.exe"):
            os.system("onnx2ncnn.exe %s %s %s"%(proto_name,param_name,bin_name))
            print("onnx2ncnn success,param(%s) bin(%s)"%(param_name,bin_name))
        else:
            print("onnx2ncnn failed")
    else:
        print("don't find file(%s)"%pkl_name)
    print("over")

