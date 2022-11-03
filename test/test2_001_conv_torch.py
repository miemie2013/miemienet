import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import cv2
import time

from cspresnet import CSPResNet
from custom_pan import CustomCSPPAN
from ppyoloe_head import PPYOLOEHead
from resnet import ConvNormLayer, BottleNeck, ResNet
from test2_utils import save_as_txt, save_weights_as_txt, save_weights_as_miemienet, load_ckpt


def swish(x):
    return x * torch.sigmoid(x)


class Res(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True, groups=1):
        super(Res, self).__init__()
        self.fc = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=bias, groups=groups)
        if bias:
            torch.nn.init.normal_(self.fc.bias, 0., 1.)
        # self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.33)
        # torch.nn.init.normal_(self.bn.weight, 0., 1.)
        # torch.nn.init.normal_(self.bn.bias, 0., 1.)

    def __call__(self, x):
        y = self.fc(x)
        # y = self.bn(y)
        y = self.act(y)
        y = torch.sigmoid(y)
        return y + x
        # return y

class Model(nn.Module):
    def __init__(self, in_features, out_features, num_classes, bias=True):
        super(Model, self).__init__()
        self.res0 = Res(in_features, out_features, 3, 1, 1, bias, groups=out_features)
        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.res1 = Res(out_features, out_features, 1, 1, 0, bias, groups=1)
        # self.fc = nn.Linear(out_features, num_classes, True)

    def __call__(self, x):
        y = self.res0(x)

        # y = self.avgpool(y)

        # y = F.avg_pool2d(y, kernel_size=3, stride=2, padding=1)
        y = F.max_pool2d(y, kernel_size=3, stride=2, padding=1)
        y = self.res1(y)
        # y = F.softmax(y, dim=1)
        # aaa = y.permute((0, 1, 3, 2))
        # y = torch.cat([aaa, y, aaa, y], 1)
        # y = F.interpolate(y, scale_factor=2.)
        # y = F.adaptive_avg_pool2d(y, (1, 1))

        # 全局平均池化
        # y = F.adaptive_avg_pool2d(y, (1, 1)).squeeze(3).squeeze(2)
        # 全局平均池化另一种写法
        # y = y.mean([2, 3])

        # logits = self.fc(y)
        # return logits
        return y


torch.manual_seed(15)
fuse_conv_bn = True

batch_size = 8
in_features = 32
out_features = 32
bias = True
# bias = False
num_classes = 5
# miemienet_image_data_format = "NCHW"
miemienet_image_data_format = "NHWC"

test_name = '001'
os.makedirs('save_data', exist_ok=True)

model = Model(in_features, out_features, num_classes, bias)

model_std = model.state_dict()

print('save ...')
save_weights_as_miemienet("save_data/%s-model" % (test_name, ), model_std, miemienet_image_data_format, fuse_conv_bn, 1e-5)
print('saved!')

print('bbbbbbbbbbb')
# input_size = 2
input_size = 64

model.eval()
model_std = model.state_dict()
x = torch.randn([batch_size, in_features, input_size, input_size])
'''
im = cv2.imread('000000000019.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_shape = im.shape
im_scale_x = float(input_size) / float(im_shape[1])
im_scale_y = float(input_size) / float(im_shape[0])
im = cv2.resize(
    im,
    None,
    None,
    fx=im_scale_x,
    fy=im_scale_y,
    interpolation=2)

mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]
im = im.astype(np.float32, copy=False)
mean = np.array(mean)[np.newaxis, np.newaxis, :]
std = np.array(std)[np.newaxis, np.newaxis, :]
im -= mean
im /= std

im = np.swapaxes(im, 1, 2)
im = np.swapaxes(im, 1, 0)
x = torch.from_numpy(im).to(torch.float32)
x = torch.unsqueeze(x, 0)
'''

x.requires_grad_(True)

for batch_idx in range(8):
    start_time = time.time()
    y = model(x)
    cost = time.time() - start_time
    print('eval forward cost_time: {0:.3f} ms'.format(cost * 1000.))


if miemienet_image_data_format == "NCHW":
    save_as_txt("save_data/%s-eval-x.txt" % (test_name, ), x)
    save_as_txt("save_data/%s-eval-y.txt" % (test_name, ), y)
elif miemienet_image_data_format == "NHWC":
    save_as_txt("save_data/%s-eval-x.txt" % (test_name, ), x.permute((0, 2, 3, 1)))
    save_as_txt("save_data/%s-eval-y.txt" % (test_name, ), y.permute((0, 2, 3, 1)))

print('save ...')
save_weights_as_miemienet("save_data/%s-modelfinal" % (test_name, ), model.state_dict(), miemienet_image_data_format, fuse_conv_bn, 1e-5)
print('saved!')

print()
