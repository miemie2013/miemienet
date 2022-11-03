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
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True):
        super(Res, self).__init__()
        self.fc = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=bias)
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

class Model3(nn.Module):
    def __init__(self, backbone, neck, yolo_head=None):
        super(Model3, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head

    def forward(self, x, scale_factor=None, targets=None):
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.yolo_head(fpn_feats, targets)
        y = out[1]
        return y


torch.manual_seed(15)
fuse_conv_bn = True

num_classes = 80
# miemienet_image_data_format = "NCHW"
miemienet_image_data_format = "NHWC"

test_name = '002'
os.makedirs('save_data', exist_ok=True)



layers=[3, 6, 6, 3]
channels=[64, 128, 256, 512, 1024]
act='swish'
return_idx=[1, 2, 3]
depth_wise=False
use_large_stem=True
width_mult=0.5
depth_mult=0.33
freeze_at=-1


in_channels = [int(256 * width_mult), int(512 * width_mult), int(1024 * width_mult)]
out_channels = [768, 384, 192]
stage_num = 1
block_num = 3
act = 'swish'
spp = True

num_classes = 80
test_size = [640, 640]
head_cfg = dict(
    in_channels=[int(768 * width_mult), int(384 * width_mult), int(192 * width_mult)],
    fpn_strides=[32, 16, 8],
    grid_cell_scale=5.0,
    grid_cell_offset=0.5,
    static_assigner_epoch=100,
    use_varifocal_loss=True,
    num_classes=num_classes,
    loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, },
    eval_size=test_size,
)

backbone = CSPResNet(layers, channels, act, return_idx, depth_wise, use_large_stem, width_mult, depth_mult, freeze_at)
neck = CustomCSPPAN(in_channels=in_channels, out_channels=out_channels, stage_num=stage_num, block_num=block_num, act=act, spp=spp, depth_mult=depth_mult, width_mult=width_mult)
head = PPYOLOEHead(**head_cfg)
model = Model3(backbone, neck, head)

# 权重来自miemiedetection，第一次先跑这个，连同全连接层的权重一起保存
ckpt = torch.load("ppyoloe_crn_s_300e_coco.pth", map_location="cpu")
model = load_ckpt(model, ckpt["model"])

model_std = model.state_dict()

print('save ...')
save_weights_as_miemienet("save_data/%s-model" % (test_name, ), model_std, miemienet_image_data_format, fuse_conv_bn, 1e-5)
print('saved!')

print('bbbbbbbbbbb')
input_size = 640

model.eval()
model_std = model.state_dict()
# x = torch.randn([batch_size, in_features, input_size, input_size])

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
    # save_as_txt("save_data/%s-eval-y.txt" % (test_name, ), y.permute((0, 2, 3, 1)))
    save_as_txt("save_data/%s-eval-y.txt" % (test_name, ), y.permute((0, 2, 1)))

print('save ...')
save_weights_as_miemienet("save_data/%s-modelfinal" % (test_name, ), model.state_dict(), miemienet_image_data_format, fuse_conv_bn, 1e-5)
print('saved!')

print()
