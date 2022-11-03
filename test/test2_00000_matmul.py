import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import cv2
import time

from resnet import ConvNormLayer, BottleNeck, ResNet
from test2_utils import save_as_txt, save_weights_as_txt, save_weights_as_miemienet, load_ckpt


'''
im2col.shape = [N * out_H * out_W, in_C * kH * kW]
weight.shape = [out_C, in_C * kH * kW]

out = im2col * weight
out.shape = [N * out_H * out_W, out_C]

https://blog.csdn.net/m0_66201040/article/details/124329647
'''

test_name = '00000'
os.makedirs('save_data', exist_ok=True)

batch_size = 2
output_size = 256
in_features = 64
kH = 3
kW = 3
out_features = 128


M = batch_size * output_size * output_size
K = in_features * kH * kW
N = out_features


# M = 640
# K = 608
# N = 512


print("M = %d" % M)
print("K = %d" % K)
print("N = %d" % N)
print()
print()


im2col = torch.randn([M, K]).to(torch.float32)
weight = torch.randn([K, N]).to(torch.float32)
weight_t = weight.t()


_im2col = im2col.cpu().detach().numpy()
_weight = weight.cpu().detach().numpy()
_weight_t = weight_t.cpu().detach().numpy()
_im2col = _im2col.astype(np.float32)
_weight = _weight.astype(np.float32)
_weight_t = _weight_t.astype(np.float32)

for batch_idx in range(8):
    start_time = time.time()
    yyyyy1 = im2col.matmul(weight)   # pytorch cpu的结果是 110 ms 左右(3060笔记本win11 ，不充电的情况下)。
    cost = time.time() - start_time
    print('matmul cost_time: {0:.3f} ms'.format(cost * 1000.))

for batch_idx in range(8):
    start_time = time.time()
    yyyyy2 = F.linear(im2col, weight_t)
    cost = time.time() - start_time
    print('linear cost_time: {0:.3f} ms'.format(cost * 1000.))

for batch_idx in range(8):
    start_time = time.time()
    yyyyy3 = np.matmul(_im2col, _weight)   # numpy 的结果是 40 ms 左右(3060笔记本win11 ，不充电的情况下)。numpy 牛逼
    cost = time.time() - start_time
    print('numpy matmul cost_time: {0:.3f} ms'.format(cost * 1000.))


v1 = yyyyy1.cpu().detach().numpy()
v2 = yyyyy2.cpu().detach().numpy()
v3 = yyyyy3
ddd = np.sum((v1 - v2) ** 2)
print('diff=%.6f (%s)' % (ddd, 'y'))
ddd = np.sum((v1 - v3) ** 2)
print('diff=%.6f (%s)' % (ddd, 'y'))


save_as_txt("save_data/%s-x.txt" % (test_name, ), im2col)
save_as_txt("save_data/%s-w.txt" % (test_name, ), weight)
save_as_txt("save_data/%s-wt.txt" % (test_name, ), weight_t)
save_as_txt("save_data/%s-y.txt" % (test_name, ), yyyyy1)

print()
