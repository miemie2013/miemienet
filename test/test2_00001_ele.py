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



test_name = '00001'
os.makedirs('save_data', exist_ok=True)



N = 2
C = 128
H = 64
W = 64


N = 8
C = 128
H = 128
W = 128
R = 7


im2col = torch.randn([N*C*H*W + R]).to(torch.float32)
weight = torch.randn([N*C*H*W + R]).to(torch.float32)


_im2col = im2col.cpu().detach().numpy()
_weight = weight.cpu().detach().numpy()
_im2col = _im2col.astype(np.float32)
_weight = _weight.astype(np.float32)

for batch_idx in range(8):
    start_time = time.time()
    add_result = im2col + weight
    cost = time.time() - start_time
    print('torch cost_time: {0:.3f} ms'.format(cost * 1000.))

for batch_idx in range(8):
    start_time = time.time()
    add_result2 = _im2col + _weight
    cost = time.time() - start_time
    print('numpy cost_time: {0:.3f} ms'.format(cost * 1000.))

for batch_idx in range(8):
    start_time = time.time()
    mul_result = im2col * weight
    cost = time.time() - start_time
    print('torch cost_time: {0:.3f} ms'.format(cost * 1000.))

for batch_idx in range(8):
    start_time = time.time()
    mul_result2 = _im2col * _weight
    cost = time.time() - start_time
    print('numpy cost_time: {0:.3f} ms'.format(cost * 1000.))



v1 = add_result.cpu().detach().numpy()
v2 = add_result2
ddd = np.sum((v1 - v2) ** 2)
print('diff=%.6f (%s)' % (ddd, 'y'))


v1 = mul_result.cpu().detach().numpy()
v2 = mul_result2
ddd = np.sum((v1 - v2) ** 2)
print('diff=%.6f (%s)' % (ddd, 'y'))

# mul_result = mul_result * 0.0 + 3.3
save_as_txt("save_data/%s-x.txt" % (test_name, ), im2col)
save_as_txt("save_data/%s-w.txt" % (test_name, ), weight)
save_as_txt("save_data/%s-y.txt" % (test_name, ), add_result)
save_as_txt("save_data/%s-z.txt" % (test_name, ), mul_result)

print()
