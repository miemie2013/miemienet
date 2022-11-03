import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import cv2
import time



from test2_utils import save_as_txt, save_weights_as_txt, read_weights_from_miemienet, load_ckpt




class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True, groups=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=bias, groups=groups)
        if bias:
            torch.nn.init.normal_(self.conv.bias, 0., 1.)
        # self.act = nn.LeakyReLU(0.33)

    def __call__(self, x):
        y = self.conv(x)
        # y = self.act(y)
        return y




test_name = '00004'


# batch_size = 8
# input_size = 256
# in_features = 32
# out_features = 32
# kernel_size = 3
# groups = 1

batch_size = 8
input_size = 256
in_features = 32
out_features = 32
kernel_size = 3
groups = 32


stride = 1
padding = (kernel_size - 1) // 2
bias = True


model = Model(in_features, out_features, kernel_size, stride, padding, bias, groups)
model.eval()



x_arr = np.zeros((batch_size * input_size * input_size * in_features, ), dtype=np.float32)
x_arr = read_weights_from_miemienet('save_data/%s-x.bin' % test_name, x_arr)
x_arr = np.reshape(x_arr, (batch_size, input_size, input_size, in_features))
x_arr = np.transpose(x_arr, (0, 3, 1, 2))

y_arr = np.zeros((batch_size * input_size * input_size * out_features, ), dtype=np.float32)
y_arr = read_weights_from_miemienet('save_data/%s-y.bin' % test_name, y_arr)
y_arr = np.reshape(y_arr, (batch_size, input_size, input_size, out_features))
y_arr = np.transpose(y_arr, (0, 3, 1, 2))


w_arr = np.zeros((kernel_size * kernel_size * in_features * out_features // groups, ), dtype=np.float32)
w_arr = read_weights_from_miemienet('save_data/%s-w.bin' % test_name, w_arr)
w_arr = np.reshape(w_arr, (kernel_size, kernel_size, in_features // groups, out_features))
w_arr = np.transpose(w_arr, (3, 2, 0, 1))



b_arr = np.zeros((out_features, ), dtype=np.float32)
b_arr = read_weights_from_miemienet('save_data/%s-b.bin' % test_name, b_arr)




model_std = model.state_dict()
model_std['conv.weight'] = torch.from_numpy(w_arr)
model_std['conv.bias'] = torch.from_numpy(b_arr)
model.load_state_dict(model_std)


x = torch.from_numpy(x_arr)

x.requires_grad_(True)

for batch_idx in range(10):
    start_time = time.time()
    y = model(x)
    cost = time.time() - start_time
    print('eval forward cost_time: {0:.3f} ms'.format(cost * 1000.))
    y_torch = y.cpu().detach().numpy()
    ddd = np.sum((y_torch - y_arr) ** 2)
    print('diff=%.6f (%s)' % (ddd, 'y'))

print()





