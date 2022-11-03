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
    def __init__(self):
        super(Model, self).__init__()
        self.act = nn.LeakyReLU(0.33)

    def __call__(self, x):
        y = self.act(x)
        return y




test_name = '00006'



# H = 256*256
# W = 256

H = 256
W = 256*256

# H = 4096
# W = 4096


model = Model()
model.eval()



x_arr = np.zeros((H * W, ), dtype=np.float32)
x_arr = read_weights_from_miemienet('save_data/%s-x.bin' % test_name, x_arr)
x_arr = np.reshape(x_arr, (H, W))
# x_arr = np.transpose(x_arr, (0, 3, 1, 2))

y_arr = np.zeros((H * W, ), dtype=np.float32)
y_arr = read_weights_from_miemienet('save_data/%s-y.bin' % test_name, y_arr)
y_arr = np.reshape(y_arr, (H, W))
# y_arr = np.transpose(y_arr, (0, 3, 1, 2))



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





