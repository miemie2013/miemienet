import numpy as np
import os
import time
from SIMD_tools import _mm256_setzero_ps, _mm256_loadu_ps, _mm256_fmadd_ps, _mm256_broadcast_ss, _mm256_storeu_ps
from test2_utils import save_ndarray_as_txt


'''
用python模拟SIMD的矩阵乘法。
SIMD的寄存器，每次

'''

test_name = '00002'
os.makedirs('save_data', exist_ok=True)



batch_size = 16
ch_in = 8
ch_out = 24 + 7

batch_size = 8400
ch_in = 512
ch_out = 512 + 7

# batch_size = 2
# ch_in = 3
# ch_out = 4

np.random.seed(13)

x = np.random.random((batch_size, ch_in)).astype(np.float32)
w = np.random.random((ch_out, ch_in)).astype(np.float32)


# im2col = [-0.959000, -0.533000,]
# weight = [-0.666000, -0.500000,0.169000,0.724000,0.478000,0.358000,-0.038000,-0.536000,0.705000,-0.855000,0.281000,-0.173000,0.961000,-0.509000,-0.005000,0.942000,-0.173000,0.436000]
# im2col = np.array(im2col).astype(np.float32)
# weight = np.array(weight).astype(np.float32)
# im2col = np.reshape(im2col, (batch_size, ch_in))
# weight = np.reshape(weight, (ch_in, ch_out))
# x = np.copy(im2col)
# w = np.transpose(np.copy(weight))


w_t = np.transpose(w)

y2 = np.matmul(x, w_t)

# save_ndarray_as_txt("save_data/%s-x.txt" % (test_name, ), x)
# save_ndarray_as_txt("save_data/%s-w.txt" % (test_name, ), w_t)
# save_ndarray_as_txt("save_data/%s-y.txt" % (test_name, ), y2)

for batch_idx in range(8):
    start_time = time.time()
    y2 = np.matmul(x, w_t)
    cost = time.time() - start_time
    print('numpy cost_time: {0:.3f} ms'.format(cost * 1000.))



# SIMD求的输出
y = np.zeros((batch_size, ch_out)).astype(np.float32)

# 压成1维数组,模拟SIMD
x = np.reshape(x, (-1, ))
w = np.reshape(w, (-1, ))
w_t = np.reshape(w_t, (-1, ))
y = np.reshape(y, (-1, ))

elempack = 8

# naive
# for bs in range(0, batch_size, 1):
#     for oc in range(0, ch_out, 1):
#         for ic in range(0, ch_in, 1):
#             y[bs * ch_out + oc] += x[bs * ch_in + ic] * w[oc * ch_in + ic]


'''
 C  = A * B
3x4  3xn * nx4

a a a ...               b b b b          c c c c
a a a ...        *      b b b b        = c c c c
a a a ...               b b b b          c c c c
                        . . . .
                        . . . .
                        . . . .
换个角度思考
假设C初始化为全0,
取左矩阵A的1个元素a00, 取右矩阵B的1行(b00, b01, b02, b03), a分别与R里每个元素相乘,
a00 * b00累加到c00
a00 * b01累加到c01
...
即各自结果累加到矩阵C的1行(c00, c01, c02, c03)

按照这样思考, 可以提高cache命中率(要求B的形状是(ch_in, ch_out))
用SIMD指令每次取C中连续的pack个元素进行累加, 每次取B中连续的pack个元素进行计算(要求B的形状是(ch_in, ch_out)),
每次取A中1个元素广播成pack个进行计算.
3个循环由外到内是batch_size, ch_in, ch_out
'''

# AVX
bs = 0
while bs < batch_size:
    ic = 0
    while ic < ch_in:
        _a = _mm256_broadcast_ss(x, bs * ch_in + ic)
        oc = 0
        while oc + elempack - 1 < ch_out:
            _b = _mm256_loadu_ps(w_t, ic * ch_out + oc)
            _out = _mm256_loadu_ps(y, bs * ch_out + oc)
            _out = _mm256_fmadd_ps(_a, _b, _out)
            _mm256_storeu_ps(y, bs * ch_out + oc, _out)
            oc += elempack
        # print(y)
        # while oc < ch_out:
        #     y[bs * ch_out + oc] += x[bs * ch_in + ic] * w_t[ic * ch_out + oc]
        #     oc += 1
        # print(y)
        ic += 1
    bs += 1


y1 = np.reshape(y, y2.shape)
diff = np.sum((y1 - y2)**2)
print('diff=%.6f' % diff)

print()







