import numpy as np
import os
import time
from SIMD_tools import _mm256_setzero_ps, _mm256_loadu_ps, _mm256_fmadd_ps, _mm256_broadcast_ss, _mm256_storeu_ps
from test2_utils import save_ndarray_as_txt

def imNHWC2col_cpp_kernel(im, im2col, num, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups):
    for n in range(N):
        for oh in range(out_H):
            for ow in range(out_W):
                for kh in range(kH):
                    for kw in range(kW):
                        for ic in range(in_C):
                            h_in = oh * stride_h - padding_h
                            w_in = ow * stride_w - padding_w
                            h = h_in + kh * dilation_h
                            w = w_in + kw * dilation_w
                            cond = h > -1 and w > -1 and h < H and w < W;
                            for ic in range(in_C):
                                val = 0.
                                if cond:
                                    val = im[(((n * H) + h) * W + w) * in_C + ic]
                                im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + kw) * in_C + ic)] = val
    return im2col


def imNHWC2col_k3s1d1_cpp_kernel(im, im2col, num, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups):
    for n in range(N):
        for oh in range(out_H):
            for ow in range(out_W):
                for kh in range(kH):
                    x00_h = oh - padding_h + kh;
                    x00_w = ow - padding_w + 0;
                    x01_h = oh - padding_h + kh;
                    x01_w = ow - padding_w + 1;
                    x02_h = oh - padding_h + kh;
                    x02_w = ow - padding_w + 2;



                    if ow == 0:
                        cond00 = x00_h > -1 and x00_w > -1 and x00_h < H and x00_w < W;
                        for ic in range(in_C):
                            val = 0.
                            if cond00:
                                val = im[(((n * H) + x00_h) * W + x00_w) * in_C + ic]
                            im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 0) * in_C + ic)] = val;
                        cond01 = x01_h > -1 and x01_w > -1 and x01_h < H and x01_w < W;
                        for ic in range(in_C):
                            val = 0.
                            if cond01:
                                val = im[(((n * H) + x01_h) * W + x01_w) * in_C + ic]
                            im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 1) * in_C + ic)] = val;
                            if ow + 1 < out_W:
                                im2col[((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 1 - 1) * in_C + ic)] = val;
                    cond02 = x02_h > -1 and x02_w > -1 and x02_h < H and x02_w < W;
                    for ic in range(in_C):
                        val = 0.
                        if cond02:
                            val = im[(((n * H) + x02_h) * W + x02_w) * in_C + ic]
                        im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 2) * in_C + ic)] = val;
                        if ow + 1 < out_W:
                            im2col[((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 2 - 1) * in_C + ic)] = val;
                        if ow + 2 < out_W:
                            im2col[((((((n * out_H) + oh) * out_W + ow + 2) * kH + kh) * kW + 2 - 2) * in_C + ic)] = val;


    return im2col





N = 1



H = 1
W = 1




H = 4
W = 4


H = 6
W = 6





in_C = 8

kH = 3
kW = 3

stride_h = 1
stride_w = 1
padding_h = 1
padding_w = 1


dilation_h = 1
dilation_w = 1
groups = 1

kernel_extent_h = dilation_h * (kH - 1) + 1
kernel_extent_w = dilation_w * (kW - 1) + 1
out_H = (H + padding_h + padding_h - kernel_extent_h) // stride_h + 1
out_W = (W + padding_w + padding_w - kernel_extent_w) // stride_w + 1

input_numel = N * H * W * in_C
out_numel = N * out_H * out_W * kH * kW * in_C


np.random.seed(13)

input = np.random.random((N, H, W, in_C)).astype(np.float32)
out_true = np.zeros((N * out_H * out_W, kH * kW * in_C)).astype(np.float32)
out = np.zeros((N * out_H * out_W, kH * kW * in_C)).astype(np.float32)
out -= 33.

input = np.reshape(input, (-1, ))
out_true = np.reshape(out_true, (-1, ))
out = np.reshape(out, (-1, ))

out_true = imNHWC2col_cpp_kernel(input, out_true, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);


out = imNHWC2col_k3s1d1_cpp_kernel(input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);


# aaaaa1 = np.reshape(out_true, (out_H, out_W, kH, kW))
# aaaaa2 = np.reshape(out, (out_H, out_W, kH, kW))

# aaaaa1 = np.reshape(out_true, (out_H* out_W, kH, kW))
# aaaaa2 = np.reshape(out, (out_H* out_W, kH, kW))


diff = np.sum((out_true - out)**2)
print('diff=%.6f' % diff)


print()







