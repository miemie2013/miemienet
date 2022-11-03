

import numpy as np


def winograd23_weight_transform(x):
    G=np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
    aaaaaaaaaaa = np.matmul(G,x)
    aaaaaaaaaaa2 = np.matmul(aaaaaaaaaaa,G.T)
    return np.matmul(np.matmul(G,x),G.T)


def winograd23_NHWC(input, weight, out_H, out_W):
    H, W, in_C = input.shape
    kH, kW, in_C, out_C = weight.shape
    if kH != 3 or kW != 3:
        return None

    tile_size = 4  # 一次计算输出的2x2像素瓦片，对应的输入是原图的lxl像素瓦片，这里l=4
    otile_size = 2  # 一次计算输出的2x2像素瓦片，对应的输入是原图的lxl像素瓦片，这里l=4
    tile_num = int(out_H / otile_size) * int(out_W / otile_size)  # 像素瓦片个数
    U = np.zeros((tile_size, tile_size, in_C, out_C))  # [4, 4, in_C, out_C]   3x3卷积变成了4x4卷积
    V = np.zeros((tile_size, tile_size, tile_num, in_C))  # [4, 4, out_H/2*out_W/2, in_C]     输入图片变成了这样
    Q = np.zeros((tile_size, tile_size, tile_num, out_C))  # Q = V * U
    Y = np.zeros((otile_size, otile_size, tile_num, out_C))
    O = np.zeros((out_H, out_W, out_C))
    # 转换卷积核。U是新的卷积核？ 3x3卷积变成了4x4卷积
    for ic in range(in_C):
        for oc in range(out_C):
            U[:, :, ic, oc] = winograd23_weight_transform(weight[:, :, ic, oc])
    # 转换输入
    for tid in range(tile_num):
        th = (tid // (out_W // otile_size) ) * 2
        tw = (tid % (out_W // otile_size) ) * 2
        for ic in range(in_C):
            V[0, 0, tid, ic] = input[th][tw][ic] - input[th+2][tw][ic] - input[th][tw+2][ic] + input[th+2][tw+2][ic]
            V[0, 1, tid, ic] = input[th][tw+1][ic] - input[th+2][tw+1][ic] + input[th][tw+2][ic] - input[th+2][tw+2][ic]
            V[0, 2, tid, ic] = -input[th][tw+1][ic] + input[th+2][tw+1][ic] + input[th][tw+2][ic] - input[th+2][tw+2][ic]
            V[0, 3, tid, ic] = input[th][tw+1][ic] - input[th+2][tw+1][ic] - input[th][tw+3][ic] + input[th+2][tw+3][ic]
            V[1, 0, tid, ic] = input[th+1][tw][ic] + input[th+2][tw][ic] - input[th+1][tw+2][ic] - input[th+2][tw+2][ic]
            V[1, 1, tid, ic] = input[th+1][tw+1][ic] + input[th+2][tw+1][ic] + input[th+1][tw+2][ic] + input[th+2][tw+2][ic]
            V[1, 2, tid, ic] = -input[th+1][tw+1][ic] - input[th+2][tw+1][ic] + input[th+1][tw+2][ic] + input[th+2][tw+2][ic]
            V[1, 3, tid, ic] = input[th+1][tw+1][ic] + input[th+2][tw+1][ic] - input[th+1][tw+3][ic] - input[th+2][tw+3][ic]
            V[2, 0, tid, ic] = -input[th+1][tw][ic] + input[th+2][tw][ic] + input[th+1][tw+2][ic] - input[th+2][tw+2][ic]
            V[2, 1, tid, ic] = -input[th+1][tw+1][ic] + input[th+2][tw+1][ic] - input[th+1][tw+2][ic] + input[th+2][tw+2][ic]
            V[2, 2, tid, ic] = input[th+1][tw+1][ic] - input[th+2][tw+1][ic] - input[th+1][tw+2][ic] + input[th+2][tw+2][ic]
            V[2, 3, tid, ic] = -input[th+1][tw+1][ic] + input[th+2][tw+1][ic] + input[th+1][tw+3][ic] - input[th+2][tw+3][ic]
            V[3, 0, tid, ic] = input[th+1][tw][ic] - input[th+3][tw][ic] - input[th+1][tw+2][ic] + input[th+3][tw+2][ic]
            V[3, 1, tid, ic] = input[th+1][tw+1][ic] - input[th+3][tw+1][ic] + input[th+1][tw+2][ic] - input[th+3][tw+2][ic]
            V[3, 2, tid, ic] = -input[th+1][tw+1][ic] + input[th+3][tw+1][ic] + input[th+1][tw+2][ic] - input[th+3][tw+2][ic]
            V[3, 3, tid, ic] = input[th+1][tw+1][ic] - input[th+3][tw+1][ic] - input[th+1][tw+3][ic] + input[th+3][tw+3][ic]
    for i in range(tile_size):
        for j in range(tile_size):
            Q[i, j, :, :] = np.matmul(V[i, j, :, :], U[i, j, :, :])
    for tid in range(tile_num):
        for oc in range(out_C):
            Y[0, 0, tid, oc] = Q[0][0][tid][oc] + Q[1][0][tid][oc] + Q[2][0][tid][oc] + Q[0][1][tid][oc] + Q[1][1][tid][oc] + Q[2][1][tid][oc] + Q[0][2][tid][oc] + Q[1][2][tid][oc] + Q[2][2][tid][oc]
            Y[0, 1, tid, oc] = Q[0][1][tid][oc] + Q[1][1][tid][oc] + Q[2][1][tid][oc] - Q[0][2][tid][oc] - Q[1][2][tid][oc] - Q[2][2][tid][oc] - Q[0][3][tid][oc] - Q[1][3][tid][oc] - Q[2][3][tid][oc]
            Y[1, 0, tid, oc] = Q[1][0][tid][oc] - Q[2][0][tid][oc] - Q[3][0][tid][oc] + Q[1][1][tid][oc] - Q[2][1][tid][oc] - Q[3][1][tid][oc] + Q[1][2][tid][oc] - Q[2][2][tid][oc] - Q[3][2][tid][oc]
            Y[1, 1, tid, oc] = Q[1][1][tid][oc] - Q[2][1][tid][oc] - Q[3][1][tid][oc] - Q[1][2][tid][oc] + Q[2][2][tid][oc] + Q[3][2][tid][oc] - Q[1][3][tid][oc] + Q[2][3][tid][oc] + Q[3][3][tid][oc]

    # Y = np.zeros((otile_size, otile_size, tile_num, out_C))
    # O = np.zeros((out_H, out_W, out_C))

    for tid in range(tile_num):
        th = (tid // (out_W // otile_size) ) * 2
        tw = (tid % (out_W // otile_size) ) * 2
        for oc in range(out_C):
            O[th][tw][oc] = Y[0, 0, tid, oc]
            O[th][tw+1][oc] = Y[0, 1, tid, oc]
            O[th+1][tw][oc] = Y[1, 0, tid, oc]
            O[th+1][tw+1][oc] = Y[1, 1, tid, oc]
    return O





'''
1509.09308.pdf

'''





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






N = 1

H = 4
W = 4

# H = 5
# W = 5


# H = 5
# W = 6


# H = 6
# W = 5





in_C = 1
out_C = 1

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
weight = np.random.random((kH*kW*in_C, out_C)).astype(np.float32)

input = np.reshape(input, (-1, ))
weight = np.reshape(weight, (-1, ))
for i in range(N * H * W * in_C):
    # input[i] = (i + 1) % 100
    input[i] = i + 1
for i in range(kH * kW * in_C * out_C):
    weight[i] = (i + 100) / 10 + 0.5
input = np.reshape(input, (N, H, W, in_C))
weight = np.reshape(weight, (kH*kW*in_C, out_C))


im2col = np.zeros((N * out_H * out_W, kH * kW * in_C)).astype(np.float32)

input = np.reshape(input, (-1, ))
im2col = np.reshape(im2col, (-1, ))

im2col = imNHWC2col_cpp_kernel(input, im2col, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);


im2col = np.reshape(im2col, (N * out_H * out_W, kH * kW * in_C))

out_true = np.matmul(im2col, weight)

out_true = np.reshape(out_true, (out_H, out_W, out_C))




input = np.reshape(input, (H, W, in_C))
pad_input = np.zeros((H + 2, W + 2, in_C))

if H % 2 == 1 and W % 2 == 1:
    pad_input = np.zeros((H + 2 + 1, W + 2 + 1, in_C))
if H % 2 == 0 and W % 2 == 1:
    pad_input = np.zeros((H + 2 + 0, W + 2 + 1, in_C))
if H % 2 == 1 and W % 2 == 0:
    pad_input = np.zeros((H + 2 + 1, W + 2 + 0, in_C))

pad_input[1:1+H, 1:1+W, :] = input
weight = np.reshape(weight, (kH, kW, in_C, out_C))
out_wino = winograd23_NHWC(pad_input, weight, out_H, out_W)
# out_wino2 = winograd23_NHWC_nopad(input, weight, out_H, out_W, padding_h, padding_w)


# ccccccc = ccccccc.transpose((3, 2, 0, 1))
# ccccccc = ccccccc.transpose((2, 0, 1))
# print(np.max(np.abs(bbbbbbb-ccccccc)))

if H % 2 == 1 and W % 2 == 1:
    out_wino = out_wino[:-1, :-1, :]
if H % 2 == 0 and W % 2 == 1:
    out_wino = out_wino[:, :-1, :]
if H % 2 == 1 and W % 2 == 0:
    out_wino = out_wino[:-1, :, :]
print(np.max(np.abs(out_true-out_wino)))

# print(np.max(np.abs(out_true-out_wino2)))

print()

