#ifndef __F_CONV2D_COMMON_H__
#define __F_CONV2D_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void conv2d(Tensor* input, Tensor* weight, Tensor* group_weights, Tensor* bias, Tensor* im2col, Tensor* output_t, Tensor* output, int stride_h = 1, int stride_w = 1, int padding_h = 0, int padding_w = 0, int dilation_h = 1, int dilation_w = 1, int groups = 1);

void conv2d(Tensor* input, Tensor* weight, Tensor* group_weights, Tensor* bias, Tensor* im2col, Tensor* output_t, Tensor* output, int stride = 1, int padding = 0, int dilation = 1, int groups = 1);

NS_MM_F_END

#endif // __F_CONV2D_COMMON_H__
