#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "maxpool2d.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/maxpool2d_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/maxpool2d_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding, bool ceil_mode)
{
    this->kernel_h = kernel_size;
    this->kernel_w = kernel_size;
    this->stride_h = stride;
    this->stride_w = stride;
    this->padding_h = padding;
    this->padding_w = padding;
    this->ceil_mode = ceil_mode;
}

MaxPool2d::MaxPool2d(int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, bool ceil_mode)
{
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;
    this->stride_h = stride_h;
    this->stride_w = stride_w;
    this->padding_h = padding_h;
    this->padding_w = padding_w;
    this->ceil_mode = ceil_mode;
}

MaxPool2d::~MaxPool2d()
{
}

Tensor* MaxPool2d::create_tensors(Tensor* input)
{
    input->referenceCount++;
    input_tensors->push_back(input);

    Config* cfg = Config::getInstance();
    const int N = input->shape->at(0);
    int C, H, W;
    if (cfg->image_data_format == NCHW)
    {
        C = input->shape->at(1);
        H = input->shape->at(2);
        W = input->shape->at(3);
    }
    else if (cfg->image_data_format == NHWC)
    {
        H = input->shape->at(1);
        W = input->shape->at(2);
        C = input->shape->at(3);
    }
    // 输出形状推导借鉴自 https://gitee.com/paddlepaddle/Paddle/blob/release/2.0/paddle/fluid/operators/pool_op.cc
    int out_H;
    int out_W;
    if (!ceil_mode) {
        out_H = (H + padding_h + padding_h - kernel_h) / stride_h + 1;
        out_W = (W + padding_w + padding_w - kernel_w) / stride_w + 1;
    }
    else  // ceil_mode==true表示上取整
    {
        out_H = (H + padding_h + padding_h - kernel_h + stride_h - 1) / stride_h + 1;
        out_W = (W + padding_w + padding_w - kernel_w + stride_w - 1) / stride_w + 1;
    }

    Tensor* output;
    if (cfg->image_data_format == NCHW)
    {
        output = new SNT Tensor(MMSHAPE4D(N, C, out_H, out_W), FP32, false, false);
    }
    else if (cfg->image_data_format == NHWC)
    {
        output = new SNT Tensor(MMSHAPE4D(N, out_H, out_W, C), FP32, false, false);
    }
    output->referenceCount++;
    output_tensors->push_back(output);
    return output;
}

Tensor* MaxPool2d::feed_forward(Tensor* input)
{
    Tensor* output = output_tensors->at(0);
    miemienet::functional::maxpool2d(input, output, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, ceil_mode);
    return output;
}

NS_MM_END
