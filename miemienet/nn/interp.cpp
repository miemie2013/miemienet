#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "interp.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/interp_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/interp_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Interp::Interp(int size_h, int size_w, float scale_h, float scale_w, char* mode, bool align_corners, bool recompute_scale_factor)
{
    this->size_h = size_h;
    this->size_w = size_w;
    this->scale_h = scale_h;
    this->scale_w = scale_w;
    this->mode = mode;
    this->align_corners = align_corners;
    this->recompute_scale_factor = recompute_scale_factor;
    this->son = nullptr;
}

Interp::~Interp()
{
    if (son != nullptr)
    {
        delete son;
    }
}

Tensor* Interp::create_tensors(Tensor* input)
{
    if (input_tensors->size() == 0)
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
        int out_H = H * 2;
        int out_W = W * 2;
//        if (!ceil_mode) {
//            out_H = (H + padding_h + padding_h - kernel_h) / stride_h + 1;
//            out_W = (W + padding_w + padding_w - kernel_w) / stride_w + 1;
//        }
//        else  // ceil_mode==true表示上取整
//        {
//            out_H = (H + padding_h + padding_h - kernel_h + stride_h - 1) / stride_h + 1;
//            out_W = (W + padding_w + padding_w - kernel_w + stride_w - 1) / stride_w + 1;
//        }

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
    else
    {
        if (son == nullptr)
        {
            son = new SNT Interp(size_h, size_w, scale_h, scale_w, mode, align_corners, recompute_scale_factor);
        }
        return son->create_tensors(input);
    }
}

Tensor* Interp::feed_forward(Tensor* input)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::interp(input, output, size_h, size_w, scale_h, scale_w, mode, align_corners, recompute_scale_factor);
        return output;
    }
    else
    {
        return son->feed_forward(input);
    }
}

NS_MM_END
