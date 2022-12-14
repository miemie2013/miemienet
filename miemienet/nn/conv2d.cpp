#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "conv2d.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/conv2d_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/conv2d_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, bool use_bias, bool create_weights)
{
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->use_bias = use_bias;
    this->kernel_h = kernel_size;
    this->kernel_w = kernel_size;
    this->stride_h = stride;
    this->stride_w = stride;
    this->padding_h = padding;
    this->padding_w = padding;
    this->dilation_h = dilation;
    this->dilation_w = dilation;
    this->groups = groups;
    this->son = nullptr;
    if (!create_weights)
    {
        this->weight = nullptr;
        this->bias = nullptr;
        this->group_weights = nullptr;
        return;
    }
    if (in_channels % groups != 0)
    {
        printf("in_channels must be divisible by groups.\n");
        exit(1);
    }
    if (out_channels % groups != 0)
    {
        printf("out_channels must be divisible by groups.\n");
        exit(1);
    }
    if (Config::getInstance()->image_data_format == NCHW)
    {
        this->weight = this->create_parameter("weight", MMSHAPE4D(out_channels, in_channels / groups, this->kernel_h, this->kernel_w), FP32, false);
    }
    else if (Config::getInstance()->image_data_format == NHWC)
    {
        this->weight = this->create_parameter("weight", MMSHAPE4D(this->kernel_h, this->kernel_w, in_channels / groups, out_channels), FP32, false);
    }
    this->weight->referenceCount++;
    if (use_bias)
    {
        if (Config::getInstance()->image_data_format == NCHW)
        {
            this->bias = this->create_parameter("bias", MMSHAPE4D(out_channels, 1, 1, 1), FP32, true, 0.f);
        }
        else if (Config::getInstance()->image_data_format == NHWC)
        {
            this->bias = this->create_parameter("bias", MMSHAPE4D(1, 1, 1, out_channels), FP32, true, 0.f);
        }
        this->bias->referenceCount++;
    }
    else
    {
        this->bias = nullptr;
    }
    this->group_weights = nullptr;
    this->reset_parameters();
}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups, bool use_bias, bool create_weights)
{
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->use_bias = use_bias;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;
    this->stride_h = stride_h;
    this->stride_w = stride_w;
    this->padding_h = padding_h;
    this->padding_w = padding_w;
    this->dilation_h = dilation_h;
    this->dilation_w = dilation_w;
    this->groups = groups;
    this->son = nullptr;
    if (!create_weights)
    {
        this->weight = nullptr;
        this->bias = nullptr;
        this->group_weights = nullptr;
        return;
    }
    if (in_channels % groups != 0)
    {
        printf("in_channels must be divisible by groups.\n");
        exit(1);
    }
    if (out_channels % groups != 0)
    {
        printf("out_channels must be divisible by groups.\n");
        exit(1);
    }
    if (Config::getInstance()->image_data_format == NCHW)
    {
        this->weight = this->create_parameter("weight", MMSHAPE4D(out_channels, in_channels / groups, this->kernel_h, this->kernel_w), FP32, false);
    }
    else if (Config::getInstance()->image_data_format == NHWC)
    {
        this->weight = this->create_parameter("weight", MMSHAPE4D(this->kernel_h, this->kernel_w, in_channels / groups, out_channels), FP32, false);
    }
    this->weight->referenceCount++;
    if (use_bias)
    {
        if (Config::getInstance()->image_data_format == NCHW)
        {
            this->bias = this->create_parameter("bias", MMSHAPE4D(out_channels, 1, 1, 1), FP32, true, 0.f);
        }
        else if (Config::getInstance()->image_data_format == NHWC)
        {
            this->bias = this->create_parameter("bias", MMSHAPE4D(1, 1, 1, out_channels), FP32, true, 0.f);
        }
        this->bias->referenceCount++;
    }
    else
    {
        this->bias = nullptr;
    }
    this->group_weights = nullptr;
    this->reset_parameters();
}

Conv2d::~Conv2d()
{
    if (weight != nullptr)
    {
        weight->referenceCount--;
        if (weight->referenceCount <= 0){
            delete weight;
        }
    }
    if (bias != nullptr)
    {
        bias->referenceCount--;
        if (bias->referenceCount <= 0){
            delete bias;
        }
    }
    if (group_weights != nullptr)
    {
        group_weights->referenceCount--;
        if (group_weights->referenceCount <= 0){
            delete group_weights;
        }
    }
    if (son != nullptr)
    {
        delete son;
    }
}

void Conv2d::reset_parameters()
{
}

Tensor* Conv2d::create_tensors(Tensor* input)
{
    if (input_tensors->size() == 0)
    {
        input->referenceCount++;
        input_tensors->push_back(input);

        Config* cfg = Config::getInstance();
        const int N = input->shape->at(0);
        int in_C, H, W, out_C, kH, kW;
        if (cfg->image_data_format == NCHW)
        {
            in_C = input->shape->at(1);
            H = input->shape->at(2);
            W = input->shape->at(3);
            out_C = weight->shape->at(0);
            kH = weight->shape->at(2);
            kW = weight->shape->at(3);
        }
        else if (cfg->image_data_format == NHWC)
        {
            H = input->shape->at(1);
            W = input->shape->at(2);
            in_C = input->shape->at(3);
            kH = weight->shape->at(0);
            kW = weight->shape->at(1);
            out_C = weight->shape->at(3);
        }

        const int kernel_extent_h = dilation_h * (kH - 1) + 1;
        const int kernel_extent_w = dilation_w * (kW - 1) + 1;
        const int out_H = (H + padding_h + padding_h - kernel_extent_h) / stride_h + 1;
        const int out_W = (W + padding_w + padding_w - kernel_extent_w) / stride_w + 1;

        Tensor* im2col;
        Tensor* output_t;
        bool input_as_im2col = kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0 && groups == 1;
        if (input_as_im2col)
        {
            im2col = nullptr;
            output_t = nullptr;
        }
        else if (in_channels == out_channels && groups == out_channels)
        {
//            this->group_weights = this->create_parameter("group_weights", MMSHAPE3D(groups, this->kernel_h * this->kernel_w, in_channels * out_channels / groups / groups), FP32, false);
//            this->group_weights->referenceCount++;
//            int iC = in_channels / groups;
//            int oC = out_channels / groups;
//            const int num_threads_ = cfg->num_threads;
//            #pragma omp parallel for num_threads(num_threads_)
//            for (int kh = 0; kh < kernel_h; kh++) {
//                int p = kh * kernel_w * in_channels * out_channels / groups;
//                for (int kw = 0; kw < kernel_w; kw++) {
//                    for (int ic = 0; ic < iC; ic++) {
//                        for (int oc = 0; oc < out_channels; oc++) {
//                            group_weights->data_fp32[((oc / oC * (kernel_h * kernel_w)) + (kh * kernel_w + kw)) * (in_channels * out_channels / groups / groups) + (ic * out_channels / groups + oc % oC)] = weight->data_fp32[p++];
//                        }
//                    }
//                }
//            }
//
//            im2col = new SNT Tensor(MMSHAPE3D(groups, N * out_H * out_W, kH * kW * iC), FP32, false, false);
//            im2col->referenceCount++;
//            output_t = new SNT Tensor(MMSHAPE2D(groups, N * out_H * out_W), FP32, false, false);
//            output_t->referenceCount++;

            im2col = nullptr;
            output_t = nullptr;
        }
        else
        {
            im2col = new SNT Tensor(MMSHAPE2D(N * out_H * out_W, kH * kW * in_C), FP32, false, false);
            im2col->referenceCount++;
            output_t = nullptr;
        }
        temp_tensors->push_back(im2col);
        temp_tensors->push_back(output_t);

        Tensor* output;
        if (cfg->image_data_format == NCHW)
        {
            output = new SNT Tensor(MMSHAPE4D(N, out_C, out_H, out_W), FP32, false, false);
        }
        else if (cfg->image_data_format == NHWC)
        {
            output = new SNT Tensor(MMSHAPE4D(N, out_H, out_W, out_C), FP32, false, false);
        }
        output->referenceCount++;
        output_tensors->push_back(output);

        return output;
    }
    else
    {
        if (son == nullptr)
        {
            son = new SNT Conv2d(in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups, use_bias, false);
            weight->referenceCount++;
            if (bias != nullptr)
            {
                bias->referenceCount++;
            }
            if (group_weights != nullptr)
            {
                group_weights->referenceCount++;
            }
            son->weight = weight;
            son->bias = bias;
            son->group_weights = group_weights;
        }
        return son->create_tensors(input);
    }
}

Tensor* Conv2d::feed_forward(Tensor* input)
{
    Tensor* input_ = input_tensors->at(0);
//    if (output->id == 207)
//    {
//        output->print_msg("output reg_feat");
//    }

    // ppyoloe???????????????reg??????????????????reg_feat???reshape???[N, HW, 4, reg_max + 1] ????????????[N, H, W, 4*(reg_max + 1)]??????
    // ??????reshape???inplace???????????????????????????feed_forward()??? ????????? ??? ??????????????? ??? reg_feat ???????????? ????????????
    // ???????????? reg_feat ????????????????????????
    // inplace?????????????????????????????????????????????bug???

    // son???miemienet????????????????????????????????????????????????
    // ??? PPYOLOEHead ?????? proj_conv ???????????? ??????????????? ?????????????????????
    // ?????????inplace(???????????????????????????????????????????????????)?????????????????????son????????? Activation???Softmax ??????
    // ????????????inplace??????????????????son????????? Concat???Interp???Reduce ??????
    // son???????????????????????????1??????????????????????????????????????????????????????
    // ???????????????inplace????????????????????? miemienet::functional::softmax(reg_feat, reg_feat, -1);
    // ??????????????????????????? Softmax ??????????????????????????????????????????????????????????????????????????????son??????
    // ????????????????????? ??????????????? ???inplace???Softmax(??????????????????????????????)???????????????????????????Softmax??????????????????Softmax??????dim?????????????????????Softmax???????????????son???
    // ????????????son??????????????????????????????????????????????????????PPYOLOEHead??? std::vector<Layer*>* global_avgpools; ???????????????????????????????????????????????????
    if (input_->id == input->id)
    {
        Tensor* im2col = temp_tensors->at(0);
        Tensor* output_t = temp_tensors->at(1);
        Tensor* output = output_tensors->at(0);
        if (output->dims != 4)
        {
            output->restore_shape();
        }
        else
        {
            if (output->shape->at(0) == output->ori_D0 && output->shape->at(1) == output->ori_D1 && output->shape->at(2) == output->ori_D2 && output->shape->at(3) == output->ori_D3)
            {
                ;
            }
            else
            {
                output->restore_shape();
            }
        }
        miemienet::functional::conv2d(input, weight, group_weights, bias, im2col, output_t, output, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
        return output;
    }
    else
    {
        return son->feed_forward(input);
    }
}

void Conv2d::print_msg(char* name)
{
    printf("Conv2d Layer \'%s\' msg: ", name);
    printf("in_channels=%d, ", in_channels);
    printf("out_channels=%d, ", out_channels);
    printf("kernel_h=%d, ", kernel_h);
    printf("kernel_w=%d, ", kernel_w);
    printf("stride_h=%d, ", stride_h);
    printf("stride_w=%d, ", stride_w);
    printf("padding_h=%d, ", padding_h);
    printf("padding_w=%d, ", padding_w);
    printf("dilation_h=%d, ", dilation_h);
    printf("dilation_w=%d, ", dilation_w);
    printf("groups=%d, ", groups);
    printf("use_bias=%d, ", use_bias);
    printf("\n");
    weight->print_msg(name);
}

NS_MM_END
