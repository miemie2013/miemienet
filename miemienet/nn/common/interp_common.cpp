#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../interp.h"
#include "interp_common.h"

#include "elementwise_common.h"
#include "matmul_common.h"
//#include "reduce_common.h"

NS_MM_F_BEGIN


template<typename data_t>
void interp_cpp_kernel(const int num_threads_, const data_t* im, data_t* out, int num, int N, int out_H, int out_W, int in_C){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int ic = 0; ic < in_C; ic++) {
                    out[((n * out_H + oh) * out_W + ow) * in_C + ic] = im[((n * out_H/2 + oh/2) * out_W/2 + ow/2) * in_C + ic];
                }
            }
        }
    }
}




void interp(Tensor* input, Tensor* output, int size_h, int size_w, float scale_h, float scale_w, char* mode, bool align_corners, bool recompute_scale_factor)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;
    const int N = input->shape->at(0);
    int C, H, W;
    int out_H;
    int out_W;
    if (cfg->image_data_format == NCHW)
    {
        C = input->shape->at(1);
        H = input->shape->at(2);
        W = input->shape->at(3);
        out_H = output->shape->at(2);
        out_W = output->shape->at(3);
    }
    else if (cfg->image_data_format == NHWC)
    {
        H = input->shape->at(1);
        W = input->shape->at(2);
        C = input->shape->at(3);
        out_H = output->shape->at(1);
        out_W = output->shape->at(2);
    }

    if (cfg->image_data_format == NCHW)
    {
        ;
//        interp_cpp_kernel<float>(num_threads_, input->data_fp32, out->data_fp32, out->numel, N, out_H, out_W, C);
    }
    else if (cfg->image_data_format == NHWC)
    {
        interp_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, output->numel, N, out_H, out_W, C);
    }
}

NS_MM_F_END
