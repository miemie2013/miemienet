#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../reduce.h"
#include "reduce_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

template<typename data_t>
void reduce4d_NCHW_dim12_mean_cpp_kernel(const int num_threads_, const data_t* x,
                                         data_t* y,
                                         int num,
                                         int N,
                                         int C,
                                         int H,
                                         int W){
    const float M = (float)(C * H);
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int w = 0; w < W; w++) {
            float sum = 0.f;
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    sum += x[((n * C + c) * H + h) * W + w];
                }
            }
            y[n * W + w] = sum / M;
        }
    }
}

void reduce(Tensor* input, Tensor* output, std::vector<int>* dims, bool keepdim, int op_type)
{
/*
op_type类型的宏定义见macros.h
#define RED_SUM 0
#define RED_SUMSQUARE 2
#define RED_MEAN 3
#define RED_MAX 4
#define RED_MIN 5
#define RED_PROD 6
#define RED_L1 7
#define RED_L2 8
#define RED_LOGSUM 9
#define RED_LOGSUMEXP 10
*/
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    const int tensor_dims = input->dims;

    if (cfg->use_cpp_compute)
    {
        if (op_type == RED_MEAN)
        {
            if (tensor_dims == 4 && dims->at(0) == 1 && dims->at(1) == 2)
            {
                const int N = input->shape->at(0);
                const int C = input->shape->at(1);
                const int H = input->shape->at(2);
                const int W = input->shape->at(3);
                reduce4d_NCHW_dim12_mean_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
            }
            else
            {
                input->print_msg("input");
                printf("Reduce op type tensor_dims == %d && dim == %d not implemented!\n", tensor_dims, 333);
                exit(1);
            }
        }
    }
    else
    {
#if BACKEND_X86
#endif // BACKEND_X86

#if BACKEND_ARM
#endif // BACKEND_ARM
    }
}

NS_MM_F_END
