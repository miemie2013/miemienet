#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../softmax.h"
#include "softmax_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

template<typename data_t>
void softmax4d_3_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                float max = x[((n * C + c) * H + h) * W];
                for (int w = 1; w < W; w++) {
                    max = std::max(max, x[((n * C + c) * H + h) * W + w]);
                }

                float sum = 0.f;
                for (int w = 0; w < W; w++) {
                    float temp = x[((n * C + c) * H + h) * W + w];
                    temp = static_cast<float>(exp(temp - max));
                    sum += temp;
                    y[((n * C + c) * H + h) * W + w] = temp;
                }

                for (int w = 0; w < W; w++) {
                    y[((n * C + c) * H + h) * W + w] /= sum;
                }
            }
        }
    }
}


void softmax(Tensor* input, Tensor* output, int dim)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    const int tensor_dims = input->dims;
    int positive_dim = dim < 0 ? tensor_dims + dim : dim;
    if (positive_dim < 0 || positive_dim >= tensor_dims)
    {
        printf("Error from softmax op, invalid arg dim=%d.\n", dim);
        exit(1);
    }

    if (cfg->use_cpp_compute)
    {
        const int N = input->shape->at(0);
        const int C = input->shape->at(1);
        const int H = input->shape->at(2);
        const int W = input->shape->at(3);
        if (tensor_dims == 4 && positive_dim == 3)
        {
            softmax4d_3_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else
        {
            printf("Softmax op type tensor_dims == %d && dim == %d not implemented!\n", tensor_dims, dim);
            exit(1);
        }
    }
    else
    {
#if BACKEND_X86
        const int N = input->shape->at(0);
        const int C = input->shape->at(1);
        const int H = input->shape->at(2);
        const int W = input->shape->at(3);
        if (tensor_dims == 4 && positive_dim == 3)
        {
            softmax4d_3_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else
        {
            printf("Softmax op type tensor_dims == %d && dim == %d not implemented!\n", tensor_dims, dim);
            exit(1);
        }
#endif // BACKEND_X86

#if BACKEND_ARM
#endif // BACKEND_ARM
    }
}

NS_MM_F_END
