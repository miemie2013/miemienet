#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../concat.h"
#include "concat_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

template<typename data_t>
void concat4d_2tensor_dim3_cpp_kernel(const int num_threads_, const data_t* tensor1, const data_t* tensor2, data_t* out, int num, int N, int C, int H, int W1, int W2){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w1 = 0; w1 < W1; w1++) {
                    out[((n * C + c) * H + h) * (W1 + W2) + w1] = tensor1[((n * C + c) * H + h) * W1 + w1];
                }
                for (int w2 = 0; w2 < W2; w2++) {
                    out[((n * C + c) * H + h) * (W1 + W2) + w2 + W1] = tensor2[((n * C + c) * H + h) * W2 + w2];
                }
            }
        }
    }
}

template<typename data_t>
void concat4d_4tensor_dim3_cpp_kernel(const int num_threads_, const data_t* tensor1, const data_t* tensor2, const data_t* tensor3, const data_t* tensor4, data_t* out, int num, int N, int C, int H, int W1, int W2, int W3, int W4){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w1 = 0; w1 < W1; w1++) {
                    out[((n * C + c) * H + h) * (W1 + W2 + W3 + W4) + w1] = tensor1[((n * C + c) * H + h) * W1 + w1];
                }
                for (int w2 = 0; w2 < W2; w2++) {
                    out[((n * C + c) * H + h) * (W1 + W2 + W3 + W4) + w2 + W1] = tensor2[((n * C + c) * H + h) * W2 + w2];
                }
                for (int w3 = 0; w3 < W3; w3++) {
                    out[((n * C + c) * H + h) * (W1 + W2 + W3 + W4) + w3 + W1 + W2] = tensor3[((n * C + c) * H + h) * W3 + w3];
                }
                for (int w4 = 0; w4 < W4; w4++) {
                    out[((n * C + c) * H + h) * (W1 + W2 + W3 + W4) + w4 + W1 + W2 + W3] = tensor4[((n * C + c) * H + h) * W4 + w4];
                }
            }
        }
    }
}


template<typename data_t>
void concat3d_3tensor_dim1_cpp_kernel(const int num_threads_, const data_t* tensor1, const data_t* tensor2, const data_t* tensor3, data_t* out, int num, int N, int H1, int H2, int H3, int W){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H1 + H2 + H3; h++) {
            if (h < H1)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3) + h) * W + w] = tensor1[(n * H1 + h) * W + w];
                }
            }
            else if (h < H1 + H2)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3) + h) * W + w] = tensor2[(n * H2 + h - H1) * W + w];
                }
            }
            else if (h < H1 + H2 + H3)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3) + h) * W + w] = tensor3[(n * H3 + h - H1 - H2) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void concat3d_4tensor_dim1_cpp_kernel(const int num_threads_, const data_t* tensor1, const data_t* tensor2, const data_t* tensor3, const data_t* tensor4, data_t* out, int num, int N, int H1, int H2, int H3, int H4, int W){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H1 + H2 + H3 + H4; h++) {
            if (h < H1)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3 + H4) + h) * W + w] = tensor1[(n * H1 + h) * W + w];
                }
            }
            else if (h < H1 + H2)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3 + H4) + h) * W + w] = tensor2[(n * H2 + h - H1) * W + w];
                }
            }
            else if (h < H1 + H2 + H3)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3 + H4) + h) * W + w] = tensor3[(n * H3 + h - H1 - H2) * W + w];
                }
            }
            else if (h < H1 + H2 + H3 + H4)
            {
                for (int w = 0; w < W; w++) {
                    out[(n * (H1 + H2 + H3 + H4) + h) * W + w] = tensor3[(n * H4 + h - H1 - H2 - H3) * W + w];
                }
            }
        }
    }
}


void concat(Tensor* input1, Tensor* input2, Tensor* output, int dim)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    const int dims = input1->dims;
    int positive_dim = dim < 0 ? dims + dim : dim;

    if (cfg->use_cpp_compute)
    {
        if (dims == 4 && positive_dim == 3)
        {
            const int N = input1->shape->at(0);
            const int C = input1->shape->at(1);
            const int H = input1->shape->at(2);
            const int W1 = input1->shape->at(3);
            const int W2 = input2->shape->at(3);
            concat4d_2tensor_dim3_cpp_kernel<float>(num_threads_, input1->data_fp32, input2->data_fp32, output->data_fp32, output->numel, N, C, H, W1, W2);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
            exit(1);
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

void concat(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* output, int dim)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    const int dims = input1->dims;
    int positive_dim = dim < 0 ? dims + dim : dim;

    if (cfg->use_cpp_compute)
    {
        if (dims == 3 && positive_dim == 1)
        {
            const int N = input1->shape->at(0);
            const int H1 = input1->shape->at(1);
            const int H2 = input2->shape->at(1);
            const int H3 = input3->shape->at(1);
            const int W = input1->shape->at(2);
            concat3d_3tensor_dim1_cpp_kernel<float>(num_threads_, input1->data_fp32, input2->data_fp32, input3->data_fp32, output->data_fp32, output->numel, N, H1, H2, H3, W);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
            exit(1);
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

void concat(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4, Tensor* output, int dim)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    const int dims = input1->dims;
    int positive_dim = dim < 0 ? dims + dim : dim;

    if (cfg->use_cpp_compute)
    {
        if (dims == 4 && positive_dim == 3)
        {
            const int N = input1->shape->at(0);
            const int C = input1->shape->at(1);
            const int H = input1->shape->at(2);
            const int W1 = input1->shape->at(3);
            const int W2 = input2->shape->at(3);
            const int W3 = input3->shape->at(3);
            const int W4 = input4->shape->at(3);
            concat4d_4tensor_dim3_cpp_kernel<float>(num_threads_, input1->data_fp32, input2->data_fp32, input3->data_fp32, input4->data_fp32, output->data_fp32, output->numel, N, C, H, W1, W2, W3, W4);
        }
        else if (dims == 3 && positive_dim == 1)
        {
            const int N = input1->shape->at(0);
            const int H1 = input1->shape->at(1);
            const int H2 = input2->shape->at(1);
            const int H3 = input3->shape->at(1);
            const int H4 = input4->shape->at(1);
            const int W = input1->shape->at(2);
            concat3d_4tensor_dim1_cpp_kernel<float>(num_threads_, input1->data_fp32, input2->data_fp32, input3->data_fp32, input4->data_fp32, output->data_fp32, output->numel, N, H1, H2, H3, H4, W);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
            exit(1);
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
