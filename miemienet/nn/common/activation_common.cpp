#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../activation.h"
#include "activation_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

template<typename data_t>
void leakyrelu_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, float alpha, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = x[i] * alpha;
        }
    }
}

template<typename data_t>
void leakyrelu_x86_kernel(const int num_threads_, const data_t* x, data_t* y, float alpha, int num){

    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = x[i] * alpha;
        }
    }

//    const int BLOCK = 128;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int b = 0; b < num; b+=BLOCK) {
//        for (int i = 0; i < BLOCK; i++) {
//            int j = i + b;
//            if (x[j] > static_cast<data_t>(0.f))
//            {
//                y[j] = x[j];
//            }
//            else
//            {
//                y[j] = x[j] * alpha;
//            }
//        }
//    }


//    int H = 256;
//    int W = 256*256;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int h = 0; h < H; h+=BLOCK) {
//        for (int w = 0; w < W; w+=BLOCK) {
//            for (int i = 0; i < BLOCK; i++) {
//                for (int j = 0; j < BLOCK; j++) {
//                    float val = x[((h + j) * W) + (w + i)];
//                    if (val > static_cast<data_t>(0.f))
//                    {
//                        y[((h + j) * W) + (w + i)] = val;
//                    }
//                    else
//                    {
//                        y[((h + j) * W) + (w + i)] = val * alpha;
//                    }
//                }
//            }
//        }
//    }
}

template<typename data_t>
void leakyrelu_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* x, data_t* dx, float alpha, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = dy[i] * alpha;
        }
    }
}

template<typename data_t>
void relu_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            y[i] = x[i];
        }
        else
        {
            y[i] = static_cast<data_t>(0.f);
        }
    }
}

template<typename data_t>
void relu_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* x, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(0.f))
        {
            dx[i] = dy[i];
        }
        else
        {
            dx[i] = static_cast<data_t>(0.f);
        }
    }
}

// expf instead of exp should be used for float type, complement
// and register float kernel separatelly
template<typename data_t>
void sigmoid_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(1.f / (1.f + expf(-x[i])));
    }
}

template<typename data_t>
void sigmoid_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* y, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        dx[i] = dy[i] * y[i] * (1.f - y[i]);
    }
}

template<typename data_t>
void tanh_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = tanhf(x[i]);
    }
}

template<typename data_t>
void tanh_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* y, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

template<typename data_t>
void mish_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        // y1 = ln(1+e^x)
        // y2 = tanh(y1)
        // y = mish(x) = x * y2 = x * tanh(ln(1+e^x))
        y[i] = x[i] * tanhf(logf(1.f + expf(x[i])));
    }
}

// from https://gitee.com/paddlepaddle/Paddle/blob/release/2.0/paddle/fluid/operators/mish_op.cu   KeMishBwFP32()
template<typename data_t>
void mish_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* x, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        float sp = logf(1.f + expf(x[i]));
        float tsp = tanhf(sp);
        float grad_sp = -expm1f(-sp);
        float grad_tsp = (static_cast<float>(1) - tsp * tsp) * grad_sp;
        dx[i] = dy[i] * (x[i] * grad_tsp + tsp);
    }
}

template<typename data_t>
void swish_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(x[i] / (1.f + expf(-x[i])));
    }
}


template<typename data_t>
void swish_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
#if defined(WINDOWS)
    const int elempack = 8;
    const int num_packs = num / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        const float* x_ptr = x + pid * elempack;
        float* y_ptr = y + pid * elempack;
        __m256 _x = _mm256_loadu_ps(x_ptr);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 _y = _mm256_div_ps(_x, _mm256_add_ps(one, _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), _x))));
        _mm256_storeu_ps(y_ptr, _y);
    }
    int offset_ = num_packs * elempack;
    if (num - offset_ >= 4)
    {
        const float* x_ptr = x + offset_;
        float* y_ptr = y + offset_;
        __m128 _x = _mm_load_ps(x_ptr);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 _y = _mm_div_ps(_x, _mm_add_ps(one, _mm_exp_ps(_mm_sub_ps(_mm_setzero_ps(), _x))));
        _mm_store_ps(y_ptr, _x);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < num; i++) {
        y[i] = static_cast<float>(x[i] / (1.f + expf(-x[i])));
    }
#endif
#if defined(LINUX)
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(x[i] / (1.f + expf(-x[i])));
    }
#endif
}

template<typename data_t>
void swish_grad_cpp_kernel(const int num_threads_, const data_t* dy, const data_t* y, const data_t* x, data_t* dx, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        dx[i] = dy[i] * (y[i] + (1.f - y[i]) / (1.f + expf(-x[i])));
    }
}

template<typename data_t>
void hardsigmoid_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(3.f))
        {
            y[i] = 1.f;
        }
        else if (x[i] < static_cast<data_t>(-3.f))
        {
            y[i] = 0.f;
        }
        else
        {
            y[i] = x[i] / 6.f + 0.5f;
        }
    }
}

template<typename data_t>
void hardswish_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        if (x[i] > static_cast<data_t>(3.f))
        {
            y[i] = x[i];
        }
        else if (x[i] < static_cast<data_t>(-3.f))
        {
            y[i] = 0.f;
        }
        else
        {
            y[i] = x[i] * (x[i] + 3.f) / 6.f;
        }
    }
}

template<typename data_t>
void exp_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(expf(x[i]));
    }
}

template<typename data_t>
void square_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = x[i] * x[i];
    }
}

template<typename data_t>
void sqrt_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, float eps, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(sqrtf(x[i] + eps));
    }
}

template<typename data_t>
void rsqrt_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num){
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num; i++) {
        y[i] = static_cast<float>(1.f / sqrtf(x[i]));
    }
}



void activation(Tensor* input, Tensor* output, char* type, float alpha)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    bool use_cpp_compute = cfg->use_cpp_compute;
    use_cpp_compute = false;
    if (use_cpp_compute)
    {
        if (strcmp(type, "leakyrelu") == 0)
        {
            leakyrelu_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, alpha, input->numel);
        }
        else if (strcmp(type, "relu") == 0)
        {
            relu_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "sigmoid") == 0)
        {
            sigmoid_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "tanh") == 0)
        {
            tanh_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "mish") == 0)
        {
            mish_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "swish") == 0)
        {
            swish_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "hardsigmoid") == 0)
        {
            hardsigmoid_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "hardswish") == 0)
        {
            hardswish_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "exp") == 0)
        {
            exp_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "square") == 0)
        {
            square_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "sqrt") == 0)
        {
            sqrt_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, alpha, input->numel);
        }
        else if (strcmp(type, "rsqrt") == 0)
        {
            rsqrt_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else
        {
            printf("Activation type \'%s\' not implemented!\n", type);
            exit(1);
        }
    }
    else
    {
#if BACKEND_X86
        if (strcmp(type, "leakyrelu") == 0)
        {
            leakyrelu_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, alpha, input->numel);
        }
        else if (strcmp(type, "relu") == 0)
        {
            relu_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "sigmoid") == 0)
        {
            sigmoid_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "tanh") == 0)
        {
            tanh_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "mish") == 0)
        {
            mish_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "swish") == 0)
        {
            swish_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "hardsigmoid") == 0)
        {
            hardsigmoid_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "hardswish") == 0)
        {
            hardswish_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "exp") == 0)
        {
            exp_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "square") == 0)
        {
            square_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else if (strcmp(type, "sqrt") == 0)
        {
            sqrt_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, alpha, input->numel);
        }
        else if (strcmp(type, "rsqrt") == 0)
        {
            rsqrt_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel);
        }
        else
        {
            printf("Activation type \'%s\' not implemented!\n", type);
            exit(1);
        }
#endif // BACKEND_X86

#if BACKEND_ARM
#endif // BACKEND_ARM
    }
}

NS_MM_F_END
