//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <immintrin.h>

float calc_diff(float* x, float* y, int numel)
{
    float diff = 0.f;
    float M = 1.f;
//    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}




template<typename data_t>
void transpose2d_10_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int H, int W) {
    // y[w][h] = x[h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            y[(w * H) + h] = x[(h * W) + w];
        }
    }
}


void matmul_true_cpp_kernel(const int num_threads_, const float* input, const float* weight, float* output, int batch_size, int ch_in, int ch_out) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int oc = 0; oc < ch_out; oc++) {
            for (int ic = 0; ic < ch_in; ic++) {
                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
            }
        }
    }
}


template<typename data_t>
void matmul_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {

    // 以这种方式遍历时，左矩阵是一行一行地遍历，右矩阵是一行一行地遍历，cache命中率高。
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int ic = 0; ic < ch_in; ic++) {
            const float _a = input[bs * ch_in + ic];
            for (int oc = 0; oc < ch_out; oc++) {
                output[bs * ch_out + oc] += _a * weight[ic * ch_out + oc];
            }
        }
    }
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int bs = 0; bs < batch_size; bs++) {
//        for (int oc = 0; oc < ch_out; oc++) {
//            for (int ic = 0; ic < ch_in; ic++) {
//                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
//            }
//        }
//    }


//    int elempack = 8;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int bs = 0; bs < batch_size; bs++) {
//        const float* w_ptr = weight;
//        const float* x_ptr = input + bs * ch_in;
//        for (int ic = 0; ic < ch_in; ic++) {
//            float* out_ptr = output + bs * ch_out;
//            __m256 _a8 = _mm256_broadcast_ss(x_ptr);
//            int oc = 0;
//            for (; oc + (elempack - 1) < ch_out; oc += elempack) {
//                __m256 _b = _mm256_loadu_ps(w_ptr);
//                __m256 _out = _mm256_loadu_ps(out_ptr);
//                _out = _mm256_fmadd_ps(_a8, _b, _out);
//                _mm256_storeu_ps(out_ptr, _out);
//                w_ptr += elempack;
//                out_ptr += elempack;
//            }
//            __m128 _a4 = _mm_broadcast_ss(x_ptr);
//            for (; oc + 3 < ch_out; oc += 4) {
//                __m128 _b = _mm_load_ps(w_ptr);
//                __m128 _out = _mm_load_ps(out_ptr);
//                _out = _mm_fmadd_ps(_a4, _b, _out);
//                _mm_store_ps(out_ptr, _out);
//                w_ptr += 4;
//                out_ptr += 4;
//            }
//            for (; oc < ch_out; oc++) {
//                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
//                w_ptr += 1;
//                out_ptr += 1;
//            }
//            x_ptr++;
//        }
//    }
}



template<typename data_t>
void matmul_transB_cpp_kernel(const int num_threads_, const data_t* input, const data_t* weight_t, data_t* output, int batch_size, int ch_in, int ch_out) {

    // 以这种方式遍历时，左矩阵是一行一行地遍历，右矩阵是一行一行地遍历，cache命中率高。
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int bs = 0; bs < batch_size; bs++) {
//        for (int oc = 0; oc < ch_out; oc++) {
//            for (int ic = 0; ic < ch_in; ic++) {
//                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight_t[oc * ch_in + ic];
//            }
//        }
//    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int ic = 0; ic < ch_in; ic++) {
            const float _a = input[bs * ch_in + ic];
            for (int oc = 0; oc < ch_out; oc++) {
                output[bs * ch_out + oc] += _a * weight_t[oc * ch_in + ic];
            }
        }
    }


}



int main(int argc, char** argv)
{
/*
g++ test/test2_00002_matmul3.cpp -fopenmp -march=native -o test2_00002_matmul3_fast.out -w -O3

./test2_00002_matmul3_fast.out


g++ test/test2_00002_matmul3.cpp -fopenmp -march=native -o test2_00002_matmul3.out -w

./test2_00002_matmul3.out


*/
    const int num_threads_ = 12;



/*
做实验对比
    printf("\n\n======================== calc out = im2col * weight ========================\n");
    printf("\n\n======================== calc out = im2col * weight_t (torch) ========================\n");
谁更快，使用的是朴素矩阵乘法
发现 out = im2col * weight 更快。

*/

//    int batch_size = 8400;
//    int ch_in = 512;
//    int ch_out = 512;

    // 骨干网络前面的卷积
//    int batch_size = 65536;
//    int ch_in = 64;
//    int ch_out = 64;

//    int batch_size = 65536;
//    int ch_in = 64*9;
//    int ch_out = 64;

    // 骨干网络中间的卷积
//    int batch_size = 6400;
//    int ch_in = 256;
//    int ch_out = 256;

    int batch_size = 6400;
    int ch_in = 256*9;
    int ch_out = 256;

    // 骨干网络后面的卷积
//    int batch_size = 400;
//    int ch_in = 1024;
//    int ch_out = 1024;

//    int batch_size = 400;
//    int ch_in = 1024*9;
//    int ch_out = 1024;

    // 最后的卷积
//    int batch_size = 400;
//    int ch_in = 1024;
//    int ch_out = 80;


//    float* im2col = (float*) malloc(batch_size * ch_in);
//    float* weight = (float*) malloc(ch_in * ch_out);
//    float* out_true = (float*) malloc(batch_size * ch_out);
//    float* out = (float*) malloc(batch_size * ch_out);

    float* im2col = new float[batch_size * ch_in];
    float* weight = new float[ch_in * ch_out];
    float* out_true = new float[batch_size * ch_out];
    float* out = new float[batch_size * ch_out];

//    float* im2col_t = new float[batch_size * ch_in];
    float* weight_t = new float[ch_in * ch_out];
//    float* out_t_true = new float[batch_size * ch_out];
//    float* out_t = new float[batch_size * ch_out];

    char file_name[256];
    printf("======================== init ========================\n");
    printf("im2col      = ");
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < batch_size * ch_in; i++)
    {
        int ttt = rand() % 2000;
        float val = (float)ttt / 1000.f - 1.f;
//        printf("%f,", val);
        *(im2col + i) = val;
    }
    printf("\nweight      = ");
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < ch_in * ch_out; i++)
    {
        int ttt = rand() % 2000;
        float val = (float)ttt / 1000.f - 1.f;
//        printf("%f,", val);
        *(weight + i) = val;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < batch_size * ch_out; i++)
    {
        *(out_true + i) = 0.f;
    }
    matmul_true_cpp_kernel(num_threads_, im2col, weight, out_true, batch_size, ch_in, ch_out);

//    transpose2d_10_cpp_kernel<float>(num_threads_, im2col, im2col_t, batch_size * ch_in, batch_size, ch_in);
    transpose2d_10_cpp_kernel<float>(num_threads_, weight, weight_t, ch_in * ch_out, ch_in, ch_out);
//    transpose2d_10_cpp_kernel<float>(num_threads_, out_true, out_t_true, batch_size * ch_out, batch_size, ch_out);


    float diff = 0.0;

    printf("\n\n======================== calc out = im2col * weight ========================\n");
    for (int batch_idx = 0; batch_idx < 3; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < batch_size * ch_out; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
        matmul_cpp_kernel<float>(num_threads_, im2col, weight, out, batch_size, ch_in, ch_out);
        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("matmul forward cost_time = %f ms\n", cost_ms);
        diff = calc_diff(out, out_true, batch_size * ch_out);
        printf("diff=%f (%s)\n", diff, "y");
    }

    printf("\n\n======================== calc out = im2col * weight_t (torch) ========================\n");
    for (int batch_idx = 0; batch_idx < 30; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < batch_size * ch_out; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
        matmul_transB_cpp_kernel<float>(num_threads_, im2col, weight_t, out, batch_size, ch_in, ch_out);
        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("matmul forward cost_time = %f ms\n", cost_ms);
        diff = calc_diff(out, out_true, batch_size * ch_out);
        printf("diff=%f (%s)\n", diff, "y");
    }


    diff = calc_diff(out, out_true, batch_size * ch_out);
    printf("diff=%f (%s)\n", diff, "y");

    delete im2col;
    delete weight;
    delete out;
    delete out_true;

    return 0;
}