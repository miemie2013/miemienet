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
    //    float M = 1.f;
    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}

void load_from_txt(char* name, float* data_fp32, int numel, int num_threads_)
{
    /*FILE* fp = fopen(name, "r");
    if (!fp)
    {
        printf("file %s not exist.\n", name);
        exit(1);
    }*/

    int bytes = 0;
    bytes = sizeof(float) * numel;
    float* temp = (float*)malloc(bytes);
    const int N = 36;
    char buf[N];
    for (int i = 0; i < numel; i++)
    {
        /*fgets(buf, N, fp);
        float value = atof(buf);*/
        float value = 0.f;
        temp[i] = value;
    }

#pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < numel; i++)
    {
        *(data_fp32 + i) = temp[i];
    }

    free(temp);
    temp = nullptr;
    //fclose(fp);
}

void matmul(float* A, float* B, float* C, int batch_size, int ch_in, int ch_out, int num_threads_)
{
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int ic = 0; ic < ch_in; ic++) {
            const float _a = A[bs * ch_in + ic];
            for (int oc = 0; oc < ch_out; oc++) {
                C[bs * ch_out + oc] += _a * B[ic * ch_out + oc];
            }
        }
    }

        /*int elempack = 8;
        #pragma omp parallel for num_threads(num_threads_)
        for (int bs = 0; bs < batch_size; bs++) {
            const float* w_ptr = B;
            const float* x_ptr = A + bs * ch_in;
            for (int ic = 0; ic < ch_in; ic++) {
                float* out_ptr = C + bs * ch_out;
                __m256 _a = _mm256_broadcast_ss(x_ptr);
                for (int oc = 0; oc < ch_out; oc += elempack) {
                    __m256 _b = _mm256_loadu_ps(w_ptr);
                    __m256 _out = _mm256_loadu_ps(out_ptr);
                    _out = _mm256_fmadd_ps(_a, _b, _out);
                    _mm256_storeu_ps(out_ptr, _out);
                    w_ptr += elempack;
                    out_ptr += elempack;
                }
                x_ptr++;
            }
        }*/
}

int main(int argc, char** argv)
{
/*
g++ test/test2_00000_matmul.cpp -fopenmp -march=native -o test2_00000_matmul.out -w -Ofast

g++ test/test2_00000_matmul.cpp -fopenmp -march=native -o test2_00000_matmul.out -w

./test2_00000_matmul.out

mkdir build

cd build

rm -f CMakeCache.txt && cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_CXX_COMPILER=/usr/bin/g++


Windows下安装MinGW，编译c/c++时出现cannot find -lpthread解决办法:
http://www.javashuo.com/article/p-axxxlbvh-kn.html

g++ test/test2_00000_matmul.cpp -fopenmp -march=native -o test2_00000_matmul.exe -w -Ofast

g++ test/test2_00000_matmul.cpp -fopenmp -march=native -o test2_00000_matmul.exe -w

cl test/test2_00000_matmul.cpp /JMC /Zi


cl test/test2_00000_matmul.cpp /Yc /Yu



./test2_00000_matmul.exe


*/
    char test_name[6] = "00000";
    const int num_threads_ = 12;

    int batch_size = 2;
    int output_size = 256;
    int in_features = 64;
    int kH = 3;
    int kW = 3;
    int out_features = 128;

    int M = batch_size * output_size * output_size;
    int K = in_features * kH * kW;
    int N = out_features;

    //int M = 640;
    //int K = 608;
    //int N = 512;

    int bytes = sizeof(float) * M * K;
    float* im2col = (float*)malloc(bytes);
    bytes = sizeof(float) * K * N;
    float* weight = (float*)malloc(bytes);
    bytes = sizeof(float) * M * N;
    float* out_true = (float*)malloc(bytes);
    float* out = (float*)malloc(bytes);

    char file_name[256];
    sprintf(file_name, "test/save_data/%s-x.txt", test_name);
    load_from_txt(file_name, im2col, M * K, num_threads_);
    sprintf(file_name, "test/save_data/%s-w.txt", test_name);
    load_from_txt(file_name, weight, K * N, num_threads_);
    sprintf(file_name, "test/save_data/%s-y.txt", test_name);
    load_from_txt(file_name, out_true, M * N, num_threads_);

    printf("======================== calc ========================\n");
    for (int batch_idx = 0; batch_idx < 8; batch_idx++)
    {
#pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < M * N; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
        matmul(im2col, weight, out, M, K, N, num_threads_);
        auto endTime = std::chrono::system_clock::now();
        // 1秒=1000毫秒=1000,000微秒
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("eval forward cost_time = %f ms\n", cost_ms);

        float diff = calc_diff(out, out_true, M * N);
        printf("diff=%f (%s)\n", diff, "y");
    }
    free(im2col);
    im2col = nullptr;
    free(weight);
    weight = nullptr;
    free(out);
    out = nullptr;

    return 0;
}