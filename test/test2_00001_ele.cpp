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

void load_from_txt(char* name, float* data_fp32, int numel, int num_threads_)
{
    FILE* fp = fopen(name, "r");
    if (!fp)
    {
        printf("file %s not exist.\n", name);
        exit(1);
    }

    int bytes = 0;
    bytes = sizeof(float) * numel;
    float* temp = (float*) malloc(bytes);
    const int N = 36;
    char buf[N];
    for (int i = 0; i < numel; i++)
    {
        fgets(buf, N, fp);
        float value = atof(buf);
        temp[i] = value;
    }

    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < numel; i++)
    {
        *(data_fp32 + i) = temp[i];
    }

    free(temp);
    temp = nullptr;
    fclose(fp);
}

template<typename data_t>
void elem4d_NCHW_add_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int n = 0; n < N; n++) {
//        for (int c = 0; c < C; c++) {
//            for (int h = 0; h < H; h++) {
//                for (int w = 0; w < W; w++) {
//                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[((n * C + c) * H + h) * W + w];
//                }
//            }
//        }
//    }

//    #pragma omp parallel for num_threads(num_threads_)
//    for (int n = 0; n < num; n++) {
//        z[n] = x[n] + y[n];
//    }


//    const int elempack = 8;
//    const int num_packs = num / elempack;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int pid = 0; pid < num_packs; pid++) {
//        const float* x_ptr = x + pid * elempack;
//        const float* y_ptr = y + pid * elempack;
//        float* z_ptr = z + pid * elempack;
//        __m256 _a = _mm256_loadu_ps(x_ptr);
//        __m256 _b = _mm256_loadu_ps(y_ptr);
//        __m256 _out = _mm256_add_ps(_a, _b);
//        _mm256_storeu_ps(z_ptr, _out);
//    }


    int w = 8;
    int h = 128;
    int d = 128;
    int channels = 16;
    const int elempack = 8;
    int size = w * h * d * elempack;
    const int num_packs = num / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int q = 0; q < channels; q++) {
        const float* ptr = x + q * size;
        const float* ptr1 = y + q * size;
        float* outptr = z + q * size;
        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr1);
            __m256 _outp = _mm256_add_ps(_p, _p1);
            _mm256_storeu_ps(outptr, _outp);
            ptr += 8;
            ptr1 += 8;
            outptr += 8;
        }
    }
}


template<typename data_t>
void elem4d_NCHW_mul_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W, int R) {
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int n = 0; n < N; n++) {
//        for (int c = 0; c < C; c++) {
//            for (int h = 0; h < H; h++) {
//                for (int w = 0; w < W; w++) {
//                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[((n * C + c) * H + h) * W + w];
//                }
//            }
//        }
//    }

    // 这样更优, 不用计算坐标
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int n = 0; n < num; n++) {
//        z[n] = x[n] * y[n];
//    }


    // x86
    const int elempack = 8;
    const int num_packs = num / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        const float* x_ptr = x + pid * elempack;
        const float* y_ptr = y + pid * elempack;
        float* z_ptr = z + pid * elempack;
        __m256 _a = _mm256_loadu_ps(x_ptr);
        __m256 _b = _mm256_loadu_ps(y_ptr);
        __m256 _out = _mm256_mul_ps(_a, _b);
        _mm256_storeu_ps(z_ptr, _out);
    }
    int offset_ = num_packs * elempack;
    if (num - offset_ >= 4)
    {
        const float* x_ptr = x + offset_;
        const float* y_ptr = y + offset_;
        float* z_ptr = z + offset_;
        __m128 _a = _mm_load_ps(x_ptr);
        __m128 _b = _mm_load_ps(y_ptr);
        __m128 _out = _mm_mul_ps(_a, _b);
        _mm_store_ps(z_ptr, _out);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < num; i++) {
        z[i] = x[i] * y[i];
    }
}

template<typename data_t>
void init_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num) {
    // 这样更优, 不用计算坐标
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int n = 0; n < num; n++) {
//        y[n] = x[n];
//    }

    // x86
    const int elempack = 8;
    const int num_packs = num / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        const float* x_ptr = x + pid * elempack;
        float* y_ptr = y + pid * elempack;
        __m256 _x = _mm256_loadu_ps(x_ptr);
        _mm256_storeu_ps(y_ptr, _x);
    }
    int offset_ = num_packs * elempack;
    if (num - offset_ >= 4)
    {
        const float* x_ptr = x + offset_;
        float* y_ptr = y + offset_;
        __m128 _x = _mm_load_ps(x_ptr);
        _mm_store_ps(y_ptr, _x);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < num; i++) {
        y[i] = x[i];
    }
}

template<typename data_t>
void init_x86_kernel(const int num_threads_, const data_t val, data_t* y, int num) {
    // x86
    const int elempack = 8;
    const int num_packs = num / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        float* y_ptr = y + pid * elempack;
        __m256 _x = _mm256_broadcast_ss(&val);
        _mm256_storeu_ps(y_ptr, _x);
    }
    int offset_ = num_packs * elempack;
    if (num - offset_ >= 4)
    {
        float* y_ptr = y + offset_;
        __m128 _x = _mm_broadcast_ss(&val);
        _mm_store_ps(y_ptr, _x);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < num; i++) {
        y[i] = val;
    }
}


int main(int argc, char** argv)
{
/*
g++ test/test2_00001_ele.cpp -fopenmp -march=native -o test2_00001_ele_fast.out -w -O3

g++ test/test2_00001_ele.cpp -fopenmp -march=native -o test2_00001_ele.out -w

./test2_00001_ele_fast.out

./test2_00001_ele.out


*/
    char* test_name = "00001";
    const int num_threads_ = 12;

    int N = 8;
    int C = 128;
    int H = 128;
    int W = 128;
    int R = 7;

    int numel = N * C * H * W + R;
    int bytes = sizeof(float) * numel;
    float* im2col = (float*) malloc(bytes);
    float* weight = (float*) malloc(bytes);
    float* out_true = (float*) malloc(bytes);
    float* out = (float*) malloc(bytes);

    char file_name[256];
    sprintf(file_name, "test/save_data/%s-x.txt", test_name);
    load_from_txt(file_name, im2col, numel, num_threads_);
    sprintf(file_name, "test/save_data/%s-w.txt", test_name);
    load_from_txt(file_name, weight, numel, num_threads_);
    sprintf(file_name, "test/save_data/%s-y.txt", test_name);
    load_from_txt(file_name, out_true, numel, num_threads_);

    float diff = 99.9;

    printf("======================== calc ========================\n");
//    for (int batch_idx = 0; batch_idx < 8; batch_idx++)
//    {
//        #pragma omp parallel for num_threads(num_threads_)
//        for (int i = 0; i < numel; i++)
//        {
//            *(out + i) = 0.f;
//        }
//        auto startTime = std::chrono::system_clock::now();
//        elem4d_NCHW_add_NCHW_cpp_kernel<float>(num_threads_, im2col, weight, out, numel, N, C, H, W);
//        auto endTime = std::chrono::system_clock::now();
//        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
//        float cost_ms = (float)cost_microseconds / 1000.f;
//        printf("add forward cost_time = %f ms\n", cost_ms);
//    }
//    diff = calc_diff(out, out_true, numel);
//    printf("diff=%f (%s)\n", diff, "y");


    sprintf(file_name, "test/save_data/%s-z.txt", test_name);
    load_from_txt(file_name, out_true, numel, num_threads_);

    for (int batch_idx = 0; batch_idx < 30; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < numel; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
        elem4d_NCHW_mul_NCHW_cpp_kernel<float>(num_threads_, im2col, weight, out, numel, N, C, H, W, R);
//        init_x86_kernel<float>(num_threads_, out_true, out, numel);
//        init_x86_kernel<float>(num_threads_, 3.3f, out, numel);
        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("mul forward cost_time = %f ms\n", cost_ms);
    }
    diff = calc_diff(out, out_true, numel);
    printf("diff=%f (%s)\n", diff, "y");

    free(im2col);
    im2col = nullptr;
    free(weight);
    weight = nullptr;
    free(out);
    out = nullptr;

    return 0;
}