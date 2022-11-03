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
void imNHWC2col_cpp_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * kH * kW * in_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        // im.shape = [N, H, W, in_C]
                        // 求出对应的im元素的坐标 n h w ic
                        int h_in = oh * stride_h - padding_h;
                        int w_in = ow * stride_w - padding_w;
                        const int h = h_in + kh * dilation_h;
                        const int w = w_in + kw * dilation_w;

                        // 越界取0，否则取im[n, h, w, ic]
                        const bool cond = h > -1 && w > -1 && h < H && w < W;
                        for (int ic = 0; ic < in_C; ic++) {
                            float val = cond ? im[(((n * H) + h) * W + w) * in_C + ic] : 0.f;
//                            im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + kw) * in_C + ic)] = val;
                            im2col[i++] = val;
                        }
                    }
                }
            }
        }
    }
}





template<typename data_t>
void imNHWC2col_x86_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    int elempack = 8;
//    const float zeros8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
//    const float* zeros8_ptr = zeros8;
//    const float zeros4[4] = {0.f, 0.f, 0.f, 0.f};
//    const float* zeros4_ptr = zeros4;

    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * kH * kW * in_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        int h_in = oh * stride_h - padding_h;
                        int w_in = ow * stride_w - padding_w;
                        const int h = h_in + kh * dilation_h;
                        const int w = w_in + kw * dilation_w;
                        const bool cond = h > -1 && w > -1 && h < H && w < W;
                        int ic = 0;
                        for (; ic + (elempack - 1) < in_C; ic += elempack) {
                            __m256 _val = _mm256_setzero_ps();
                            if (cond)
                                _val = _mm256_loadu_ps(im + (((n * H) + h) * W + w) * in_C + ic);
                            _mm256_storeu_ps(im2col + i, _val);
                            i += elempack;
                        }
                        for (; ic + 3 < in_C; ic += 4) {
                            __m128 _val = _mm_setzero_ps();
                            if (cond)
                                _val = _mm_load_ps(im + (((n * H) + h) * W + w) * in_C + ic);
                            _mm_store_ps(im2col + i, _val);
                            i += 4;
                        }
                        for (; ic < in_C; ic++) {
                            float val = cond ? im[(((n * H) + h) * W + w) * in_C + ic] : 0.f;
                            im2col[i++] = val;
                        }
                    }
                }
            }
        }
    }
}



// 输出中有输入的重复元素
// 优化方向是，对输入元素只读1次，然后多次保存。
// 这里先只实现在kH上的读1次保存多次。kH
template<typename data_t>
void imNHWC2col_k3s1d1_only_kW_cpp_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * kH * kW * in_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int kh = 0; kh < kH; kh++) {
                    const int x00_h = oh - padding_h + kh;
                    const int x00_w = ow - padding_w + 0;
                    const int x01_h = oh - padding_h + kh;
                    const int x01_w = ow - padding_w + 1;
                    const int x02_h = oh - padding_h + kh;
                    const int x02_w = ow - padding_w + 2;


                    if (ow == 0)
                    {
                        const bool cond00 = x00_h > -1 && x00_w > -1 && x00_h < H && x00_w < W;
                        for (int ic = 0; ic < in_C; ic++) {
                            float val = cond00 ? im[(((n * H) + x00_h) * W + x00_w) * in_C + ic] : 0.f;
                            im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 0) * in_C + ic)] = val;
                        }
                        const bool cond01 = x01_h > -1 && x01_w > -1 && x01_h < H && x01_w < W;
                        for (int ic = 0; ic < in_C; ic++) {
                            float val = cond01 ? im[(((n * H) + x01_h) * W + x01_w) * in_C + ic] : 0.f;
                            im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 1) * in_C + ic)] = val;
                            // 当前卷积核框住的像素块右边的3*2区域，写进下一次滑动后左边3*2区域，im2col out_W维度坐标+1，即ow + 1，以及 kW维的坐标-1
                            if (ow + 1 < out_W)
                                im2col[((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 1 - 1) * in_C + ic)] = val;
                        }
                    }
                    const bool cond02 = x02_h > -1 && x02_w > -1 && x02_h < H && x02_w < W;
                    for (int ic = 0; ic < in_C; ic++) {
                        float val = cond02 ? im[(((n * H) + x02_h) * W + x02_w) * in_C + ic] : 0.f;
                        im2col[((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 2) * in_C + ic)] = val;
                        // 当前卷积核框住的像素块右边的3*2区域，写进下一次滑动后左边3*2区域，im2col out_W维度坐标+1，即ow + 1，以及 kW维的坐标-1
                        if (ow + 1 < out_W)
                            im2col[((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 2 - 1) * in_C + ic)] = val;
                        // 当前卷积核框住的像素块右边的3*1区域，写进下下次滑动后左边3*1区域，im2col out_W维度坐标+2，即ow + 2，以及 kW维的坐标-2
                        if (ow + 2 < out_W)
                            im2col[((((((n * out_H) + oh) * out_W + ow + 2) * kH + kh) * kW + 2 - 2) * in_C + ic)] = val;
                    }


                }
            }
        }
    }
}


template<typename data_t>
void imNHWC2col_k3s1d1_only_kW_v2_cpp_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * kH * kW * in_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int kh = 0; kh < kH; kh++) {
                    const int x00_h = oh - padding_h + kh;
                    const int x00_w = ow - padding_w + 0;
                    const int x01_h = oh - padding_h + kh;
                    const int x01_w = ow - padding_w + 1;
                    const int x02_h = oh - padding_h + kh;
                    const int x02_w = ow - padding_w + 2;


                    if (ow == 0)
                    {
                        const bool cond00 = x00_h > -1 && x00_w > -1 && x00_h < H && x00_w < W;
                        int im_index = (((n * H) + x00_h) * W + x00_w) * in_C + 0;
                        int index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 0) * in_C + 0);
                        for (int ic = 0; ic < in_C; ic++) {
                            float val = cond00 ? im[im_index] : 0.f;
                            im_index++;
                            im2col[index0++] = val;
                        }

                        const bool cond01 = x01_h > -1 && x01_w > -1 && x01_h < H && x01_w < W;
                        im_index = (((n * H) + x01_h) * W + x01_w) * in_C + 0;
                        index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 1) * in_C + 0);
                        int index1 = ((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 1 - 1) * in_C + 0);
                        for (int ic = 0; ic < in_C; ic++) {
                            float val = cond01 ? im[im_index] : 0.f;
                            im_index++;
                            im2col[index0++] = val;
                            if (ow + 1 < out_W)
                                im2col[index1++] = val;
                        }
                    }
                    const bool cond02 = x02_h > -1 && x02_w > -1 && x02_h < H && x02_w < W;
                    int im_index = (((n * H) + x02_h) * W + x02_w) * in_C + 0;
                    int index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 2) * in_C + 0);
                    int index1 = ((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 2 - 1) * in_C + 0);
                    int index2 = ((((((n * out_H) + oh) * out_W + ow + 2) * kH + kh) * kW + 2 - 2) * in_C + 0);
                    for (int ic = 0; ic < in_C; ic++) {
                        float val = cond02 ? im[im_index] : 0.f;
                        im_index++;
                        im2col[index0++] = val;
                        if (ow + 1 < out_W)
                            im2col[index1++] = val;
                        if (ow + 2 < out_W)
                            im2col[index2++] = val;
                    }


                }
            }
        }
    }
}


// 与imNHWC2col_x86_kernel差不多速度，优化了个寂寞。或许也许可能kH也一起读一次保存多次，才有加速。
template<typename data_t>
void imNHWC2col_k3s1d1_only_kW_v3_x86_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    int elempack = 8;
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * kH * kW * in_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int kh = 0; kh < kH; kh++) {
                    const int x00_h = oh - padding_h + kh;
                    const int x00_w = ow - padding_w + 0;
                    const int x01_h = oh - padding_h + kh;
                    const int x01_w = ow - padding_w + 1;
                    const int x02_h = oh - padding_h + kh;
                    const int x02_w = ow - padding_w + 2;


                    if (ow == 0)
                    {
                        const bool cond00 = x00_h > -1 && x00_w > -1 && x00_h < H && x00_w < W;
                        int im_index = (((n * H) + x00_h) * W + x00_w) * in_C + 0;
                        int index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 0) * in_C + 0);
//                        for (int ic = 0; ic < in_C; ic++) {
//                            float val = cond00 ? im[im_index] : 0.f;
//                            im_index++;
//                            im2col[index0++] = val;
//                        }
                        int ic = 0;
                        for (; ic + (elempack - 1) < in_C; ic += elempack) {
                            __m256 _val = _mm256_setzero_ps();
                            if (cond00)
                                _val = _mm256_loadu_ps(im + im_index);
                            _mm256_storeu_ps(im2col + index0, _val);
                            index0 += elempack;
                            im_index += elempack;
                        }
                        for (; ic + 3 < in_C; ic += 4) {
                            __m128 _val = _mm_setzero_ps();
                            if (cond00)
                                _val = _mm_load_ps(im + im_index);
                            _mm_store_ps(im2col + index0, _val);
                            index0 += 4;
                            im_index += 4;
                        }
                        for (; ic < in_C; ic++) {
                            float val = cond00 ? im[im_index] : 0.f;
                            im2col[index0++] = val;
                            im_index++;
                        }



                        const bool cond01 = x01_h > -1 && x01_w > -1 && x01_h < H && x01_w < W;
                        im_index = (((n * H) + x01_h) * W + x01_w) * in_C + 0;
                        index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 1) * in_C + 0);
                        int index1 = ((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 1 - 1) * in_C + 0);
//                        for (int ic = 0; ic < in_C; ic++) {
//                            float val = cond01 ? im[im_index] : 0.f;
//                            im_index++;
//                            im2col[index0++] = val;
//                            if (ow + 1 < out_W)
//                                im2col[index1++] = val;
//                        }

                        ic = 0;
                        for (; ic + (elempack - 1) < in_C; ic += elempack) {
                            __m256 _val = _mm256_setzero_ps();
                            if (cond01)
                                _val = _mm256_loadu_ps(im + im_index);
                            _mm256_storeu_ps(im2col + index0, _val);
                            index0 += elempack;
                            im_index += elempack;
                            if (ow + 1 < out_W) {
                                _mm256_storeu_ps(im2col + index1, _val);
                                index1 += elempack;
                            }
                        }
                        for (; ic + 3 < in_C; ic += 4) {
                            __m128 _val = _mm_setzero_ps();
                            if (cond01)
                                _val = _mm_load_ps(im + im_index);
                            _mm_store_ps(im2col + index0, _val);
                            index0 += 4;
                            im_index += 4;
                            if (ow + 1 < out_W) {
                                _mm_store_ps(im2col + index1, _val);
                                index1 += 4;
                            }
                        }
                        for (; ic < in_C; ic++) {
                            float val = cond01 ? im[im_index] : 0.f;
                            im2col[index0++] = val;
                            im_index++;
                            if (ow + 1 < out_W) {
                                im2col[index1++] = val;
                            }
                        }
                    }
                    const bool cond02 = x02_h > -1 && x02_w > -1 && x02_h < H && x02_w < W;
                    int im_index = (((n * H) + x02_h) * W + x02_w) * in_C + 0;
                    int index0 = ((((((n * out_H) + oh) * out_W + ow) * kH + kh) * kW + 2) * in_C + 0);
                    int index1 = ((((((n * out_H) + oh) * out_W + ow + 1) * kH + kh) * kW + 2 - 1) * in_C + 0);
                    int index2 = ((((((n * out_H) + oh) * out_W + ow + 2) * kH + kh) * kW + 2 - 2) * in_C + 0);
//                    for (int ic = 0; ic < in_C; ic++) {
//                        float val = cond02 ? im[im_index] : 0.f;
//                        im_index++;
//                        im2col[index0++] = val;
//                        if (ow + 1 < out_W)
//                            im2col[index1++] = val;
//                        if (ow + 2 < out_W)
//                            im2col[index2++] = val;
//                    }

                    int ic = 0;
                    for (; ic + (elempack - 1) < in_C; ic += elempack) {
                        __m256 _val = _mm256_setzero_ps();
                        if (cond02)
                            _val = _mm256_loadu_ps(im + im_index);
                        _mm256_storeu_ps(im2col + index0, _val);
                        index0 += elempack;
                        im_index += elempack;
                        if (ow + 1 < out_W) {
                            _mm256_storeu_ps(im2col + index1, _val);
                            index1 += elempack;
                        }
                        if (ow + 2 < out_W) {
                            _mm256_storeu_ps(im2col + index2, _val);
                            index2 += elempack;
                        }
                    }
                    for (; ic + 3 < in_C; ic += 4) {
                        __m128 _val = _mm_setzero_ps();
                        if (cond02)
                            _val = _mm_load_ps(im + im_index);
                        _mm_store_ps(im2col + index0, _val);
                        index0 += 4;
                        im_index += 4;
                        if (ow + 1 < out_W) {
                            _mm_store_ps(im2col + index1, _val);
                            index1 += 4;
                        }
                        if (ow + 2 < out_W) {
                            _mm_store_ps(im2col + index2, _val);
                            index2 += 4;
                        }
                    }
                    for (; ic < in_C; ic++) {
                        float val = cond02 ? im[im_index] : 0.f;
                        im2col[index0++] = val;
                        im_index++;
                        if (ow + 1 < out_W) {
                            im2col[index1++] = val;
                        }
                        if (ow + 2 < out_W) {
                            im2col[index2++] = val;
                        }
                    }


                }
            }
        }
    }
}





int main(int argc, char** argv)
{
/*
g++ test/test2_00002_matmul2.cpp -fopenmp -march=native -o test2_00002_matmul2_fast.out -w -O3

./test2_00002_matmul2_fast.out


g++ test/test2_00002_matmul2.cpp -fopenmp -march=native -o test2_00002_matmul2.out -w

./test2_00002_matmul2.out


*/
    const int num_threads_ = 12;

    int N = 1;
    int H = 512;
    int W = 512;

//    int N = 4;
//    int H = 64;
//    int W = 64;

    int in_C = 128;
    int kH = 3;
    int kW = 3;
    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 1;
    int padding_w = 1;



    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;

    if (kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0 && groups == 1)
    {
    }


    const int kernel_extent_h = dilation_h * (kH - 1) + 1;
    const int kernel_extent_w = dilation_w * (kW - 1) + 1;
    const int out_H = (H + padding_h + padding_h - kernel_extent_h) / stride_h + 1;
    const int out_W = (W + padding_w + padding_w - kernel_extent_w) / stride_w + 1;

    const int input_numel = N * H * W * in_C;
    const int out_numel = N * out_H * out_W * kH * kW * in_C;
    float* input = new float[input_numel];
    float* out_true = new float[out_numel];
    float* out = new float[out_numel];


    printf("======================== init ========================\n");
    printf("input      = ");
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < input_numel; i++)
    {
        int ttt = rand() % 2000;
        float val = (float)ttt / 1000.f - 1.f;
//        printf("%f,", val);
        *(input + i) = val;
    }
    imNHWC2col_cpp_kernel<float>(num_threads_, input, out_true, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);



    float diff = 0.0;


    printf("======================== calc ========================\n");
    for (int batch_idx = 0; batch_idx < 10; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < out_numel; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
//        imNHWC2col_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
        imNHWC2col_x86_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("matmul forward cost_time = %f ms\n", cost_ms);
        diff = calc_diff(out, out_true, out_numel);
        printf("diff=%f (%s)\n", diff, "y");
    }

    printf("======================== calc ========================\n");
    for (int batch_idx = 0; batch_idx < 10; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < out_numel; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();
//        imNHWC2col_k3s1d1_only_kW_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//        imNHWC2col_k3s1d1_only_kW_v2_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
        imNHWC2col_k3s1d1_only_kW_v3_x86_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//        imNHWC2col_k3s1d1_only_kW_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//        imNHWC2col_k3s1d1_only_kW_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//        imNHWC2col_k3s1d1_only_kW_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//        imNHWC2col_k3s1d1_only_kW_cpp_kernel<float>(num_threads_, input, out, out_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("matmul forward cost_time = %f ms\n", cost_ms);
        diff = calc_diff(out, out_true, out_numel);
        printf("diff=%f (%s)\n", diff, "y");
//        printf("out      = ");
//        for (int i = 0; i < out_numel; i++)
//        {
//            printf("%f,", out[i]);
//        }
//        printf("\nout_true = ");
//        for (int i = 0; i < batch_size * ch_out; i++)
//        {
//            printf("%f,", out_true[i]);
//        }
//        printf("\n-----------------------------------------------\n");
    }


    diff = calc_diff(out, out_true, out_numel);
    printf("diff=%f (%s)\n", diff, "y");

    delete input;
    delete out;
    delete out_true;

    return 0;
}