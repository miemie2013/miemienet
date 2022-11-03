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
void matmul_block_pack_8_8_8_SIMD_consider_mod_x86_kernel(const int num_threads_, const data_t* input, const data_t* weight, data_t* output, int batch_size, int ch_in, int ch_out) {
    const int B_mod = batch_size % 8;
    const int I_mod = ch_in % 8;
    const int O_mod = ch_out % 8;
    const int pack4_offset = (batch_size / 8) * 8;
    const int pack1_offset = (batch_size / 4) * 4;
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < pack4_offset; bs+=8) {
        int ic = 0;
        for (; ic + 7 < ch_in; ic+=8) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);
            __m256 _e0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 4);
            __m256 _f0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 5);
            __m256 _g0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 6);
            __m256 _h0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 7);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
            __m256 _e1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
            __m256 _f1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
            __m256 _g1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
            __m256 _h1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
            __m256 _e2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
            __m256 _f2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
            __m256 _g2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
            __m256 _h2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
            __m256 _e3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
            __m256 _f3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
            __m256 _g3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
            __m256 _h3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _b4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
            __m256 _c4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
            __m256 _d4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);
            __m256 _e4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 4);
            __m256 _f4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 5);
            __m256 _g4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 6);
            __m256 _h4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 7);

            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _b5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
            __m256 _c5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
            __m256 _d5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);
            __m256 _e5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 4);
            __m256 _f5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 5);
            __m256 _g5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 6);
            __m256 _h5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 7);

            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _b6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
            __m256 _c6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
            __m256 _d6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);
            __m256 _e6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 4);
            __m256 _f6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 5);
            __m256 _g6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 6);
            __m256 _h6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 7);

            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);
            __m256 _b7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
            __m256 _c7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
            __m256 _d7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);
            __m256 _e7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 4);
            __m256 _f7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 5);
            __m256 _g7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 6);
            __m256 _h7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 7);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);
                __m256 _w4 = _mm256_loadu_ps(w_ptr4);
                __m256 _w5 = _mm256_loadu_ps(w_ptr5);
                __m256 _w6 = _mm256_loadu_ps(w_ptr6);
                __m256 _w7 = _mm256_loadu_ps(w_ptr7);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _y0 = _mm256_fmadd_ps(_e0, _w4, _y0);
                _y0 = _mm256_fmadd_ps(_f0, _w5, _y0);
                _y0 = _mm256_fmadd_ps(_g0, _w6, _y0);
                _y0 = _mm256_fmadd_ps(_h0, _w7, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _y1 = _mm256_fmadd_ps(_e1, _w4, _y1);
                _y1 = _mm256_fmadd_ps(_f1, _w5, _y1);
                _y1 = _mm256_fmadd_ps(_g1, _w6, _y1);
                _y1 = _mm256_fmadd_ps(_h1, _w7, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _y2 = _mm256_fmadd_ps(_e2, _w4, _y2);
                _y2 = _mm256_fmadd_ps(_f2, _w5, _y2);
                _y2 = _mm256_fmadd_ps(_g2, _w6, _y2);
                _y2 = _mm256_fmadd_ps(_h2, _w7, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _y3 = _mm256_fmadd_ps(_e3, _w4, _y3);
                _y3 = _mm256_fmadd_ps(_f3, _w5, _y3);
                _y3 = _mm256_fmadd_ps(_g3, _w6, _y3);
                _y3 = _mm256_fmadd_ps(_h3, _w7, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _y4 = _mm256_fmadd_ps(_b4, _w1, _y4);
                _y4 = _mm256_fmadd_ps(_c4, _w2, _y4);
                _y4 = _mm256_fmadd_ps(_d4, _w3, _y4);
                _y4 = _mm256_fmadd_ps(_e4, _w4, _y4);
                _y4 = _mm256_fmadd_ps(_f4, _w5, _y4);
                _y4 = _mm256_fmadd_ps(_g4, _w6, _y4);
                _y4 = _mm256_fmadd_ps(_h4, _w7, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _y5 = _mm256_fmadd_ps(_b5, _w1, _y5);
                _y5 = _mm256_fmadd_ps(_c5, _w2, _y5);
                _y5 = _mm256_fmadd_ps(_d5, _w3, _y5);
                _y5 = _mm256_fmadd_ps(_e5, _w4, _y5);
                _y5 = _mm256_fmadd_ps(_f5, _w5, _y5);
                _y5 = _mm256_fmadd_ps(_g5, _w6, _y5);
                _y5 = _mm256_fmadd_ps(_h5, _w7, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _y6 = _mm256_fmadd_ps(_b6, _w1, _y6);
                _y6 = _mm256_fmadd_ps(_c6, _w2, _y6);
                _y6 = _mm256_fmadd_ps(_d6, _w3, _y6);
                _y6 = _mm256_fmadd_ps(_e6, _w4, _y6);
                _y6 = _mm256_fmadd_ps(_f6, _w5, _y6);
                _y6 = _mm256_fmadd_ps(_g6, _w6, _y6);
                _y6 = _mm256_fmadd_ps(_h6, _w7, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _y7 = _mm256_fmadd_ps(_b7, _w1, _y7);
                _y7 = _mm256_fmadd_ps(_c7, _w2, _y7);
                _y7 = _mm256_fmadd_ps(_d7, _w3, _y7);
                _y7 = _mm256_fmadd_ps(_e7, _w4, _y7);
                _y7 = _mm256_fmadd_ps(_f7, _w5, _y7);
                _y7 = _mm256_fmadd_ps(_g7, _w6, _y7);
                _y7 = _mm256_fmadd_ps(_h7, _w7, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);
                __m128 _w4 = _mm_load_ps(w_ptr4);
                __m128 _w5 = _mm_load_ps(w_ptr5);
                __m128 _w6 = _mm_load_ps(w_ptr6);
                __m128 _w7 = _mm_load_ps(w_ptr7);


                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);
                __m128 __e0 = _mm_broadcast_ss(input + bs * ch_in + ic + 4);
                __m128 __f0 = _mm_broadcast_ss(input + bs * ch_in + ic + 5);
                __m128 __g0 = _mm_broadcast_ss(input + bs * ch_in + ic + 6);
                __m128 __h0 = _mm_broadcast_ss(input + bs * ch_in + ic + 7);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
                __m128 __e1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
                __m128 __f1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
                __m128 __g1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
                __m128 __h1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
                __m128 __e2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
                __m128 __f2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
                __m128 __g2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
                __m128 __h2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
                __m128 __e3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
                __m128 __f3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
                __m128 __g3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
                __m128 __h3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

                __m128 __a4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic);
                __m128 __b4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
                __m128 __c4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
                __m128 __d4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);
                __m128 __e4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 4);
                __m128 __f4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 5);
                __m128 __g4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 6);
                __m128 __h4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 7);

                __m128 __a5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic);
                __m128 __b5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
                __m128 __c5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
                __m128 __d5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);
                __m128 __e5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 4);
                __m128 __f5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 5);
                __m128 __g5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 6);
                __m128 __h5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 7);

                __m128 __a6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic);
                __m128 __b6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
                __m128 __c6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
                __m128 __d6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);
                __m128 __e6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 4);
                __m128 __f6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 5);
                __m128 __g6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 6);
                __m128 __h6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 7);

                __m128 __a7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic);
                __m128 __b7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
                __m128 __c7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
                __m128 __d7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);
                __m128 __e7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 4);
                __m128 __f7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 5);
                __m128 __g7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 6);
                __m128 __h7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 7);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _y0 = _mm_fmadd_ps(__e0, _w4, _y0);
                _y0 = _mm_fmadd_ps(__f0, _w5, _y0);
                _y0 = _mm_fmadd_ps(__g0, _w6, _y0);
                _y0 = _mm_fmadd_ps(__h0, _w7, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _y1 = _mm_fmadd_ps(__e1, _w4, _y1);
                _y1 = _mm_fmadd_ps(__f1, _w5, _y1);
                _y1 = _mm_fmadd_ps(__g1, _w6, _y1);
                _y1 = _mm_fmadd_ps(__h1, _w7, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _y2 = _mm_fmadd_ps(__e2, _w4, _y2);
                _y2 = _mm_fmadd_ps(__f2, _w5, _y2);
                _y2 = _mm_fmadd_ps(__g2, _w6, _y2);
                _y2 = _mm_fmadd_ps(__h2, _w7, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _y3 = _mm_fmadd_ps(__e3, _w4, _y3);
                _y3 = _mm_fmadd_ps(__f3, _w5, _y3);
                _y3 = _mm_fmadd_ps(__g3, _w6, _y3);
                _y3 = _mm_fmadd_ps(__h3, _w7, _y3);
                _mm_store_ps(y_ptr3, _y3);

                __m128 _y4 = _mm_load_ps(y_ptr4);
                _y4 = _mm_fmadd_ps(__a4, _w0, _y4);
                _y4 = _mm_fmadd_ps(__b4, _w1, _y4);
                _y4 = _mm_fmadd_ps(__c4, _w2, _y4);
                _y4 = _mm_fmadd_ps(__d4, _w3, _y4);
                _y4 = _mm_fmadd_ps(__e4, _w4, _y4);
                _y4 = _mm_fmadd_ps(__f4, _w5, _y4);
                _y4 = _mm_fmadd_ps(__g4, _w6, _y4);
                _y4 = _mm_fmadd_ps(__h4, _w7, _y4);
                _mm_store_ps(y_ptr4, _y4);

                __m128 _y5 = _mm_load_ps(y_ptr5);
                _y5 = _mm_fmadd_ps(__a5, _w0, _y5);
                _y5 = _mm_fmadd_ps(__b5, _w1, _y5);
                _y5 = _mm_fmadd_ps(__c5, _w2, _y5);
                _y5 = _mm_fmadd_ps(__d5, _w3, _y5);
                _y5 = _mm_fmadd_ps(__e5, _w4, _y5);
                _y5 = _mm_fmadd_ps(__f5, _w5, _y5);
                _y5 = _mm_fmadd_ps(__g5, _w6, _y5);
                _y5 = _mm_fmadd_ps(__h5, _w7, _y5);
                _mm_store_ps(y_ptr5, _y5);

                __m128 _y6 = _mm_load_ps(y_ptr6);
                _y6 = _mm_fmadd_ps(__a6, _w0, _y6);
                _y6 = _mm_fmadd_ps(__b6, _w1, _y6);
                _y6 = _mm_fmadd_ps(__c6, _w2, _y6);
                _y6 = _mm_fmadd_ps(__d6, _w3, _y6);
                _y6 = _mm_fmadd_ps(__e6, _w4, _y6);
                _y6 = _mm_fmadd_ps(__f6, _w5, _y6);
                _y6 = _mm_fmadd_ps(__g6, _w6, _y6);
                _y6 = _mm_fmadd_ps(__h6, _w7, _y6);
                _mm_store_ps(y_ptr6, _y6);

                __m128 _y7 = _mm_load_ps(y_ptr7);
                _y7 = _mm_fmadd_ps(__a7, _w0, _y7);
                _y7 = _mm_fmadd_ps(__b7, _w1, _y7);
                _y7 = _mm_fmadd_ps(__c7, _w2, _y7);
                _y7 = _mm_fmadd_ps(__d7, _w3, _y7);
                _y7 = _mm_fmadd_ps(__e7, _w4, _y7);
                _y7 = _mm_fmadd_ps(__f7, _w5, _y7);
                _y7 = _mm_fmadd_ps(__g7, _w6, _y7);
                _y7 = _mm_fmadd_ps(__h7, _w7, _y7);
                _mm_store_ps(y_ptr7, _y7);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);
                float _w4 = *(w_ptr4);
                float _w5 = *(w_ptr5);
                float _w6 = *(w_ptr6);
                float _w7 = *(w_ptr7);


                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);
                float __e0 = *(input + bs * ch_in + ic + 4);
                float __f0 = *(input + bs * ch_in + ic + 5);
                float __g0 = *(input + bs * ch_in + ic + 6);
                float __h0 = *(input + bs * ch_in + ic + 7);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);
                float __e1 = *(input + (bs + 1) * ch_in + ic + 4);
                float __f1 = *(input + (bs + 1) * ch_in + ic + 5);
                float __g1 = *(input + (bs + 1) * ch_in + ic + 6);
                float __h1 = *(input + (bs + 1) * ch_in + ic + 7);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);
                float __e2 = *(input + (bs + 2) * ch_in + ic + 4);
                float __f2 = *(input + (bs + 2) * ch_in + ic + 5);
                float __g2 = *(input + (bs + 2) * ch_in + ic + 6);
                float __h2 = *(input + (bs + 2) * ch_in + ic + 7);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);
                float __e3 = *(input + (bs + 3) * ch_in + ic + 4);
                float __f3 = *(input + (bs + 3) * ch_in + ic + 5);
                float __g3 = *(input + (bs + 3) * ch_in + ic + 6);
                float __h3 = *(input + (bs + 3) * ch_in + ic + 7);

                float __a4 = *(input + (bs + 4) * ch_in + ic);
                float __b4 = *(input + (bs + 4) * ch_in + ic + 1);
                float __c4 = *(input + (bs + 4) * ch_in + ic + 2);
                float __d4 = *(input + (bs + 4) * ch_in + ic + 3);
                float __e4 = *(input + (bs + 4) * ch_in + ic + 4);
                float __f4 = *(input + (bs + 4) * ch_in + ic + 5);
                float __g4 = *(input + (bs + 4) * ch_in + ic + 6);
                float __h4 = *(input + (bs + 4) * ch_in + ic + 7);

                float __a5 = *(input + (bs + 5) * ch_in + ic);
                float __b5 = *(input + (bs + 5) * ch_in + ic + 1);
                float __c5 = *(input + (bs + 5) * ch_in + ic + 2);
                float __d5 = *(input + (bs + 5) * ch_in + ic + 3);
                float __e5 = *(input + (bs + 5) * ch_in + ic + 4);
                float __f5 = *(input + (bs + 5) * ch_in + ic + 5);
                float __g5 = *(input + (bs + 5) * ch_in + ic + 6);
                float __h5 = *(input + (bs + 5) * ch_in + ic + 7);

                float __a6 = *(input + (bs + 6) * ch_in + ic);
                float __b6 = *(input + (bs + 6) * ch_in + ic + 1);
                float __c6 = *(input + (bs + 6) * ch_in + ic + 2);
                float __d6 = *(input + (bs + 6) * ch_in + ic + 3);
                float __e6 = *(input + (bs + 6) * ch_in + ic + 4);
                float __f6 = *(input + (bs + 6) * ch_in + ic + 5);
                float __g6 = *(input + (bs + 6) * ch_in + ic + 6);
                float __h6 = *(input + (bs + 6) * ch_in + ic + 7);

                float __a7 = *(input + (bs + 7) * ch_in + ic);
                float __b7 = *(input + (bs + 7) * ch_in + ic + 1);
                float __c7 = *(input + (bs + 7) * ch_in + ic + 2);
                float __d7 = *(input + (bs + 7) * ch_in + ic + 3);
                float __e7 = *(input + (bs + 7) * ch_in + ic + 4);
                float __f7 = *(input + (bs + 7) * ch_in + ic + 5);
                float __g7 = *(input + (bs + 7) * ch_in + ic + 6);
                float __h7 = *(input + (bs + 7) * ch_in + ic + 7);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                _y0 += __e0 * _w4;
                _y0 += __f0 * _w5;
                _y0 += __g0 * _w6;
                _y0 += __h0 * _w7;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                _y1 += __b1 * _w1;
                _y1 += __c1 * _w2;
                _y1 += __d1 * _w3;
                _y1 += __e1 * _w4;
                _y1 += __f1 * _w5;
                _y1 += __g1 * _w6;
                _y1 += __h1 * _w7;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                _y2 += __b2 * _w1;
                _y2 += __c2 * _w2;
                _y2 += __d2 * _w3;
                _y2 += __e2 * _w4;
                _y2 += __f2 * _w5;
                _y2 += __g2 * _w6;
                _y2 += __h2 * _w7;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                _y3 += __b3 * _w1;
                _y3 += __c3 * _w2;
                _y3 += __d3 * _w3;
                _y3 += __e3 * _w4;
                _y3 += __f3 * _w5;
                _y3 += __g3 * _w6;
                _y3 += __h3 * _w7;
                *y_ptr3 = _y3;

                float _y4 = *(y_ptr4);
                _y4 += __a4 * _w0;
                _y4 += __b4 * _w1;
                _y4 += __c4 * _w2;
                _y4 += __d4 * _w3;
                _y4 += __e4 * _w4;
                _y4 += __f4 * _w5;
                _y4 += __g4 * _w6;
                _y4 += __h4 * _w7;
                *y_ptr4 = _y4;

                float _y5 = *(y_ptr5);
                _y5 += __a5 * _w0;
                _y5 += __b5 * _w1;
                _y5 += __c5 * _w2;
                _y5 += __d5 * _w3;
                _y5 += __e5 * _w4;
                _y5 += __f5 * _w5;
                _y5 += __g5 * _w6;
                _y5 += __h5 * _w7;
                *y_ptr5 = _y5;

                float _y6 = *(y_ptr6);
                _y6 += __a6 * _w0;
                _y6 += __b6 * _w1;
                _y6 += __c6 * _w2;
                _y6 += __d6 * _w3;
                _y6 += __e6 * _w4;
                _y6 += __f6 * _w5;
                _y6 += __g6 * _w6;
                _y6 += __h6 * _w7;
                *y_ptr6 = _y6;

                float _y7 = *(y_ptr7);
                _y7 += __a7 * _w0;
                _y7 += __b7 * _w1;
                _y7 += __c7 * _w2;
                _y7 += __d7 * _w3;
                _y7 += __e7 * _w4;
                _y7 += __f7 * _w5;
                _y7 += __g7 * _w6;
                _y7 += __h7 * _w7;
                *y_ptr7 = _y7;
            }
        }
        for (; ic + 3 < ch_in; ic+=4) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);

            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _b4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
            __m256 _c4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
            __m256 _d4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);

            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _b5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
            __m256 _c5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
            __m256 _d5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);

            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _b6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
            __m256 _c6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
            __m256 _d6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);

            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);
            __m256 _b7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
            __m256 _c7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
            __m256 _d7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _y4 = _mm256_fmadd_ps(_b4, _w1, _y4);
                _y4 = _mm256_fmadd_ps(_c4, _w2, _y4);
                _y4 = _mm256_fmadd_ps(_d4, _w3, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _y5 = _mm256_fmadd_ps(_b5, _w1, _y5);
                _y5 = _mm256_fmadd_ps(_c5, _w2, _y5);
                _y5 = _mm256_fmadd_ps(_d5, _w3, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _y6 = _mm256_fmadd_ps(_b6, _w1, _y6);
                _y6 = _mm256_fmadd_ps(_c6, _w2, _y6);
                _y6 = _mm256_fmadd_ps(_d6, _w3, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _y7 = _mm256_fmadd_ps(_b7, _w1, _y7);
                _y7 = _mm256_fmadd_ps(_c7, _w2, _y7);
                _y7 = _mm256_fmadd_ps(_d7, _w3, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);

                __m128 __a4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic);
                __m128 __b4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 1);
                __m128 __c4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 2);
                __m128 __d4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic + 3);

                __m128 __a5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic);
                __m128 __b5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 1);
                __m128 __c5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 2);
                __m128 __d5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic + 3);

                __m128 __a6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic);
                __m128 __b6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 1);
                __m128 __c6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 2);
                __m128 __d6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic + 3);

                __m128 __a7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic);
                __m128 __b7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 1);
                __m128 __c7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 2);
                __m128 __d7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic + 3);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _mm_store_ps(y_ptr3, _y3);

                __m128 _y4 = _mm_load_ps(y_ptr4);
                _y4 = _mm_fmadd_ps(__a4, _w0, _y4);
                _y4 = _mm_fmadd_ps(__b4, _w1, _y4);
                _y4 = _mm_fmadd_ps(__c4, _w2, _y4);
                _y4 = _mm_fmadd_ps(__d4, _w3, _y4);
                _mm_store_ps(y_ptr4, _y4);

                __m128 _y5 = _mm_load_ps(y_ptr5);
                _y5 = _mm_fmadd_ps(__a5, _w0, _y5);
                _y5 = _mm_fmadd_ps(__b5, _w1, _y5);
                _y5 = _mm_fmadd_ps(__c5, _w2, _y5);
                _y5 = _mm_fmadd_ps(__d5, _w3, _y5);
                _mm_store_ps(y_ptr5, _y5);

                __m128 _y6 = _mm_load_ps(y_ptr6);
                _y6 = _mm_fmadd_ps(__a6, _w0, _y6);
                _y6 = _mm_fmadd_ps(__b6, _w1, _y6);
                _y6 = _mm_fmadd_ps(__c6, _w2, _y6);
                _y6 = _mm_fmadd_ps(__d6, _w3, _y6);
                _mm_store_ps(y_ptr6, _y6);

                __m128 _y7 = _mm_load_ps(y_ptr7);
                _y7 = _mm_fmadd_ps(__a7, _w0, _y7);
                _y7 = _mm_fmadd_ps(__b7, _w1, _y7);
                _y7 = _mm_fmadd_ps(__c7, _w2, _y7);
                _y7 = _mm_fmadd_ps(__d7, _w3, _y7);
                _mm_store_ps(y_ptr7, _y7);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);

                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);

                float __a4 = *(input + (bs + 4) * ch_in + ic);
                float __b4 = *(input + (bs + 4) * ch_in + ic + 1);
                float __c4 = *(input + (bs + 4) * ch_in + ic + 2);
                float __d4 = *(input + (bs + 4) * ch_in + ic + 3);

                float __a5 = *(input + (bs + 5) * ch_in + ic);
                float __b5 = *(input + (bs + 5) * ch_in + ic + 1);
                float __c5 = *(input + (bs + 5) * ch_in + ic + 2);
                float __d5 = *(input + (bs + 5) * ch_in + ic + 3);

                float __a6 = *(input + (bs + 6) * ch_in + ic);
                float __b6 = *(input + (bs + 6) * ch_in + ic + 1);
                float __c6 = *(input + (bs + 6) * ch_in + ic + 2);
                float __d6 = *(input + (bs + 6) * ch_in + ic + 3);

                float __a7 = *(input + (bs + 7) * ch_in + ic);
                float __b7 = *(input + (bs + 7) * ch_in + ic + 1);
                float __c7 = *(input + (bs + 7) * ch_in + ic + 2);
                float __d7 = *(input + (bs + 7) * ch_in + ic + 3);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                _y1 += __b1 * _w1;
                _y1 += __c1 * _w2;
                _y1 += __d1 * _w3;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                _y2 += __b2 * _w1;
                _y2 += __c2 * _w2;
                _y2 += __d2 * _w3;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                _y3 += __b3 * _w1;
                _y3 += __c3 * _w2;
                _y3 += __d3 * _w3;
                *y_ptr3 = _y3;

                float _y4 = *(y_ptr4);
                _y4 += __a4 * _w0;
                _y4 += __b4 * _w1;
                _y4 += __c4 * _w2;
                _y4 += __d4 * _w3;
                *y_ptr4 = _y4;

                float _y5 = *(y_ptr5);
                _y5 += __a5 * _w0;
                _y5 += __b5 * _w1;
                _y5 += __c5 * _w2;
                _y5 += __d5 * _w3;
                *y_ptr5 = _y5;

                float _y6 = *(y_ptr6);
                _y6 += __a6 * _w0;
                _y6 += __b6 * _w1;
                _y6 += __c6 * _w2;
                _y6 += __d6 * _w3;
                *y_ptr6 = _y6;

                float _y7 = *(y_ptr7);
                _y7 += __a7 * _w0;
                _y7 += __b7 * _w1;
                _y7 += __c7 * _w2;
                _y7 += __d7 * _w3;
                *y_ptr7 = _y7;
            }
        }
        for (; ic < ch_in; ic++) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _a4 = _mm256_broadcast_ss(input + (bs + 4) * ch_in + ic);
            __m256 _a5 = _mm256_broadcast_ss(input + (bs + 5) * ch_in + ic);
            __m256 _a6 = _mm256_broadcast_ss(input + (bs + 6) * ch_in + ic);
            __m256 _a7 = _mm256_broadcast_ss(input + (bs + 7) * ch_in + ic);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);

                __m256 _y4 = _mm256_loadu_ps(y_ptr4);
                _y4 = _mm256_fmadd_ps(_a4, _w0, _y4);
                _mm256_storeu_ps(y_ptr4, _y4);

                __m256 _y5 = _mm256_loadu_ps(y_ptr5);
                _y5 = _mm256_fmadd_ps(_a5, _w0, _y5);
                _mm256_storeu_ps(y_ptr5, _y5);

                __m256 _y6 = _mm256_loadu_ps(y_ptr6);
                _y6 = _mm256_fmadd_ps(_a6, _w0, _y6);
                _mm256_storeu_ps(y_ptr6, _y6);

                __m256 _y7 = _mm256_loadu_ps(y_ptr7);
                _y7 = _mm256_fmadd_ps(_a7, _w0, _y7);
                _mm256_storeu_ps(y_ptr7, _y7);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);

                __m128 __a4 = _mm_broadcast_ss(input + (bs + 4) * ch_in + ic);

                __m128 __a5 = _mm_broadcast_ss(input + (bs + 5) * ch_in + ic);

                __m128 __a6 = _mm_broadcast_ss(input + (bs + 6) * ch_in + ic);

                __m128 __a7 = _mm_broadcast_ss(input + (bs + 7) * ch_in + ic);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _mm_store_ps(y_ptr3, _y3);

                __m128 _y4 = _mm_load_ps(y_ptr4);
                _y4 = _mm_fmadd_ps(__a4, _w0, _y4);
                _mm_store_ps(y_ptr4, _y4);

                __m128 _y5 = _mm_load_ps(y_ptr5);
                _y5 = _mm_fmadd_ps(__a5, _w0, _y5);
                _mm_store_ps(y_ptr5, _y5);

                __m128 _y6 = _mm_load_ps(y_ptr6);
                _y6 = _mm_fmadd_ps(__a6, _w0, _y6);
                _mm_store_ps(y_ptr6, _y6);

                __m128 _y7 = _mm_load_ps(y_ptr7);
                _y7 = _mm_fmadd_ps(__a7, _w0, _y7);
                _mm_store_ps(y_ptr7, _y7);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;
                float* y_ptr4 = output + (bs + 4) * ch_out + oc;
                float* y_ptr5 = output + (bs + 5) * ch_out + oc;
                float* y_ptr6 = output + (bs + 6) * ch_out + oc;
                float* y_ptr7 = output + (bs + 7) * ch_out + oc;

                float _w0 = *(w_ptr0);

                float __a0 = *(input + bs * ch_in + ic);

                float __a1 = *(input + (bs + 1) * ch_in + ic);

                float __a2 = *(input + (bs + 2) * ch_in + ic);

                float __a3 = *(input + (bs + 3) * ch_in + ic);

                float __a4 = *(input + (bs + 4) * ch_in + ic);

                float __a5 = *(input + (bs + 5) * ch_in + ic);

                float __a6 = *(input + (bs + 6) * ch_in + ic);

                float __a7 = *(input + (bs + 7) * ch_in + ic);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                *y_ptr3 = _y3;

                float _y4 = *(y_ptr4);
                _y4 += __a4 * _w0;
                *y_ptr4 = _y4;

                float _y5 = *(y_ptr5);
                _y5 += __a5 * _w0;
                *y_ptr5 = _y5;

                float _y6 = *(y_ptr6);
                _y6 += __a6 * _w0;
                *y_ptr6 = _y6;

                float _y7 = *(y_ptr7);
                _y7 += __a7 * _w0;
                *y_ptr7 = _y7;
            }
        }
    }

    // 下面的情况主要是复制上面pack8的代码后删掉部分代码。
    // 比如，pack4时bs取不到 bs + 4、bs + 5、bs + 6、bs + 7 ，把涉及 bs + 4、bs + 5、bs + 6、bs + 7 的变量、指针全部删去即可。
    // 只有输入张量、输出张量有 batch_size 维度，被删的是这2个张量相关的代码。
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = pack4_offset; bs < pack1_offset; bs+=4) {
        int ic = 0;
        for (; ic + 7 < ch_in; ic+=8) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);
            __m256 _e0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 4);
            __m256 _f0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 5);
            __m256 _g0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 6);
            __m256 _h0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 7);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
            __m256 _e1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
            __m256 _f1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
            __m256 _g1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
            __m256 _h1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
            __m256 _e2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
            __m256 _f2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
            __m256 _g2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
            __m256 _h2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
            __m256 _e3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
            __m256 _f3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
            __m256 _g3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
            __m256 _h3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);
                __m256 _w4 = _mm256_loadu_ps(w_ptr4);
                __m256 _w5 = _mm256_loadu_ps(w_ptr5);
                __m256 _w6 = _mm256_loadu_ps(w_ptr6);
                __m256 _w7 = _mm256_loadu_ps(w_ptr7);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _y0 = _mm256_fmadd_ps(_e0, _w4, _y0);
                _y0 = _mm256_fmadd_ps(_f0, _w5, _y0);
                _y0 = _mm256_fmadd_ps(_g0, _w6, _y0);
                _y0 = _mm256_fmadd_ps(_h0, _w7, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _y1 = _mm256_fmadd_ps(_e1, _w4, _y1);
                _y1 = _mm256_fmadd_ps(_f1, _w5, _y1);
                _y1 = _mm256_fmadd_ps(_g1, _w6, _y1);
                _y1 = _mm256_fmadd_ps(_h1, _w7, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _y2 = _mm256_fmadd_ps(_e2, _w4, _y2);
                _y2 = _mm256_fmadd_ps(_f2, _w5, _y2);
                _y2 = _mm256_fmadd_ps(_g2, _w6, _y2);
                _y2 = _mm256_fmadd_ps(_h2, _w7, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _y3 = _mm256_fmadd_ps(_e3, _w4, _y3);
                _y3 = _mm256_fmadd_ps(_f3, _w5, _y3);
                _y3 = _mm256_fmadd_ps(_g3, _w6, _y3);
                _y3 = _mm256_fmadd_ps(_h3, _w7, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);
                __m128 _w4 = _mm_load_ps(w_ptr4);
                __m128 _w5 = _mm_load_ps(w_ptr5);
                __m128 _w6 = _mm_load_ps(w_ptr6);
                __m128 _w7 = _mm_load_ps(w_ptr7);


                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);
                __m128 __e0 = _mm_broadcast_ss(input + bs * ch_in + ic + 4);
                __m128 __f0 = _mm_broadcast_ss(input + bs * ch_in + ic + 5);
                __m128 __g0 = _mm_broadcast_ss(input + bs * ch_in + ic + 6);
                __m128 __h0 = _mm_broadcast_ss(input + bs * ch_in + ic + 7);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);
                __m128 __e1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 4);
                __m128 __f1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 5);
                __m128 __g1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 6);
                __m128 __h1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 7);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);
                __m128 __e2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 4);
                __m128 __f2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 5);
                __m128 __g2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 6);
                __m128 __h2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 7);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);
                __m128 __e3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 4);
                __m128 __f3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 5);
                __m128 __g3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 6);
                __m128 __h3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 7);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _y0 = _mm_fmadd_ps(__e0, _w4, _y0);
                _y0 = _mm_fmadd_ps(__f0, _w5, _y0);
                _y0 = _mm_fmadd_ps(__g0, _w6, _y0);
                _y0 = _mm_fmadd_ps(__h0, _w7, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _y1 = _mm_fmadd_ps(__e1, _w4, _y1);
                _y1 = _mm_fmadd_ps(__f1, _w5, _y1);
                _y1 = _mm_fmadd_ps(__g1, _w6, _y1);
                _y1 = _mm_fmadd_ps(__h1, _w7, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _y2 = _mm_fmadd_ps(__e2, _w4, _y2);
                _y2 = _mm_fmadd_ps(__f2, _w5, _y2);
                _y2 = _mm_fmadd_ps(__g2, _w6, _y2);
                _y2 = _mm_fmadd_ps(__h2, _w7, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _y3 = _mm_fmadd_ps(__e3, _w4, _y3);
                _y3 = _mm_fmadd_ps(__f3, _w5, _y3);
                _y3 = _mm_fmadd_ps(__g3, _w6, _y3);
                _y3 = _mm_fmadd_ps(__h3, _w7, _y3);
                _mm_store_ps(y_ptr3, _y3);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);
                float _w4 = *(w_ptr4);
                float _w5 = *(w_ptr5);
                float _w6 = *(w_ptr6);
                float _w7 = *(w_ptr7);


                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);
                float __e0 = *(input + bs * ch_in + ic + 4);
                float __f0 = *(input + bs * ch_in + ic + 5);
                float __g0 = *(input + bs * ch_in + ic + 6);
                float __h0 = *(input + bs * ch_in + ic + 7);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);
                float __e1 = *(input + (bs + 1) * ch_in + ic + 4);
                float __f1 = *(input + (bs + 1) * ch_in + ic + 5);
                float __g1 = *(input + (bs + 1) * ch_in + ic + 6);
                float __h1 = *(input + (bs + 1) * ch_in + ic + 7);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);
                float __e2 = *(input + (bs + 2) * ch_in + ic + 4);
                float __f2 = *(input + (bs + 2) * ch_in + ic + 5);
                float __g2 = *(input + (bs + 2) * ch_in + ic + 6);
                float __h2 = *(input + (bs + 2) * ch_in + ic + 7);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);
                float __e3 = *(input + (bs + 3) * ch_in + ic + 4);
                float __f3 = *(input + (bs + 3) * ch_in + ic + 5);
                float __g3 = *(input + (bs + 3) * ch_in + ic + 6);
                float __h3 = *(input + (bs + 3) * ch_in + ic + 7);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                _y0 += __e0 * _w4;
                _y0 += __f0 * _w5;
                _y0 += __g0 * _w6;
                _y0 += __h0 * _w7;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                _y1 += __b1 * _w1;
                _y1 += __c1 * _w2;
                _y1 += __d1 * _w3;
                _y1 += __e1 * _w4;
                _y1 += __f1 * _w5;
                _y1 += __g1 * _w6;
                _y1 += __h1 * _w7;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                _y2 += __b2 * _w1;
                _y2 += __c2 * _w2;
                _y2 += __d2 * _w3;
                _y2 += __e2 * _w4;
                _y2 += __f2 * _w5;
                _y2 += __g2 * _w6;
                _y2 += __h2 * _w7;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                _y3 += __b3 * _w1;
                _y3 += __c3 * _w2;
                _y3 += __d3 * _w3;
                _y3 += __e3 * _w4;
                _y3 += __f3 * _w5;
                _y3 += __g3 * _w6;
                _y3 += __h3 * _w7;
                *y_ptr3 = _y3;
            }
        }
        for (; ic + 3 < ch_in; ic+=4) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);

            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _b1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
            __m256 _c1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
            __m256 _d1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _b2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
            __m256 _c2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
            __m256 _d2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);
            __m256 _b3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
            __m256 _c3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
            __m256 _d3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _y1 = _mm256_fmadd_ps(_b1, _w1, _y1);
                _y1 = _mm256_fmadd_ps(_c1, _w2, _y1);
                _y1 = _mm256_fmadd_ps(_d1, _w3, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _y2 = _mm256_fmadd_ps(_b2, _w1, _y2);
                _y2 = _mm256_fmadd_ps(_c2, _w2, _y2);
                _y2 = _mm256_fmadd_ps(_d2, _w3, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _y3 = _mm256_fmadd_ps(_b3, _w1, _y3);
                _y3 = _mm256_fmadd_ps(_c3, _w2, _y3);
                _y3 = _mm256_fmadd_ps(_d3, _w3, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);
                __m128 __b1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 1);
                __m128 __c1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 2);
                __m128 __d1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic + 3);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);
                __m128 __b2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 1);
                __m128 __c2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 2);
                __m128 __d2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic + 3);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);
                __m128 __b3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 1);
                __m128 __c3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 2);
                __m128 __d3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic + 3);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _y1 = _mm_fmadd_ps(__b1, _w1, _y1);
                _y1 = _mm_fmadd_ps(__c1, _w2, _y1);
                _y1 = _mm_fmadd_ps(__d1, _w3, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _y2 = _mm_fmadd_ps(__b2, _w1, _y2);
                _y2 = _mm_fmadd_ps(__c2, _w2, _y2);
                _y2 = _mm_fmadd_ps(__d2, _w3, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _y3 = _mm_fmadd_ps(__b3, _w1, _y3);
                _y3 = _mm_fmadd_ps(__c3, _w2, _y3);
                _y3 = _mm_fmadd_ps(__d3, _w3, _y3);
                _mm_store_ps(y_ptr3, _y3);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);

                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);

                float __a1 = *(input + (bs + 1) * ch_in + ic);
                float __b1 = *(input + (bs + 1) * ch_in + ic + 1);
                float __c1 = *(input + (bs + 1) * ch_in + ic + 2);
                float __d1 = *(input + (bs + 1) * ch_in + ic + 3);

                float __a2 = *(input + (bs + 2) * ch_in + ic);
                float __b2 = *(input + (bs + 2) * ch_in + ic + 1);
                float __c2 = *(input + (bs + 2) * ch_in + ic + 2);
                float __d2 = *(input + (bs + 2) * ch_in + ic + 3);

                float __a3 = *(input + (bs + 3) * ch_in + ic);
                float __b3 = *(input + (bs + 3) * ch_in + ic + 1);
                float __c3 = *(input + (bs + 3) * ch_in + ic + 2);
                float __d3 = *(input + (bs + 3) * ch_in + ic + 3);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                _y1 += __b1 * _w1;
                _y1 += __c1 * _w2;
                _y1 += __d1 * _w3;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                _y2 += __b2 * _w1;
                _y2 += __c2 * _w2;
                _y2 += __d2 * _w3;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                _y3 += __b3 * _w1;
                _y3 += __c3 * _w2;
                _y3 += __d3 * _w3;
                *y_ptr3 = _y3;
            }
        }
        for (; ic < ch_in; ic++) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _a1 = _mm256_broadcast_ss(input + (bs + 1) * ch_in + ic);
            __m256 _a2 = _mm256_broadcast_ss(input + (bs + 2) * ch_in + ic);
            __m256 _a3 = _mm256_broadcast_ss(input + (bs + 3) * ch_in + ic);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);

                __m256 _y1 = _mm256_loadu_ps(y_ptr1);
                _y1 = _mm256_fmadd_ps(_a1, _w0, _y1);
                _mm256_storeu_ps(y_ptr1, _y1);

                __m256 _y2 = _mm256_loadu_ps(y_ptr2);
                _y2 = _mm256_fmadd_ps(_a2, _w0, _y2);
                _mm256_storeu_ps(y_ptr2, _y2);

                __m256 _y3 = _mm256_loadu_ps(y_ptr3);
                _y3 = _mm256_fmadd_ps(_a3, _w0, _y3);
                _mm256_storeu_ps(y_ptr3, _y3);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);

                __m128 __a1 = _mm_broadcast_ss(input + (bs + 1) * ch_in + ic);

                __m128 __a2 = _mm_broadcast_ss(input + (bs + 2) * ch_in + ic);

                __m128 __a3 = _mm_broadcast_ss(input + (bs + 3) * ch_in + ic);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _mm_store_ps(y_ptr0, _y0);

                __m128 _y1 = _mm_load_ps(y_ptr1);
                _y1 = _mm_fmadd_ps(__a1, _w0, _y1);
                _mm_store_ps(y_ptr1, _y1);

                __m128 _y2 = _mm_load_ps(y_ptr2);
                _y2 = _mm_fmadd_ps(__a2, _w0, _y2);
                _mm_store_ps(y_ptr2, _y2);

                __m128 _y3 = _mm_load_ps(y_ptr3);
                _y3 = _mm_fmadd_ps(__a3, _w0, _y3);
                _mm_store_ps(y_ptr3, _y3);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;
                float* y_ptr1 = output + (bs + 1) * ch_out + oc;
                float* y_ptr2 = output + (bs + 2) * ch_out + oc;
                float* y_ptr3 = output + (bs + 3) * ch_out + oc;

                float _w0 = *(w_ptr0);

                float __a0 = *(input + bs * ch_in + ic);

                float __a1 = *(input + (bs + 1) * ch_in + ic);

                float __a2 = *(input + (bs + 2) * ch_in + ic);

                float __a3 = *(input + (bs + 3) * ch_in + ic);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                *y_ptr0 = _y0;

                float _y1 = *(y_ptr1);
                _y1 += __a1 * _w0;
                *y_ptr1 = _y1;

                float _y2 = *(y_ptr2);
                _y2 += __a2 * _w0;
                *y_ptr2 = _y2;

                float _y3 = *(y_ptr3);
                _y3 += __a3 * _w0;
                *y_ptr3 = _y3;
            }
        }
    }

    // 下面的情况主要是复制上面pack4的代码后删掉部分代码。
    // 比如，pack1时bs取不到 bs + 1、bs + 2、bs + 3 ，把涉及 bs + 1、bs + 2、bs + 3 的变量、指针全部删去即可。
    // 只有输入张量、输出张量有 batch_size 维度，被删的是这2个张量相关的代码。
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = pack1_offset; bs < batch_size; bs++) {
        int ic = 0;
        for (; ic + 7 < ch_in; ic+=8) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);
            __m256 _e0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 4);
            __m256 _f0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 5);
            __m256 _g0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 6);
            __m256 _h0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 7);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);
                __m256 _w4 = _mm256_loadu_ps(w_ptr4);
                __m256 _w5 = _mm256_loadu_ps(w_ptr5);
                __m256 _w6 = _mm256_loadu_ps(w_ptr6);
                __m256 _w7 = _mm256_loadu_ps(w_ptr7);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _y0 = _mm256_fmadd_ps(_e0, _w4, _y0);
                _y0 = _mm256_fmadd_ps(_f0, _w5, _y0);
                _y0 = _mm256_fmadd_ps(_g0, _w6, _y0);
                _y0 = _mm256_fmadd_ps(_h0, _w7, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);
                __m128 _w4 = _mm_load_ps(w_ptr4);
                __m128 _w5 = _mm_load_ps(w_ptr5);
                __m128 _w6 = _mm_load_ps(w_ptr6);
                __m128 _w7 = _mm_load_ps(w_ptr7);


                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);
                __m128 __e0 = _mm_broadcast_ss(input + bs * ch_in + ic + 4);
                __m128 __f0 = _mm_broadcast_ss(input + bs * ch_in + ic + 5);
                __m128 __g0 = _mm_broadcast_ss(input + bs * ch_in + ic + 6);
                __m128 __h0 = _mm_broadcast_ss(input + bs * ch_in + ic + 7);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _y0 = _mm_fmadd_ps(__e0, _w4, _y0);
                _y0 = _mm_fmadd_ps(__f0, _w5, _y0);
                _y0 = _mm_fmadd_ps(__g0, _w6, _y0);
                _y0 = _mm_fmadd_ps(__h0, _w7, _y0);
                _mm_store_ps(y_ptr0, _y0);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                const float* w_ptr4 = weight + (ic + 4) * ch_out + oc;
                const float* w_ptr5 = weight + (ic + 5) * ch_out + oc;
                const float* w_ptr6 = weight + (ic + 6) * ch_out + oc;
                const float* w_ptr7 = weight + (ic + 7) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);
                float _w4 = *(w_ptr4);
                float _w5 = *(w_ptr5);
                float _w6 = *(w_ptr6);
                float _w7 = *(w_ptr7);


                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);
                float __e0 = *(input + bs * ch_in + ic + 4);
                float __f0 = *(input + bs * ch_in + ic + 5);
                float __g0 = *(input + bs * ch_in + ic + 6);
                float __h0 = *(input + bs * ch_in + ic + 7);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                _y0 += __e0 * _w4;
                _y0 += __f0 * _w5;
                _y0 += __g0 * _w6;
                _y0 += __h0 * _w7;
                *y_ptr0 = _y0;
            }
        }
        for (; ic + 3 < ch_in; ic+=4) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);
            __m256 _b0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 1);
            __m256 _c0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 2);
            __m256 _d0 = _mm256_broadcast_ss(input + bs * ch_in + ic + 3);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);
                __m256 _w1 = _mm256_loadu_ps(w_ptr1);
                __m256 _w2 = _mm256_loadu_ps(w_ptr2);
                __m256 _w3 = _mm256_loadu_ps(w_ptr3);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _y0 = _mm256_fmadd_ps(_b0, _w1, _y0);
                _y0 = _mm256_fmadd_ps(_c0, _w2, _y0);
                _y0 = _mm256_fmadd_ps(_d0, _w3, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);
                __m128 _w1 = _mm_load_ps(w_ptr1);
                __m128 _w2 = _mm_load_ps(w_ptr2);
                __m128 _w3 = _mm_load_ps(w_ptr3);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);
                __m128 __b0 = _mm_broadcast_ss(input + bs * ch_in + ic + 1);
                __m128 __c0 = _mm_broadcast_ss(input + bs * ch_in + ic + 2);
                __m128 __d0 = _mm_broadcast_ss(input + bs * ch_in + ic + 3);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _y0 = _mm_fmadd_ps(__b0, _w1, _y0);
                _y0 = _mm_fmadd_ps(__c0, _w2, _y0);
                _y0 = _mm_fmadd_ps(__d0, _w3, _y0);
                _mm_store_ps(y_ptr0, _y0);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                const float* w_ptr1 = weight + (ic + 1) * ch_out + oc;
                const float* w_ptr2 = weight + (ic + 2) * ch_out + oc;
                const float* w_ptr3 = weight + (ic + 3) * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                float _w0 = *(w_ptr0);
                float _w1 = *(w_ptr1);
                float _w2 = *(w_ptr2);
                float _w3 = *(w_ptr3);

                float __a0 = *(input + bs * ch_in + ic);
                float __b0 = *(input + bs * ch_in + ic + 1);
                float __c0 = *(input + bs * ch_in + ic + 2);
                float __d0 = *(input + bs * ch_in + ic + 3);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                _y0 += __b0 * _w1;
                _y0 += __c0 * _w2;
                _y0 += __d0 * _w3;
                *y_ptr0 = _y0;
            }
        }
        for (; ic < ch_in; ic++) {
            __m256 _a0 = _mm256_broadcast_ss(input + bs * ch_in + ic);

            int oc = 0;
            for (; oc + 7 < ch_out; oc+=8) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m256 _w0 = _mm256_loadu_ps(w_ptr0);


                __m256 _y0 = _mm256_loadu_ps(y_ptr0);
                _y0 = _mm256_fmadd_ps(_a0, _w0, _y0);
                _mm256_storeu_ps(y_ptr0, _y0);
            }
            for (; oc + 3 < ch_out; oc+=4) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                __m128 _w0 = _mm_load_ps(w_ptr0);

                __m128 __a0 = _mm_broadcast_ss(input + bs * ch_in + ic);


                __m128 _y0 = _mm_load_ps(y_ptr0);
                _y0 = _mm_fmadd_ps(__a0, _w0, _y0);
                _mm_store_ps(y_ptr0, _y0);
            }
            for (; oc < ch_out; oc++) {
                const float* w_ptr0 = weight + ic * ch_out + oc;
                float* y_ptr0 = output + bs * ch_out + oc;

                float _w0 = *(w_ptr0);

                float __a0 = *(input + bs * ch_in + ic);


                float _y0 = *(y_ptr0);
                _y0 += __a0 * _w0;
                *y_ptr0 = _y0;
            }
        }
    }

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


void matmul_true_cpp_kernel(const int num_threads_, const float* input, const float* weight, float* output, int batch_size, int ch_in, int ch_out) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < batch_size * ch_out; i++)
    {
        *(output + i) = 0.f;
    }


    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int oc = 0; oc < ch_out; oc++) {
            for (int ic = 0; ic < ch_in; ic++) {
                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[ic * ch_out + oc];
            }
        }
    }
}

void winograd23_weight_transform(const int num_threads_, const float* weight, float* U, int tile_size, int in_C, int out_C) {
    float G[4*3] = {1.f, 0.f, 0.f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.f, 0.f, 1.f};
    float* GW = new float[4 * 3 * in_C * out_C];
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < 4 * 3 * in_C * out_C; i++) {
        GW[i] = 0.f;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < tile_size * tile_size * in_C * out_C; i++) {
        U[i] = 0.f;
    }

//    #pragma omp parallel for num_threads(num_threads_)
    for (int ic = 0; ic < in_C; ic++) {
        for (int oc = 0; oc < out_C; oc++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        GW[((i * 3 + j) * in_C + ic) * out_C + oc] += G[i * 3 + k] * weight[((k * 3 + j) * in_C + ic) * out_C + oc];
                    }
                }
            }
        }
    }

//    #pragma omp parallel for num_threads(num_threads_)
    for (int ic = 0; ic < in_C; ic++) {
        for (int oc = 0; oc < out_C; oc++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    for (int k = 0; k < 3; k++) {
                        U[((i * 4 + j) * in_C + ic) * out_C + oc] += GW[((i * 3 + k) * in_C + ic) * out_C + oc] * G[j * 3 + k];
                    }
                }
            }
        }
    }
    delete GW;
}


void winograd23_input_cpp_transform(const int num_threads_, const float* input, float* V, int in_C, int tile_num, int out_W, int pad_W, int otile_size) {

    #pragma omp parallel for num_threads(num_threads_)
    for (int tid = 0; tid < tile_num; tid++) {
        int th = (tid / (out_W / otile_size) ) * 2;
        int tw = (tid % (out_W / otile_size) ) * 2;
        for (int ic = 0; ic < in_C; ic++) {
            V[((0 * 4 + 0) * tile_num + tid) * in_C + ic] = input[((th+0) * pad_W + tw+0) * in_C + ic] - input[((th+2) * pad_W + tw+0) * in_C + ic] - input[((th+0) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((0 * 4 + 1) * tile_num + tid) * in_C + ic] = input[((th+0) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+0) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((0 * 4 + 2) * tile_num + tid) * in_C + ic] = -input[((th+0) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+0) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((0 * 4 + 3) * tile_num + tid) * in_C + ic] = input[((th+0) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+0) * pad_W + tw+3) * in_C + ic] + input[((th+2) * pad_W + tw+3) * in_C + ic];
            V[((1 * 4 + 0) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+0) * in_C + ic] + input[((th+2) * pad_W + tw+0) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((1 * 4 + 1) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((1 * 4 + 2) * tile_num + tid) * in_C + ic] = -input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((1 * 4 + 3) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+3) * in_C + ic] - input[((th+2) * pad_W + tw+3) * in_C + ic];
            V[((2 * 4 + 0) * tile_num + tid) * in_C + ic] = -input[((th+1) * pad_W + tw+0) * in_C + ic] + input[((th+2) * pad_W + tw+0) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((2 * 4 + 1) * tile_num + tid) * in_C + ic] = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((2 * 4 + 2) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            V[((2 * 4 + 3) * tile_num + tid) * in_C + ic] = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+3) * in_C + ic] - input[((th+2) * pad_W + tw+3) * in_C + ic];
            V[((3 * 4 + 0) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+0) * in_C + ic] - input[((th+3) * pad_W + tw+0) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+3) * pad_W + tw+2) * in_C + ic];
            V[((3 * 4 + 1) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+3) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+3) * pad_W + tw+2) * in_C + ic];
            V[((3 * 4 + 2) * tile_num + tid) * in_C + ic] = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+3) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+3) * pad_W + tw+2) * in_C + ic];
            V[((3 * 4 + 3) * tile_num + tid) * in_C + ic] = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+3) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+3) * in_C + ic] + input[((th+3) * pad_W + tw+3) * in_C + ic];

        }
    }

}



void winograd23_input_x86_transform(const int num_threads_, const float* input, float* V, int in_C, int tile_num, int out_W, int pad_W, int otile_size) {

    const int elempack = 8;

    #pragma omp parallel for num_threads(num_threads_)
    for (int tid = 0; tid < tile_num; tid++) {
        int th = (tid / (out_W / otile_size) ) * 2;
        int tw = (tid % (out_W / otile_size) ) * 2;
        float* V00_ptr = V + ((0 * 4 + 0) * tile_num + tid) * in_C;
        float* V01_ptr = V + ((0 * 4 + 1) * tile_num + tid) * in_C;
        float* V02_ptr = V + ((0 * 4 + 2) * tile_num + tid) * in_C;
        float* V03_ptr = V + ((0 * 4 + 3) * tile_num + tid) * in_C;

        float* V10_ptr = V + ((1 * 4 + 0) * tile_num + tid) * in_C;
        float* V11_ptr = V + ((1 * 4 + 1) * tile_num + tid) * in_C;
        float* V12_ptr = V + ((1 * 4 + 2) * tile_num + tid) * in_C;
        float* V13_ptr = V + ((1 * 4 + 3) * tile_num + tid) * in_C;

        float* V20_ptr = V + ((2 * 4 + 0) * tile_num + tid) * in_C;
        float* V21_ptr = V + ((2 * 4 + 1) * tile_num + tid) * in_C;
        float* V22_ptr = V + ((2 * 4 + 2) * tile_num + tid) * in_C;
        float* V23_ptr = V + ((2 * 4 + 3) * tile_num + tid) * in_C;

        float* V30_ptr = V + ((3 * 4 + 0) * tile_num + tid) * in_C;
        float* V31_ptr = V + ((3 * 4 + 1) * tile_num + tid) * in_C;
        float* V32_ptr = V + ((3 * 4 + 2) * tile_num + tid) * in_C;
        float* V33_ptr = V + ((3 * 4 + 3) * tile_num + tid) * in_C;

        const float* x00_ptr = input + ((th+0) * pad_W + tw+0) * in_C;
        const float* x01_ptr = input + ((th+0) * pad_W + tw+1) * in_C;
        const float* x02_ptr = input + ((th+0) * pad_W + tw+2) * in_C;
        const float* x03_ptr = input + ((th+0) * pad_W + tw+3) * in_C;
        const float* x10_ptr = input + ((th+1) * pad_W + tw+0) * in_C;
        const float* x11_ptr = input + ((th+1) * pad_W + tw+1) * in_C;
        const float* x12_ptr = input + ((th+1) * pad_W + tw+2) * in_C;
        const float* x13_ptr = input + ((th+1) * pad_W + tw+3) * in_C;
        const float* x20_ptr = input + ((th+2) * pad_W + tw+0) * in_C;
        const float* x21_ptr = input + ((th+2) * pad_W + tw+1) * in_C;
        const float* x22_ptr = input + ((th+2) * pad_W + tw+2) * in_C;
        const float* x23_ptr = input + ((th+2) * pad_W + tw+3) * in_C;
        const float* x30_ptr = input + ((th+3) * pad_W + tw+0) * in_C;
        const float* x31_ptr = input + ((th+3) * pad_W + tw+1) * in_C;
        const float* x32_ptr = input + ((th+3) * pad_W + tw+2) * in_C;
        const float* x33_ptr = input + ((th+3) * pad_W + tw+3) * in_C;
        for (int ic = 0; ic + 7 < in_C; ic+=elempack) {
            *(V00_ptr++) = input[((th+0) * pad_W + tw+0) * in_C + ic] - input[((th+2) * pad_W + tw+0) * in_C + ic] - input[((th+0) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V01_ptr++) = input[((th+0) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+0) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V02_ptr++) = -input[((th+0) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+0) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V03_ptr++) = input[((th+0) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+0) * pad_W + tw+3) * in_C + ic] + input[((th+2) * pad_W + tw+3) * in_C + ic];
            *(V10_ptr++) = input[((th+1) * pad_W + tw+0) * in_C + ic] + input[((th+2) * pad_W + tw+0) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V11_ptr++) = input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V12_ptr++) = -input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V13_ptr++) = input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+3) * in_C + ic] - input[((th+2) * pad_W + tw+3) * in_C + ic];
            *(V20_ptr++) = -input[((th+1) * pad_W + tw+0) * in_C + ic] + input[((th+2) * pad_W + tw+0) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V21_ptr++) = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V22_ptr++) = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+2) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+2) * pad_W + tw+2) * in_C + ic];
            *(V23_ptr++) = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+2) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+3) * in_C + ic] - input[((th+2) * pad_W + tw+3) * in_C + ic];
            *(V30_ptr++) = input[((th+1) * pad_W + tw+0) * in_C + ic] - input[((th+3) * pad_W + tw+0) * in_C + ic] - input[((th+1) * pad_W + tw+2) * in_C + ic] + input[((th+3) * pad_W + tw+2) * in_C + ic];
            *(V31_ptr++) = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+3) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+3) * pad_W + tw+2) * in_C + ic];
            *(V32_ptr++) = -input[((th+1) * pad_W + tw+1) * in_C + ic] + input[((th+3) * pad_W + tw+1) * in_C + ic] + input[((th+1) * pad_W + tw+2) * in_C + ic] - input[((th+3) * pad_W + tw+2) * in_C + ic];
            *(V33_ptr++) = input[((th+1) * pad_W + tw+1) * in_C + ic] - input[((th+3) * pad_W + tw+1) * in_C + ic] - input[((th+1) * pad_W + tw+3) * in_C + ic] + input[((th+3) * pad_W + tw+3) * in_C + ic];

            __m256 _x00 = _mm256_loadu_ps(x00_ptr);
            __m256 _x01 = _mm256_loadu_ps(x01_ptr);
            __m256 _x02 = _mm256_loadu_ps(x02_ptr);
            __m256 _x03 = _mm256_loadu_ps(x03_ptr);
            __m256 _x10 = _mm256_loadu_ps(x10_ptr);
            __m256 _x11 = _mm256_loadu_ps(x11_ptr);
            __m256 _x12 = _mm256_loadu_ps(x12_ptr);
            __m256 _x13 = _mm256_loadu_ps(x13_ptr);
            __m256 _x20 = _mm256_loadu_ps(x20_ptr);
            __m256 _x21 = _mm256_loadu_ps(x21_ptr);
            __m256 _x22 = _mm256_loadu_ps(x22_ptr);
            __m256 _x23 = _mm256_loadu_ps(x23_ptr);
            __m256 _x30 = _mm256_loadu_ps(x30_ptr);
            __m256 _x31 = _mm256_loadu_ps(x31_ptr);
            __m256 _x32 = _mm256_loadu_ps(x32_ptr);
            __m256 _x33 = _mm256_loadu_ps(x33_ptr);

            __m256 _v00 = _mm256_sub_ps(_x00, _x20);
            _v00 = _mm256_sub_ps(_v00, _x02);
            _v00 = _mm256_add_ps(_v00, _x22);
            _mm256_storeu_ps(V00_ptr, _v00);

            __m256 _v01 = _mm256_sub_ps(_x01, _x21);
            _v01 = _mm256_add_ps(_v01, _x02);
            _v01 = _mm256_sub_ps(_v01, _x22);
            _mm256_storeu_ps(V01_ptr, _v01);

            __m256 _v02 = _mm256_sub_ps(_x21, _x01);
            _v02 = _mm256_add_ps(_v02, _x02);
            _v02 = _mm256_sub_ps(_v02, _x22);
            _mm256_storeu_ps(V02_ptr, _v02);

            __m256 _v03 = _mm256_sub_ps(_x01, _x21);
            _v03 = _mm256_sub_ps(_v03, _x03);
            _v03 = _mm256_add_ps(_v03, _x23);
            _mm256_storeu_ps(V02_ptr, _v03);

            V00_ptr += elempack;
            V01_ptr += elempack;
            V02_ptr += elempack;
            V03_ptr += elempack;
            V10_ptr += elempack;
            V11_ptr += elempack;
            V12_ptr += elempack;
            V13_ptr += elempack;
            V20_ptr += elempack;
            V21_ptr += elempack;
            V22_ptr += elempack;
            V23_ptr += elempack;
            V30_ptr += elempack;
            V31_ptr += elempack;
            V32_ptr += elempack;
            V33_ptr += elempack;

            x00_ptr += elempack;
            x01_ptr += elempack;
            x02_ptr += elempack;
            x03_ptr += elempack;
            x10_ptr += elempack;
            x11_ptr += elempack;
            x12_ptr += elempack;
            x13_ptr += elempack;
            x20_ptr += elempack;
            x21_ptr += elempack;
            x22_ptr += elempack;
            x23_ptr += elempack;
            x30_ptr += elempack;
            x31_ptr += elempack;
            x32_ptr += elempack;
            x33_ptr += elempack;

        }
    }

}



void winograd23_output_transform(const int num_threads_, const float* Q, float* Y, int out_C, int tile_num, int out_W, int otile_size, int tile_size) {


//    #pragma omp parallel for num_threads(num_threads_)
    for (int tid = 0; tid < tile_num; tid++) {
        for (int oc = 0; oc < out_C; oc++) {
            Y[((0 * otile_size + 0) * tile_num + tid) * out_C + oc] = Q[((0 * tile_size + 0) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 0) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 0) * tile_num + tid) * out_C + oc] + Q[((0 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((0 * tile_size + 2) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 2) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 2) * tile_num + tid) * out_C + oc];
            Y[((0 * otile_size + 1) * tile_num + tid) * out_C + oc] = Q[((0 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((0 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((1 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((0 * tile_size + 3) * tile_num + tid) * out_C + oc] - Q[((1 * tile_size + 3) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 3) * tile_num + tid) * out_C + oc];
            Y[((1 * otile_size + 0) * tile_num + tid) * out_C + oc] = Q[((1 * tile_size + 0) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 0) * tile_num + tid) * out_C + oc] - Q[((3 * tile_size + 0) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((3 * tile_size + 1) * tile_num + tid) * out_C + oc] + Q[((1 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((3 * tile_size + 2) * tile_num + tid) * out_C + oc];
            Y[((1 * otile_size + 1) * tile_num + tid) * out_C + oc] = Q[((1 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((2 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((3 * tile_size + 1) * tile_num + tid) * out_C + oc] - Q[((1 * tile_size + 2) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 2) * tile_num + tid) * out_C + oc] + Q[((3 * tile_size + 2) * tile_num + tid) * out_C + oc] - Q[((1 * tile_size + 3) * tile_num + tid) * out_C + oc] + Q[((2 * tile_size + 3) * tile_num + tid) * out_C + oc] + Q[((3 * tile_size + 3) * tile_num + tid) * out_C + oc];


        }
    }

}



void winograd23_output_relayout(const int num_threads_, const float* Y, float* out, int out_C, int tile_num, int out_W, int pad_W, int otile_size) {

    //    #pragma omp parallel for num_threads(num_threads_)
    for (int tid = 0; tid < tile_num; tid++) {
        int th = (tid / (out_W / otile_size)) * 2;
        int tw = (tid % (out_W / otile_size)) * 2;
        for (int oc = 0; oc < out_C; oc++) {
            out[((th+0) * out_W + (tw+0)) * out_C + oc] = Y[((0 * otile_size + 0) * tile_num + tid) * out_C + oc];
            out[((th+0) * out_W + (tw+1)) * out_C + oc] = Y[((0 * otile_size + 1) * tile_num + tid) * out_C + oc];
            out[((th+1) * out_W + (tw+0)) * out_C + oc] = Y[((1 * otile_size + 0) * tile_num + tid) * out_C + oc];
            out[((th+1) * out_W + (tw+1)) * out_C + oc] = Y[((1 * otile_size + 1) * tile_num + tid) * out_C + oc];

        }
    }

}




void matmul_transB_true_cpp_kernel(const int num_threads_, const float* input, const float* weight, float* output, int batch_size, int ch_in, int ch_out) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int bs = 0; bs < batch_size; bs++) {
        for (int oc = 0; oc < ch_out; oc++) {
            for (int ic = 0; ic < ch_in; ic++) {
                output[bs * ch_out + oc] += input[bs * ch_in + ic] * weight[oc * ch_in + ic];
            }
        }
    }
}


void pad4d_dim23_cpp_kernel(const int num_threads_, const float* input, float* output, int N, int H, int W, int C, int pad_H, int pad_W, int padding_h, int padding_w) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * pad_H * pad_W * C;
        int j = n * H * W * C;
        for (int h = 0; h < padding_h; h++) {
            for (int w = 0; w < pad_W; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = 0.f;
                }
            }
        }
        for (int h = padding_h; h < padding_h + H; h++) {
            for (int w = 0; w < padding_w; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = 0.f;
                }
            }
            for (int w = padding_w; w < padding_w + W; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = input[j++];
                }
            }
            for (int w = padding_w + W; w < pad_W; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = 0.f;
                }
            }
        }
        for (int h = padding_h + H; h < pad_H; h++) {
            for (int w = 0; w < pad_W; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = 0.f;
                }
            }
        }
    }
}


void crop4d_dim23_cpp_kernel(const int num_threads_, const float* input, float* output, int N, int H, int W, int C, int pad_H, int pad_W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * H * W * C;
        for (int h = 0; h < H; h++) {
            int j = (n * pad_H + h) * pad_W * C;
            for (int w = 0; w < W; w++) {
                for (int ic = 0; ic < C; ic++) {
                    output[i++] = input[j++];
                }
            }
        }
    }
}



template<typename data_t>
void wingrad23_NHWC_cpp_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int out_C, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups) {
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
                        const bool cond = h > -1 && w > -1 && h < H&& w < W;
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


//    int H = 4;
//    int W = 4;
//    int H = 4;
//    int W = 4;

//    int H = 64;
//    int W = 64;


/*
对应test2_00002_matmul5.cpp
中的
    int batch_size = 65536;
    int ch_in = 64*9;
    int ch_out = 64;
测试用例，该矩阵乘法用例在i5-9400F (win10) 上，用时约30ms （相当于卷积层中不包括im2col的时间）
希望winograd上该用例用时低于30ms
*/
    int H = 256;
    int W = 256;
    int in_C = 64;
    int out_C = 64;


//    int in_C = 1;
//    int out_C = 1;
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
    const int im2col_numel = N * out_H * out_W * kH * kW * in_C;
    const int weight_numel = kH * kW * in_C * out_C;
    const int out_numel = N * out_H * out_W * out_C;
    float* input = new float[input_numel];
    float* im2col = new float[im2col_numel];
    float* weight = new float[weight_numel];
    float* out_true = new float[out_numel];
    float* out = new float[out_numel];


    int tile_size = 4;  // 一次计算输出的2x2像素瓦片，对应的输入是原图的lxl像素瓦片，这里l=4
    int otile_size = 2;  // 一次计算输出的2x2像素瓦片，对应的输入是原图的lxl像素瓦片，这里l=4
    int pad_H = H + padding_h + padding_h;
    int pad_W = W + padding_w + padding_w;
    if (H % 2 == 0 && W % 2 == 0)
    {
        ;
    }
    else
    {
        return 0;
    }
    const int pad_input_numel = N * pad_H * pad_W * in_C;
    float* pad_input = new float[pad_input_numel];

    int tile_num = int((out_H + otile_size - 1) / otile_size) * int((out_W + otile_size - 1) / otile_size);  // 像素瓦片个数
    printf("tile_num=%d\n", tile_num);
    float* winograd23_U = new float[tile_size * tile_size * in_C * out_C];
    float* winograd23_V = new float[tile_size * tile_size * tile_num * in_C];
    float* winograd23_Q = new float[tile_size * tile_size * tile_num * out_C];
    float* winograd23_Y = new float[otile_size * otile_size * tile_num * out_C];  // Q = V * U
    float* winograd23_O = new float[tile_size * tile_size * in_C * out_C];


    printf("======================== init ========================\n");
    printf("input      = ");
//    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < input_numel; i++)
    {
        int ttt = rand() % 2000;
        float val = (float)ttt / 1000.f - 1.f;
//        int ttt = i + 1;
//        float val = (float)ttt;
//        printf("%f,", val);
        *(input + i) = val;
    }
    printf("\nweight      = ");
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < weight_numel; i++)
    {
        int ttt = rand() % 2000;
//        int ttt = i + 100;
        float val = (float)ttt / 10.f + 0.5f;
//        printf("%f,", val);
        *(weight + i) = val;
    }
    imNHWC2col_cpp_kernel<float>(num_threads_, input, im2col, im2col_numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
    matmul_true_cpp_kernel(num_threads_, im2col, weight, out_true, N * out_H * out_W, kH * kW * in_C, out_C);

    winograd23_weight_transform(num_threads_, weight, winograd23_U, tile_size, in_C, out_C);



    float diff = 0.0;


    printf("======================== calc ========================\n");
    for (int batch_idx = 0; batch_idx < 30; batch_idx++)
    {
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < out_numel; i++)
        {
            *(out + i) = 0.f;
        }
        auto startTime = std::chrono::system_clock::now();


        pad4d_dim23_cpp_kernel(num_threads_, input, pad_input, N, H, W, in_C, pad_H, pad_W, padding_h, padding_w);
//        winograd23_input_cpp_transform(num_threads_, pad_input, winograd23_V, in_C, tile_num, out_W, pad_W, otile_size);
        winograd23_input_x86_transform(num_threads_, pad_input, winograd23_V, in_C, tile_num, out_W, pad_W, otile_size);


        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < tile_size * tile_size * tile_num * out_C; i++)
        {
            *(winograd23_Q + i) = 0.f;
        }

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < tile_size; i++) {
            for (int j = 0; j < tile_size; j++) {
//                matmul_true_cpp_kernel(num_threads_, winograd23_V + (i * tile_size + j) * tile_num * in_C, winograd23_U + (i * tile_size + j) * in_C * out_C, winograd23_Q + (i * tile_size + j) * tile_num * out_C, tile_num, in_C, out_C);
                //matmul_block_pack_8_8_8_SIMD_consider_mod_x86_kernel<float>(num_threads_, winograd23_V + (i * tile_size + j) * tile_num * in_C, winograd23_U + (i * tile_size + j) * in_C * out_C, winograd23_Q + (i * tile_size + j) * tile_num * out_C, tile_num, in_C, out_C);
            }
        }
        //winograd23_output_transform(num_threads_, winograd23_Q, winograd23_Y, out_C, tile_num, out_W, otile_size, tile_size);
        //winograd23_output_relayout(num_threads_, winograd23_Y, out, out_C, tile_num, out_W, pad_W, otile_size);


        auto endTime = std::chrono::system_clock::now();
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("matmul forward cost_time = %f ms\n", cost_ms);
        diff = calc_diff(out, out_true, out_numel);
        printf("diff=%f (%s)\n", diff, "y");

//        printf("\n\n");
//        for (int i = 0; i < out_numel; i++)
//        {
//            printf("%f,", *(out + i));
//        }
//        printf("\n\n");
//        for (int i = 0; i < out_numel; i++)
//        {
//            printf("%f,", *(out_true + i));
//        }
//        printf("\n\n");
    }


    diff = calc_diff(out, out_true, out_numel);
    printf("diff=%f (%s)\n", diff, "y");

    delete input;
    delete out;
    delete out_true;

    return 0;
}