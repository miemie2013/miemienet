#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../conv2d.h"
#include "conv2d_common.h"

#include "elementwise_common.h"
#include "matmul_common.h"
#include "transpose_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

/*
template<typename data_t>
void imNCHW2col_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                for (int ic = 0; ic < in_C; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            // im.shape = [N, in_C, H, W]
                            // 求出对应的im元素的坐标 n ic h w
                            int h_in = oh * stride_h - padding_h;
                            int w_in = ow * stride_w - padding_w;
                            const int h = h_in + kh * dilation_h;
                            const int w = w_in + kw * dilation_w;

                            // 越界取0，否则取im[n, ic, h, w]
                            const bool cond = h > -1 && w > -1 && h < H && w < W;
                            float val = cond ? im[(((n * in_C) + ic) * H + h) * W + w] : 0.f;
                            im2col[((((((n * out_H) + oh) * out_W + ow) * in_C + ic) * kH + kh) * kW + kw)] = val;
                        }
                    }
                }
            }
        }
    }
}
*/



/*
当cfg->image_data_format == NHWC时
最初，卷积层的im2col和weight的形状是这样的：
im2col.shape = [N * out_H * out_W, in_C * kH * kW]
weight.shape = [in_C * kH * kW, out_C]
因为图片是NHWC排列，C维变化最快，im2col变化最快的是kW维，
如果im2col想连续访问in_C维，那么1次跳kH * kW个元素，这会导致im2col这一过程非常耗时。

当把im2col改进成和图片相匹配的维度排列时：
weight.shape = [kH * kW * in_C, out_C]
im2col.shape = [N * out_H * out_W, kH * kW * in_C]
即让im2col变化最快的是in_C维，这和图片保持一致，极大地加速了im2col这一过程。
而且，im2col把in_C维放在最后，对SIMD指令优化更友好。

*/


template<typename data_t>
void imNHWC2col_depthwise_cpp_kernel(const int num_threads_, const data_t* im, data_t* im2col, int num, int N, int out_H, int out_W, int in_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    int iC = in_C / groups;
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
//        int i = n * out_H * out_W * kH * kW * in_C;
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
//                            im2col[(((((n * out_H + oh) * out_W + ow) * kH + kh) * kW + kw) * in_C + ic)] = val;
                            im2col[((((((ic / iC * N + n) * out_H + oh) * out_W + ow) * kH + kh) * kW + kw) * iC + ic % iC)] = val;

//                            im2col = new SNT Tensor(MMSHAPE3D(groups, N * out_H * out_W, kH * kW * iC), FP32, false, false);
//                            im2col = new SNT Tensor(MMSHAPE2D(N * out_H * out_W, kH * kW * in_C), FP32, false, false);
//                            im2col[i++] = val;
                        }
                    }
                }
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



#if BACKEND_X86

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

#endif // BACKEND_X86



void naive_conv2d_NHWC_cpp_kernel(const int num_threads_, const float* im, const float* weight, float* out, int num, int N, int out_H, int out_W, int in_C, int out_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * out_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                int j = 0;
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
                            for (int oc = 0; oc < out_C; oc++) {
//                                float _w = weight[(((kh * kW) + kw) * in_C + ic) * out_C + oc];
                                float _w = weight[j++];
                                out[i + oc] += val * _w;
                            }
                        }
                    }
                }
                i += out_C;
            }
        }
    }
}


void naive_conv2d_NHWC_x86_kernel(const int num_threads_, const float* im, const float* weight, float* out, int num, int N, int out_H, int out_W, int in_C, int out_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * out_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                int j = 0;
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
                            for (int oc = 0; oc < out_C; oc++) {
//                                float _w = weight[(((kh * kW) + kw) * in_C + ic) * out_C + oc];
                                float _w = weight[j++];
                                out[i + oc] += val * _w;
                            }
                        }
                    }
                }
                i += out_C;
            }
        }
    }
}


void naive_conv2d_depthwise_NHWC_cpp_kernel(const int num_threads_, const float* im, const float* weight, float* out, int num, int N, int out_H, int out_W, int in_C, int out_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * out_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                int j = 0;
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
//                            float _w = weight[((kh * kW) + kw) * in_C + ic];
                            float _w = weight[j++];
                            out[i + ic] += val * _w;
                        }
                    }
                }
                i += out_C;
            }
        }
    }
}



void naive_conv2d_depthwise_NHWC_x86_kernel(const int num_threads_, const float* im, const float* weight, float* out, int num, int N, int out_H, int out_W, int in_C, int out_C, int kH, int kW, int H, int W, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups){
    int elempack = 8;
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        int i = n * out_H * out_W * out_C;
        for (int oh = 0; oh < out_H; oh++) {
            for (int ow = 0; ow < out_W; ow++) {
                int j = 0;
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

                        int ic = 0;
                        for (; ic + (elempack - 1) < in_C; ic += elempack) {
                            __m256 _val = _mm256_setzero_ps();
                            if (cond)
                                _val = _mm256_loadu_ps(im + (((n * H) + h) * W + w) * in_C + ic);
                            __m256 _w = _mm256_loadu_ps(weight + j);
                            __m256 _res = _mm256_loadu_ps(out + i + ic);
                            _res = _mm256_fmadd_ps(_val, _w, _res);
                            _mm256_storeu_ps(out + i + ic, _res);
                            j += elempack;
                        }
                        for (; ic + 3 < in_C; ic += 4) {
                            __m128 _val = _mm_setzero_ps();
                            if (cond)
                                _val = _mm_load_ps(im + (((n * H) + h) * W + w) * in_C + ic);
                            __m128 _w = _mm_load_ps(weight + j);
                            __m128 _res = _mm_load_ps(out + i + ic);
                            _res = _mm_fmadd_ps(_val, _w, _res);
                            _mm_store_ps(out + i + ic, _res);
                            j += 4;
                        }
                        for (; ic < in_C; ic++) {
                            float val = cond ? im[(((n * H) + h) * W + w) * in_C + ic] : 0.f;
                            float _w = weight[j++];
                            out[i + ic] += val * _w;
                        }



                    }
                }
                i += out_C;
            }
        }
    }
}



void conv2d(Tensor* input, Tensor* weight, Tensor* group_weights, Tensor* bias, Tensor* im2col, Tensor* output_t, Tensor* output, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w, int groups)
{
/*
input.shape =  [N, C, H, W]
weight.shape = [kH * kW * in_C, out_C]
im2col.shape = [N * out_H * out_W, kH * kW * in_C]

output = im2col * weight
output.shape = [N * out_H * out_W, out_C]
*/
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;
    const int N = input->shape->at(0);
    int in_C, H, W, out_C, kH, kW;
    int out_H, out_W;
    if (cfg->image_data_format == NCHW)
    {
        in_C = input->shape->at(1);
        H = input->shape->at(2);
        W = input->shape->at(3);
        out_C = weight->shape->at(0);
        kH = weight->shape->at(2);
        kW = weight->shape->at(3);
        out_H = output->shape->at(2);
        out_W = output->shape->at(3);
    }
    else if (cfg->image_data_format == NHWC)
    {
        H = input->shape->at(1);
        W = input->shape->at(2);
        in_C = input->shape->at(3);
        kH = weight->shape->at(0);
        kW = weight->shape->at(1);
        out_C = weight->shape->at(3);
        out_H = output->shape->at(1);
        out_W = output->shape->at(2);
    }

    bool input_as_im2col = kH == 1 && kW == 1 && stride_h == 1 && stride_w == 1 && padding_h == 0 && padding_w == 0 && groups == 1;

    if (cfg->image_data_format == NCHW)
    {
        ;
    }
    else if (cfg->image_data_format == NHWC)
    {
        if (input_as_im2col)
        {
            input->reshape(MMSHAPE2D(N * out_H * out_W, in_C));
            weight->reshape(MMSHAPE2D(in_C, out_C));
            output->reshape(MMSHAPE2D(N * out_H * out_W, out_C));
            matmul(input, weight, output);  // output.shape = [N * out_H * out_W, out_C]
            weight->reshape(MMSHAPE4D(1, 1, in_C, out_C));
            input->reshape(MMSHAPE4D(N, out_H, out_W, in_C));
        }
        else if (in_C == out_C && groups == out_C)
        {
            // depthwise
            output->set_data_fp32(0.f);   // 先用0初始化
//            naive_conv2d_depthwise_NHWC_cpp_kernel(num_threads_, input->data_fp32, weight->data_fp32, output->data_fp32, input->numel, N, out_H, out_W, in_C, out_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
            naive_conv2d_depthwise_NHWC_x86_kernel(num_threads_, input->data_fp32, weight->data_fp32, output->data_fp32, input->numel, N, out_H, out_W, in_C, out_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);

//            bool use_cpp_compute = cfg->use_cpp_compute;
//            use_cpp_compute = false;
//            if (use_cpp_compute)
//            {
//                imNHWC2col_depthwise_cpp_kernel<float>(num_threads_, input->data_fp32, im2col->data_fp32, im2col->numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//            }
//            else
//            {
//#if BACKEND_X86
//                imNHWC2col_depthwise_cpp_kernel<float>(num_threads_, input->data_fp32, im2col->data_fp32, im2col->numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//#endif // BACKEND_X86
//
//#if BACKEND_ARM
//#endif // BACKEND_ARM
//            }
//            matmul_depthwise(im2col, group_weights, output_t, groups);  // output.shape = [groups, N * out_H * out_W]
//            output->reshape(MMSHAPE2D(N * out_H * out_W, out_C));
//            transpose(output_t, output, TRANS2D_10);
        }
        else if (groups == 1)
        {
//            output->set_data_fp32(0.f);   // 先用0初始化
//            naive_conv2d_NHWC_cpp_kernel(num_threads_, input->data_fp32, weight->data_fp32, output->data_fp32, input->numel, N, out_H, out_W, in_C, out_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
//            naive_conv2d_NHWC_x86_kernel(num_threads_, input->data_fp32, weight->data_fp32, output->data_fp32, input->numel, N, out_H, out_W, in_C, out_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
            bool use_cpp_compute = cfg->use_cpp_compute;
            use_cpp_compute = false;
            if (use_cpp_compute)
            {
                imNHWC2col_cpp_kernel<float>(num_threads_, input->data_fp32, im2col->data_fp32, im2col->numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
            }
            else
            {
#if BACKEND_X86
                imNHWC2col_x86_kernel<float>(num_threads_, input->data_fp32, im2col->data_fp32, im2col->numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
#endif // BACKEND_X86

#if BACKEND_ARM
                imNHWC2col_cpp_kernel<float>(num_threads_, input->data_fp32, im2col->data_fp32, im2col->numel, N, out_H, out_W, in_C, kH, kW, H, W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups);
#endif // BACKEND_ARM
            }
            weight->reshape(MMSHAPE2D(kH * kW * in_C, out_C));
            output->reshape(MMSHAPE2D(N * out_H * out_W, out_C));
            matmul(im2col, weight, output);  // output.shape = [N * out_H * out_W, out_C]
            weight->reshape(MMSHAPE4D(kH, kW, in_C, out_C));
        }
        else
        {
            printf("Error from conv2d op, groups == xxx not impl.\n");
            exit(1);
        }
    }

    output->reshape(MMSHAPE4D(N, out_H, out_W, out_C));
    if (bias)
    {
        elementwise(output, bias, output, ELE_ADD);
    }


    // 使用NCHW排列，还需要额外转置1次。NHWC则不需要。
//    if (cfg->image_data_format == NCHW)
//    {
//        Tensor* out_T = transpose(out, TRANS4D_0312, create_graph);
//        delete out;
//    }
}


void conv2d(Tensor* input, Tensor* weight, Tensor* group_weights, Tensor* bias, Tensor* im2col, Tensor* output_t, Tensor* output, int stride, int padding, int dilation, int groups)
{
    conv2d(input, weight, group_weights, bias, im2col, output_t, output, stride, stride, padding, padding, dilation, dilation, groups);
}

NS_MM_F_END
