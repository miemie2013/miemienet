#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>

#include <immintrin.h>



/*
cd python_SIMD_simulator

g++ test2_000_avx.cpp -o test2_000_avx.out -w -march=native

./test2_000_avx.out


https://blog.csdn.net/zachariah2000/article/details/120731767
需要加上 -mavx 或者 -march=native

_mm256_loadu_ps()、_mm256_load_ps()的区别
_mm256_storeu_ps()、 _mm256_store_ps()的区别


https://www.cnblogs.com/hellowooorld/p/11529078.html

查文档
https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=3828,301,2553


*/

int main(int argc, char** argv){
    const int elempack = 8;
    const int out_elempack = 8;
    const int wstep = out_elempack * elempack;
    const float zeros[out_elempack] = {1.f, 2.f, 3.f, 3.f, 7.f, 8.f, 5.f, 6.f};
    const float zeros2[out_elempack] = {1.f, 2.f, 3.f, 3.f, 7.f, 8.f, 5.f, 6.f};
    const float zeros3[out_elempack] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float zeros4[out_elempack] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float* outptr = new float[out_elempack];

    int oc = 0;
    __m256 _sum = _mm256_setzero_ps();
    __m256 _v1 = _mm256_loadu_ps(zeros + oc * out_elempack);
    __m256 _v2 = _mm256_loadu_ps(zeros2 + oc * out_elempack);
//    _sum = _mm256_comp_fmadd_ps(_v1, _v2, _sum);
//    _sum = _mm256_mask_fmadd_ps(_v1, _v2, _sum);

//    _sum = _mm256_add_ps(_v1, _v2);
//    _sum = _mm256_mul_ps(_v1, _v2);
//    _sum = _mm256_addsub_ps(_v1, _v2);
    _sum = _mm256_fmadd_ps(_v1, _v2, _sum);

    // 再一次累加到_sum
    __m256 _v3 = _mm256_loadu_ps(zeros3 + oc * out_elempack);
    __m256 _v4 = _mm256_loadu_ps(zeros4 + oc * out_elempack);
    _sum = _mm256_fmadd_ps(_v3, _v4, _sum);
    _mm256_storeu_ps(outptr + 0 * out_elempack, _sum);

    // 测试广播
    _sum = _mm256_broadcast_ss(zeros4 + 0);
    _mm256_storeu_ps(outptr + 0 * out_elempack, _sum);


    for (int i = 0; i < out_elempack; i++)
    {
        printf("y=%f (%s)\n", outptr[i], "y");
    }
    return 0;
}

