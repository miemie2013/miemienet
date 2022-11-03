#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "elementwise_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM

NS_MM_F_BEGIN


// gen cpp code start
template<typename data_t>
void elem4d_NCHW_add_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }

    // x86
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
//    int offset_ = num_packs * elempack;
//    if (num - offset_ >= 4)
//    {
//        const float* x_ptr = x + offset_;
//        const float* y_ptr = y + offset_;
//        float* z_ptr = z + offset_;
//        __m128 _a = _mm_load_ps(x_ptr);
//        __m128 _b = _mm_load_ps(y_ptr);
//        __m128 _out = _mm_add_ps(_a, _b);
//        _mm_store_ps(z_ptr, _out);
//        offset_ += 4;
//    }
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int i = offset_; i < num; i++) {
//        z[i] = x[i] + y[i];
//    }
}

template<typename data_t>
void elem4d_NCHW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }

    // x86
//    const int elempack = 8;
//    const int num_packs = num / elempack;
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int pid = 0; pid < num_packs; pid++) {
//        const float* x_ptr = x + pid * elempack;
//        const float* y_ptr = y + pid * elempack;
//        float* z_ptr = z + pid * elempack;
//        __m256 _a = _mm256_loadu_ps(x_ptr);
//        __m256 _b = _mm256_loadu_ps(y_ptr);
//        __m256 _out = _mm256_mul_ps(_a, _b);
//        _mm256_storeu_ps(z_ptr, _out);
//    }
//    int offset_ = num_packs * elempack;
//    if (num - offset_ >= 4)
//    {
//        const float* x_ptr = x + offset_;
//        const float* y_ptr = y + offset_;
//        float* z_ptr = z + offset_;
//        __m128 _a = _mm_load_ps(x_ptr);
//        __m128 _b = _mm_load_ps(y_ptr);
//        __m128 _out = _mm_mul_ps(_a, _b);
//        _mm_store_ps(z_ptr, _out);
//        offset_ += 4;
//    }
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int i = offset_; i < num; i++) {
//        z[i] = x[i] * y[i];
//    }
}

template<typename data_t>
void elem4d_NCHW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[0], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[0], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[0], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[0], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[0], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[0], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[0], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = std::min(x[0], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_NCHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_N11W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[c], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[c], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1CHW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[0], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_11HW_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[0], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1C1W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[0], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1CH1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[0], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_111W_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[0], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_11H1_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[0], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1C11_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[0], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1111_cpp_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = std::max(x[0], y[0]);
                }
            }
        }
    }
}

// gen cpp code end


// gen x86 code start
#if BACKEND_X86
template<typename data_t>
void elem4d_NCHW_add_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] + y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] + y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] + y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] + y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] + y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] + y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] + y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_add_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] + y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] - y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] - y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] - y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] - y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] - y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] - y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] - y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_sub_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] - y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] * y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] * y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] * y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] * y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] * y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] * y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] * y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_mul_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] * y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[((n * C + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = x[((n * C + c) * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[(c * H + h) * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * W + w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c * W + w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c * H + h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c * H + h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[n * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = x[n * W + w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[w] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[w] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[w] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[w] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[h] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[h] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[h] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[h] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[c] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[c] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[c] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[c] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = x[0] / y[(c * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = x[0] / y[h * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = x[0] / y[c * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = x[0] / y[c * H + h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = x[0] / y[w];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = x[0] / y[h];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = x[0] / y[c];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_div_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = x[0] / y[0];
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::min(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::min(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[c], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[c], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[c], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[c], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::min(x[0], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::min(x[0], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::min(x[0], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::min(x[0], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::min(x[0], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::min(x[0], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::min(x[0], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_min_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = std::min(x[0], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_NCHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[((n * C + c) * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_NCHW_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[((n * C + c) * H + h) * W + w] = std::max(x[((n * C + c) * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CHW_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[(c * H + h) * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11HW_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * W + w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C1W_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c * W + w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c * H + h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1CH1_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c * H + h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_N11W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[n * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_N11W_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[n * W + w] = std::max(x[n * W + w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[w], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[w], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[w], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_111W_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[w], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[h], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[h], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[h], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[h], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[h], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_11H1_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[h], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[c], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[c], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[c], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[c], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1C11_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[c], y[0]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1CHW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[(c * H + h) * W + w] = std::max(x[0], y[(c * H + h) * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_11HW_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h * W + w] = std::max(x[0], y[h * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1C1W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * W + w] = std::max(x[0], y[c * W + w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1CH1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c * H + h] = std::max(x[0], y[c * H + h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_111W_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[w] = std::max(x[0], y[w]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_11H1_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[h] = std::max(x[0], y[h]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1C11_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[c] = std::max(x[0], y[c]);
                }
            }
        }
    }
}

template<typename data_t>
void elem4d_1111_max_1111_x86_kernel(const int num_threads_, const data_t* x, const data_t* y, data_t* z, int num, int N, int C, int H, int W) {
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    z[0] = std::max(x[0], y[0]);
                }
            }
        }
    }
}


#endif // BACKEND_X86
// gen x86 code end




void elementwise(Tensor* a, Tensor* b, Tensor* out, int op_type)
{
/*
op_type类型的宏定义见macros.h
#define ELE_ADD 0
#define ELE_SUB 1
#define ELE_MUL 2
#define ELE_DIV 3
#define ELE_MIN 4
#define ELE_MAX 5
*/
    const int dims = a->dims;
    if (dims != b->dims)
    {
        printf("Error from elementwise op, a->dims != b->dims.\n");
        a->print_msg("a");
        b->print_msg("b");
        exit(1);
    }
    if (op_type < 0 || op_type > 5)
    {
        printf("elementwise unsupported op_type.\n");
        exit(1);
    }
    if (dims > 4)
    {
        printf("elementwise unsupported dims=%d.\n", dims);
        exit(1);
    }

    if (dims == 3)
    {
        a->unsqueeze(0);
        b->unsqueeze(0);
    }
    else if (dims == 2)
    {
        a->unsqueeze(0);
        a->unsqueeze(0);
        b->unsqueeze(0);
        b->unsqueeze(0);
    }
    else if (dims == 1)
    {
        a->unsqueeze(0);
        a->unsqueeze(0);
        a->unsqueeze(0);
        b->unsqueeze(0);
        b->unsqueeze(0);
        b->unsqueeze(0);
    }


    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;

    if (cfg->use_cpp_compute)
    {
// gen cpp invoke code start
        const int N0 = a->shape->at(0);
        const int C0 = a->shape->at(1);
        const int H0 = a->shape->at(2);
        const int W0 = a->shape->at(3);
        const int N1 = b->shape->at(0);
        const int C1 = b->shape->at(1);
        const int H1 = b->shape->at(2);
        const int W1 = b->shape->at(3);
        const int N = std::max(N0, N1);
        const int C = std::max(C0, C1);
        const int H = std::max(H0, H1);
        const int W = std::max(W0, W1);
        if (op_type == ELE_ADD) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_add_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_add_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_SUB) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_sub_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_sub_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MUL) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_mul_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_mul_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_DIV) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_div_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_div_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MIN) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_min_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_min_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MAX) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_max_NCHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_max_N11W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_1CHW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_11HW_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_1C1W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1CH1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_111W_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_11H1_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1C11_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1111_cpp_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
// gen cpp invoke code end
    }
    else
    {
// gen x86 invoke code start
#if BACKEND_X86
        const int N0 = a->shape->at(0);
        const int C0 = a->shape->at(1);
        const int H0 = a->shape->at(2);
        const int W0 = a->shape->at(3);
        const int N1 = b->shape->at(0);
        const int C1 = b->shape->at(1);
        const int H1 = b->shape->at(2);
        const int W1 = b->shape->at(3);
        const int N = std::max(N0, N1);
        const int C = std::max(C0, C1);
        const int H = std::max(H0, H1);
        const int W = std::max(W0, W1);
        if (op_type == ELE_ADD) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_add_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_add_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_add_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_add_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_SUB) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_sub_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_sub_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_sub_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_sub_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MUL) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_mul_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_mul_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_mul_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_mul_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_DIV) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_div_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_div_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_div_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_div_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MIN) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_min_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_min_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_min_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_min_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
        else if (op_type == ELE_MAX) {
            if (N0 == N1 && N1 > 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_max_NCHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_NCHW_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_NCHW_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1CHW_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1CHW_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_11HW_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_11HW_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_1C1W_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_1C1W_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1CH1_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1CH1_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == N1 && N1 > 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_max_N11W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 > 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_N11W_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == W1 && W1 > 1) {
                elem4d_111W_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 > 1 && W1 == 1) {
                elem4d_111W_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_11H1_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == H1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 > 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_11H1_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1C11_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == C1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 > 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1C11_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_1CHW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_11HW_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_1C1W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1CH1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 > 1) {
                elem4d_1111_max_111W_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 > 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_11H1_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 > 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1C11_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else if (N0 == 1 && N1 == 1 && C0 == 1 && C1 == 1 && H0 == 1 && H1 == 1 && W0 == 1 && W1 == 1) {
                elem4d_1111_max_1111_x86_kernel<float>(num_threads_, a->data_fp32, b->data_fp32, out->data_fp32, out->numel, N, C, H, W);
            }
            else {
                printf("Error from elementwise op, (%d, %d, %d, %d) op (%d, %d, %d, %d) not implemented!\n", N0, C0, H0, W0, N1, C1, H1, W1);
                exit(1);
            }
        }
#endif // BACKEND_X86
// gen x86 invoke code end
    }


    if (dims == 3)
    {
        a->squeeze(0);
        b->squeeze(0);
        out->squeeze(0);
    }
    else if (dims == 2)
    {
        a->squeeze(0);
        a->squeeze(0);
        b->squeeze(0);
        b->squeeze(0);
        out->squeeze(0);
        out->squeeze(0);
    }
    else if (dims == 1)
    {
        a->squeeze(0);
        a->squeeze(0);
        a->squeeze(0);
        b->squeeze(0);
        b->squeeze(0);
        b->squeeze(0);
        out->squeeze(0);
        out->squeeze(0);
        out->squeeze(0);
    }
}

NS_MM_F_END
