#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "../../framework/config.h"
#include "../../framework/tensor.h"
#include "../transpose.h"
#include "transpose_common.h"

//#include "atomic_common.h"
//#include "elementwise_common.h"
//#include "matmul_common.h"
//#include "reduce_common.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM


NS_MM_F_BEGIN

// gen cpp code start
template<typename data_t>
void transpose2d_01_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int H, int W) {
    // y[h][w] = x[h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            y[(h * W) + w] = x[(h * W) + w];
        }
    }
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

template<typename data_t>
void transpose3d_012_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[n][h][w] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((n * H) + h) * W + w] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_021_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[n][w][h] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((n * W) + w) * H + h] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_102_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[h][n][w] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((h * N) + n) * W + w] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_120_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[h][w][n] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((h * W) + w) * N + n] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_201_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[w][n][h] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((w * N) + n) * H + h] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_210_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[w][h][n] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((w * H) + h) * N + n] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose4d_0123_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][c][h][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * C) + c) * H + h) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0132_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][c][w][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * C) + c) * W + w) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0213_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][h][c][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * H) + h) * C + c) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0231_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][h][w][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * H) + h) * W + w) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0312_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][w][c][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * W) + w) * C + c) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0321_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][w][h][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * W) + w) * H + h) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1023_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][n][h][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * N) + n) * H + h) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1032_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][n][w][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * N) + n) * W + w) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1203_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][h][n][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * H) + h) * N + n) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1230_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][h][w][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * H) + h) * W + w) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1302_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][w][n][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * W) + w) * N + n) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1320_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][w][h][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * W) + w) * H + h) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2013_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][n][c][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * N) + n) * C + c) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2031_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][n][w][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * N) + n) * W + w) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2103_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][c][n][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * C) + c) * N + n) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2130_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][c][w][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * C) + c) * W + w) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2301_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][w][n][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * W) + w) * N + n) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2310_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][w][c][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * W) + w) * C + c) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3012_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][n][c][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * N) + n) * C + c) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3021_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][n][h][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * N) + n) * H + h) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3102_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][c][n][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * C) + c) * N + n) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3120_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][c][h][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * C) + c) * H + h) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3201_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][h][n][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * H) + h) * N + n) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3210_cpp_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][h][c][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * H) + h) * C + c) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

// gen cpp code end

// gen x86 code start
#if BACKEND_X86
template<typename data_t>
void transpose2d_01_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int H, int W) {
    // y[h][w] = x[h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            y[(h * W) + w] = x[(h * W) + w];
        }
    }
}

template<typename data_t>
void transpose2d_10_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int H, int W) {
    // y[w][h] = x[h][w]
    // 读连续
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int h = 0; h < H; h++) {
//        for (int w = 0; w < W; w++) {
//            y[(w * H) + h] = x[(h * W) + w];
//        }
//    }

    // 写连续
    #pragma omp parallel for num_threads(num_threads_)
    for (int w = 0; w < W; w++) {
        for (int h = 0; h < H; h++) {
            y[(w * H) + h] = x[(h * W) + w];
        }
    }


// SIMD优化 https://blog.csdn.net/yan31415/article/details/107464744

//    const int BLOCK = 128;
    // 写连续
//    #pragma omp parallel for num_threads(num_threads_)
//    for (int w = 0; w < W; w+=BLOCK) {
//        for (int h = 0; h < H; h+=BLOCK) {
//            for (int i = 0; i < BLOCK; i++) {
//                for (int j = 0; j < BLOCK; j++) {
//                    y[((w + i) * H) + (h + j)] = x[((h + j) * W) + (w + i)];
//                }
//            }
//        }
//    }
}

template<typename data_t>
void transpose3d_012_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[n][h][w] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((n * H) + h) * W + w] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_021_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[n][w][h] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((n * W) + w) * H + h] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_102_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[h][n][w] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((h * N) + n) * W + w] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_120_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[h][w][n] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((h * W) + w) * N + n] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_201_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[w][n][h] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((w * N) + n) * H + h] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose3d_210_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int H, int W) {
    // y[w][h][n] = x[n][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                y[((w * H) + h) * N + n] = x[((n * H) + h) * W + w];
            }
        }
    }
}

template<typename data_t>
void transpose4d_0123_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][c][h][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * C) + c) * H + h) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0132_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][c][w][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * C) + c) * W + w) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0213_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][h][c][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * H) + h) * C + c) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0231_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][h][w][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * H) + h) * W + w) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0312_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][w][c][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * W) + w) * C + c) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_0321_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[n][w][h][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((n * W) + w) * H + h) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1023_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][n][h][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * N) + n) * H + h) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1032_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][n][w][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * N) + n) * W + w) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1203_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][h][n][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * H) + h) * N + n) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1230_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][h][w][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * H) + h) * W + w) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1302_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][w][n][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * W) + w) * N + n) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_1320_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[c][w][h][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((c * W) + w) * H + h) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2013_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][n][c][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * N) + n) * C + c) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2031_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][n][w][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * N) + n) * W + w) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2103_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][c][n][w] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * C) + c) * N + n) * W + w] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2130_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][c][w][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * C) + c) * W + w) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2301_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][w][n][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * W) + w) * N + n) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_2310_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[h][w][c][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((h * W) + w) * C + c) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3012_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][n][c][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * N) + n) * C + c) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3021_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][n][h][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * N) + n) * H + h) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3102_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][c][n][h] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * C) + c) * N + n) * H + h] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3120_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][c][h][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * C) + c) * H + h) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3201_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][h][n][c] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * H) + h) * N + n) * C + c] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}

template<typename data_t>
void transpose4d_3210_x86_kernel(const int num_threads_, const data_t* x, data_t* y, int num, int N, int C, int H, int W) {
    // y[w][h][c][n] = x[n][c][h][w]
    #pragma omp parallel for num_threads(num_threads_)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    y[(((w * H) + h) * C + c) * N + n] = x[(((n * C) + c) * H + h) * W + w];
                }
            }
        }
    }
}


#endif // BACKEND_X86
// gen x86 code end


void transpose(Tensor* input, Tensor* output, int transpose_type)
{
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;
    Tensor* out = nullptr;

    bool use_cpp_compute = cfg->use_cpp_compute;
    use_cpp_compute = false;
    if (use_cpp_compute) {
// gen cpp invoke code start
        if (transpose_type == TRANS2D_01) {
            if (input->dims != 2) {
                printf("Error from transpose op, transpose_type == TRANS2D_01, input->dims != 2\n");
                exit(1);
            }
            const int H = input->shape->at(0);
            const int W = input->shape->at(1);
            transpose2d_01_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, H, W);
        }
        else if (transpose_type == TRANS2D_10) {
            if (input->dims != 2) {
                printf("Error from transpose op, transpose_type == TRANS2D_10, input->dims != 2\n");
                exit(1);
            }
            const int H = input->shape->at(0);
            const int W = input->shape->at(1);
            transpose2d_10_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, H, W);
        }
        else if (transpose_type == TRANS3D_012) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_012, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_012_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_021) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_021, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_021_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_102) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_102, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_102_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_120) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_120, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_120_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_201) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_201, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_201_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_210) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_210, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_210_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS4D_0123) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0123, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0123_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0132) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0132, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0132_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0213) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0213, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0213_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0231) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0231, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0231_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0312) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0312, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0312_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0321) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0321, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0321_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1023) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1023, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1023_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1032) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1032, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1032_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1203) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1203, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1203_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1230) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1230, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1230_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1302) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1302, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1302_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1320) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1320, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1320_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2013) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2013, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2013_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2031) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2031, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2031_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2103) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2103, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2103_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2130) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2130, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2130_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2301) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2301, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2301_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2310) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2310, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2310_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3012) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3012, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3012_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3021) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3021, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3021_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3102) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3102, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3102_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3120) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3120, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3120_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3201) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3201, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3201_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3210) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3210, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3210_cpp_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
// gen cpp invoke code end
    } else {
// gen x86 invoke code start
#if BACKEND_X86
        if (transpose_type == TRANS2D_01) {
            if (input->dims != 2) {
                printf("Error from transpose op, transpose_type == TRANS2D_01, input->dims != 2\n");
                exit(1);
            }
            const int H = input->shape->at(0);
            const int W = input->shape->at(1);
            transpose2d_01_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, H, W);
        }
        else if (transpose_type == TRANS2D_10) {
            if (input->dims != 2) {
                printf("Error from transpose op, transpose_type == TRANS2D_10, input->dims != 2\n");
                exit(1);
            }
            const int H = input->shape->at(0);
            const int W = input->shape->at(1);
            transpose2d_10_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, H, W);
        }
        else if (transpose_type == TRANS3D_012) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_012, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_012_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_021) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_021, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_021_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_102) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_102, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_102_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_120) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_120, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_120_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_201) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_201, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_201_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS3D_210) {
            if (input->dims != 3) {
                printf("Error from transpose op, transpose_type == TRANS3D_210, input->dims != 3\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int H = input->shape->at(1);
            const int W = input->shape->at(2);
            transpose3d_210_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, H, W);
        }
        else if (transpose_type == TRANS4D_0123) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0123, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0123_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0132) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0132, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0132_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0213) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0213, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0213_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0231) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0231, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0231_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0312) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0312, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0312_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_0321) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_0321, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_0321_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1023) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1023, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1023_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1032) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1032, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1032_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1203) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1203, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1203_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1230) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1230, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1230_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1302) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1302, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1302_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_1320) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_1320, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_1320_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2013) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2013, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2013_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2031) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2031, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2031_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2103) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2103, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2103_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2130) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2130, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2130_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2301) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2301, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2301_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_2310) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_2310, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_2310_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3012) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3012, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3012_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3021) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3021, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3021_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3102) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3102, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3102_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3120) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3120, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3120_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3201) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3201, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3201_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
        else if (transpose_type == TRANS4D_3210) {
            if (input->dims != 4) {
                printf("Error from transpose op, transpose_type == TRANS4D_3210, input->dims != 4\n");
                exit(1);
            }
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            transpose4d_3210_x86_kernel<float>(num_threads_, input->data_fp32, output->data_fp32, input->numel, N, C, H, W);
        }
#endif // BACKEND_X86
// gen x86 invoke code end

#if BACKEND_ARM
#endif // BACKEND_ARM
    }
}

NS_MM_F_END
