#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "transpose.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/transpose_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/transpose_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Transpose::Transpose(int transpose_type)
{
    this->transpose_type = transpose_type;
}

Transpose::~Transpose()
{
}

Tensor* Transpose::create_tensors(Tensor* input)
{
    input->referenceCount++;
    input_tensors->push_back(input);

    Tensor* output;
// gen forward code start
    if (transpose_type == TRANS2D_01) {
        if (input->dims != 2) {
            printf("Error from transpose op, transpose_type == TRANS2D_01, input->dims != 2\n");
            exit(1);
        }
        const int H = input->shape->at(0);
        const int W = input->shape->at(1);
        output = new SNT Tensor(MMSHAPE2D(H, W), FP32, false, false);
    }
    else if (transpose_type == TRANS2D_10) {
        if (input->dims != 2) {
            printf("Error from transpose op, transpose_type == TRANS2D_10, input->dims != 2\n");
            exit(1);
        }
        const int H = input->shape->at(0);
        const int W = input->shape->at(1);
        output = new SNT Tensor(MMSHAPE2D(W, H), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_012) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_012, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(N, H, W), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_021) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_021, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(N, W, H), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_102) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_102, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(H, N, W), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_120) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_120, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(H, W, N), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_201) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_201, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(W, N, H), FP32, false, false);
    }
    else if (transpose_type == TRANS3D_210) {
        if (input->dims != 3) {
            printf("Error from transpose op, transpose_type == TRANS3D_210, input->dims != 3\n");
            exit(1);
        }
        const int N = input->shape->at(0);
        const int H = input->shape->at(1);
        const int W = input->shape->at(2);
        output = new SNT Tensor(MMSHAPE3D(W, H, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, C, H, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, C, W, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, H, C, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, H, W, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, W, C, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(N, W, H, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, N, H, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, N, W, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, H, N, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, H, W, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, W, N, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(C, W, H, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, N, C, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, N, W, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, C, N, W), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, C, W, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, W, N, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(H, W, C, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, N, C, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, N, H, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, C, N, H), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, C, H, N), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, H, N, C), FP32, false, false);
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
        output = new SNT Tensor(MMSHAPE4D(W, H, C, N), FP32, false, false);
    }
// gen forward code end
    output->referenceCount++;
    output_tensors->push_back(output);
    return output;
}

Tensor* Transpose::feed_forward(Tensor* input)
{
    Tensor* output = output_tensors->at(0);
    miemienet::functional::transpose(input, output, transpose_type);
    return output;
}

NS_MM_END
