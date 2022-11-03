#ifndef __F_MATMUL_COMMON_H__
#define __F_MATMUL_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void matmul_transB(Tensor* input, Tensor* weight, Tensor* output);

void matmul_transA(Tensor* input, Tensor* weight, Tensor* output);

void matmul(Tensor* input, Tensor* weight, Tensor* output);

void matmul_depthwise(Tensor* input, Tensor* group_weights, Tensor* output, int groups);

// 额外对结果矩阵转置
//Tensor* matmul_transB_transY(Tensor* input, Tensor* weight, bool create_graph);
//
//Tensor* matmul_transA_transY(Tensor* input, Tensor* weight, bool create_graph);
//
//Tensor* matmul_transY(Tensor* input, Tensor* weight, bool create_graph);

NS_MM_F_END

#endif // __F_MATMUL_COMMON_H__
