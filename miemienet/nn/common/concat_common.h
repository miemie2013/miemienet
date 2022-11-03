#ifndef __F_CONCAT_COMMON_H__
#define __F_CONCAT_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void concat(Tensor* input1, Tensor* input2, Tensor* output, int dim=-1);

void concat(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* output, int dim=-1);

void concat(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4, Tensor* output, int dim=-1);

NS_MM_F_END

#endif // __F_CONCAT_COMMON_H__
