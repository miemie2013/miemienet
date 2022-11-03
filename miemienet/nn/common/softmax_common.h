#ifndef __F_SOFTMAX_COMMON_H__
#define __F_SOFTMAX_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void softmax(Tensor* input, Tensor* output, int dim=-1);

NS_MM_F_END

#endif // __F_SOFTMAX_COMMON_H__
