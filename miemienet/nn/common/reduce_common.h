#ifndef __F_REDUCE_COMMON_H__
#define __F_REDUCE_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void reduce(Tensor* input, Tensor* output, std::vector<int>* dims, bool keepdim, int op_type);

NS_MM_F_END

#endif // __F_SOFTMAX_COMMON_H__
