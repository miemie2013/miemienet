#ifndef __F_TRANSPOSE_COMMON_H__
#define __F_TRANSPOSE_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void transpose(Tensor* input, Tensor* output, int transpose_type);

NS_MM_F_END

#endif // __F_TRANSPOSE_COMMON_H__
