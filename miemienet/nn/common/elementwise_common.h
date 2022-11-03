#ifndef __F_ELEMENTWISE_COMMON_H__
#define __F_ELEMENTWISE_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void elementwise(Tensor* a, Tensor* b, Tensor* out, int op_type);

NS_MM_F_END

#endif // __F_ELEMENTWISE_COMMON_H__
