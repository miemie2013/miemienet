#ifndef __F_MAXPOOL2D_COMMON_H__
#define __F_MAXPOOL2D_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void maxpool2d(Tensor* input, Tensor* output, int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, bool ceil_mode=false);

NS_MM_F_END

#endif // __F_MAXPOOL2D_COMMON_H__
