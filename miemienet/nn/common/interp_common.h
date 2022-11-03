#ifndef __F_INTERP_COMMON_H__
#define __F_INTERP_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void interp(Tensor* input, Tensor* output, int size_h=0, int size_w=0, float scale_h=-1.f, float scale_w=-1.f, char* mode="nearest", bool align_corners=false, bool recompute_scale_factor=false);

NS_MM_F_END

#endif // __F_INTERP_COMMON_H__
