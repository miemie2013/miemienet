#ifndef __F_ACTIVATION_COMMON_H__
#define __F_ACTIVATION_COMMON_H__

#include "../../macros.h"

NS_MM_F_BEGIN

void activation(Tensor* input, Tensor* output, char* type, float alpha);

NS_MM_F_END

#endif // __F_ACTIVATION_COMMON_H__
