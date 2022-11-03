#ifndef __MIEMIENET_H__
#define __MIEMIENET_H__

#include "macros.h"

#include "framework/config.h"
#include "framework/layer.h"
#include "framework/memoryallocator.h"
#include "framework/tensor.h"
#include "framework/tensoridmanager.h"

#include "nn/activation.h"
#include "nn/avgpool2d.h"
#include "nn/concat.h"
#include "nn/conv2d.h"
#include "nn/interp.h"
#include "nn/layerlist.h"
#include "nn/maxpool2d.h"
#include "nn/reduce.h"
#include "nn/sequential.h"
#include "nn/softmax.h"
#include "nn/transpose.h"

#if BACKEND_X86
#include "nn/common/activation_common.h"
#include "nn/common/avgpool2d_common.h"
#include "nn/common/concat_common.h"
#include "nn/common/conv2d_common.h"
#include "nn/common/elementwise_common.h"
#include "nn/common/interp_common.h"
#include "nn/common/maxpool2d_common.h"
#include "nn/common/reduce_common.h"
#include "nn/common/softmax_common.h"
#include "nn/common/transpose_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "nn/common/activation_common.h"
#include "nn/common/avgpool2d_common.h"
#include "nn/common/concat_common.h"
#include "nn/common/conv2d_common.h"
#include "nn/common/elementwise_common.h"
#include "nn/common/interp_common.h"
#include "nn/common/maxpool2d_common.h"
#include "nn/common/reduce_common.h"
#include "nn/common/softmax_common.h"
#include "nn/common/transpose_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

const char* miemienetVersion();

NS_MM_END

#endif // __MIEMIENET_H__
