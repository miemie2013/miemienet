#ifndef __MIEMIEDET_LCNET_H__
#define __MIEMIEDET_LCNET_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class LCNetConvBNLayer : public Layer
{
public:
    LCNetConvBNLayer(int num_channels, int filter_size, int num_filters, int stride, int groups=1);
    ~LCNetConvBNLayer();

    Conv2d* conv;
//    BatchNorm2d* bn;
    Layer* bn;
    Activation* act;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class SEModule : public Layer
{
public:
    SEModule(int channel, int reduction=4);
    ~SEModule();

    Conv2d* conv1;
    Conv2d* conv2;
    Activation* relu;
    Activation* hardsigmoid;
    Reduce* avg_pool;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class DepthwiseSeparable : public Layer
{
public:
    DepthwiseSeparable(int num_channels, int num_filters, int stride, int dw_size=3, bool use_se=false);
    ~DepthwiseSeparable();

    LCNetConvBNLayer* dw_conv;
    LCNetConvBNLayer* pw_conv;
    SEModule* se;

    bool use_se;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class LCNet : public Layer
{
public:
    LCNet(float scale=1.f, std::vector<int>* feature_maps=nullptr);
    ~LCNet();

    LCNetConvBNLayer* conv1;
    Sequential* blocks2;
    Sequential* blocks3;
    Sequential* blocks4;
    Sequential* blocks5;
    Sequential* blocks6;

    float scale;
    std::vector<int>* feature_maps;

    int make_divisible(float v, int divisor=8, int min_value=-1);
    virtual std::vector<Tensor*>* create_tensors(Tensor* input, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(Tensor* input, char miemie2013);
private:
};


}  // namespace miemiedet

#endif // __MIEMIEDET_LCNET_H__
