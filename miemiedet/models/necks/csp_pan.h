#ifndef __MIEMIEDET_CSP_PAN_H__
#define __MIEMIEDET_CSP_PAN_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class CSPPANConvBNLayer : public Layer
{
public:
    CSPPANConvBNLayer(int in_channel=96, int out_channel=96, int kernel_size=3, int stride=1, int groups=1, char* act_name="leakyrelu");
    ~CSPPANConvBNLayer();

    Conv2d* conv;
//    BatchNorm2d* bn;
    Layer* bn;
    Activation* act;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class Channel_T : public Layer
{
public:
    Channel_T(std::vector<int>* in_channels=nullptr, int out_channels=96, char* act_name="leakyrelu");
    ~Channel_T();

    LayerList* convs;

    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013);
private:
};

class DPModule : public Layer
{
public:
    DPModule(int in_channel=96, int out_channel=96, int kernel_size=3, int stride=1, char* act_name="leakyrelu", bool use_act_in_out=true);
    ~DPModule();

    Conv2d* dwconv;
    Conv2d* pwconv;
//    BatchNorm2d* bn;
    Layer* bn1;
    Layer* bn2;
    Activation* act1;
    Activation* act2;

    bool use_act_in_out;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class LCPAN : public Layer
{
public:
    LCPAN(std::vector<int>* in_channels=nullptr, int out_channels=96, int kernel_size=5, int num_features=3, bool use_depthwise=true, char* act_name="hardswish", std::vector<float>* spatial_scales=nullptr);
    ~LCPAN();

    Channel_T* conv_t;
    Layer* first_top_conv;
    Layer* second_top_conv;
    Interp* upsample;
    LayerList* top_down_blocks;
    LayerList* downsamples;
    LayerList* bottom_up_blocks;
    Concat* concat;

    int in_channels;
    int num_features;
    std::vector<float>* spatial_scales;

    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_CSP_PAN_H__
