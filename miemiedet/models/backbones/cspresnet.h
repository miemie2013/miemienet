#ifndef __MIEMIEDET_CSPRESNET_H__
#define __MIEMIEDET_CSPRESNET_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class ConvBNLayer : public Layer
{
public:
    ConvBNLayer(int ch_in, int ch_out, int filter_size=3, int stride=1, int groups=1, int padding=0, char* act_name=nullptr);
    ~ConvBNLayer();

    Conv2d* conv;
//    BatchNorm2d* bn;
    Layer* bn;
    Activation* act;

    int ch_in;
    int ch_out;
    int filter_size;
    int stride;
    int groups;
    int padding;
    char* act_name;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class RepVggBlock : public Layer
{
public:
    RepVggBlock(int ch_in, int ch_out, char* act_name="relu");
    ~RepVggBlock();

    ConvBNLayer* conv1;
    ConvBNLayer* conv2;
    Activation* act;

    int ch_in;
    int ch_out;
    char* act_name;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class BasicBlock : public Layer
{
public:
    BasicBlock(int ch_in, int ch_out, char* act_name="relu", bool shortcut=true);
    ~BasicBlock();

    ConvBNLayer* conv1;
    RepVggBlock* conv2;

    int ch_in;
    int ch_out;
    char* act_name;
    bool shortcut;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class EffectiveSELayer : public Layer
{
public:
    EffectiveSELayer(int channels, char* act_name="hardsigmoid");
    ~EffectiveSELayer();

    Conv2d* fc;
    Activation* act;
    Reduce* reduce_mean;

    int channels;
    char* act_name;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class CSPResStage : public Layer
{
public:
    CSPResStage(int ch_in, int ch_out, int n, int stride, char* act_name="relu", bool use_attn=true);
    ~CSPResStage();

    Sequential* blocks;
    ConvBNLayer* conv_down;
    ConvBNLayer* conv1;
    ConvBNLayer* conv2;
    EffectiveSELayer* attn;
    ConvBNLayer* conv3;
    Concat* concat;

    int ch_in;
    int ch_out;
    int n;
    int stride;
    char* act_name;
    bool use_attn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class CSPResNet : public Layer
{
public:
    CSPResNet(std::vector<int>* layers, std::vector<int>* channels, char* act_name="swish", std::vector<int>* return_idx=nullptr, bool depth_wise=false, bool use_large_stem=false, float width_mult=1.f, float depth_mult=1.f, int freeze_at=-1);
    ~CSPResNet();

    Sequential* stem;
    LayerList* stages;
    ConvBNLayer* conv_down;
    ConvBNLayer* conv1;
    ConvBNLayer* conv2;
    EffectiveSELayer* attn;
    ConvBNLayer* conv3;

    std::vector<int>* layers;
    std::vector<int>* channels;
    char* act_name;
    std::vector<int>* return_idx;
    bool depth_wise;
    bool use_large_stem;
    float width_mult;
    float depth_mult;
    int freeze_at;

    virtual std::vector<Tensor*>* create_tensors(Tensor* input, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(Tensor* input, char miemie2013);
private:
};


}  // namespace miemiedet

#endif // __MIEMIEDET_CSPRESNET_H__
