#ifndef __MIEMIEDET_CUSTOM_PAN_H__
#define __MIEMIEDET_CUSTOM_PAN_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class ConvBNLayer;

class SPP : public Layer
{
public:
    SPP(int ch_in, int ch_out, int k, char* act_name="swish");
    ~SPP();

    ConvBNLayer* conv;
    LayerList* maxpools;
    Concat* concat;

    int ch_in;
    int ch_out;
    int k;
    char* act_name;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class CSPStage : public Layer
{
public:
    CSPStage(int ch_in, int ch_out, int n, char* act_name="swish", bool spp=false);
    ~CSPStage();

    Sequential* convs;
    ConvBNLayer* conv1;
    ConvBNLayer* conv2;
    ConvBNLayer* conv3;
    Concat* concat;

    int ch_in;
    int ch_out;
    int n;
    char* act_name;
    bool spp;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class CustomCSPPAN : public Layer
{
public:
    CustomCSPPAN(std::vector<int>* in_channels, std::vector<int>* out_channels, char* act_name="leakyrelu", int stage_num=1, int block_num=3, bool drop_block=false, int block_size=3, float keep_prob=0.9f, bool spp=false, float width_mult=1.f, float depth_mult=1.f);
    ~CustomCSPPAN();

    std::vector<std::vector<Layer*>*>* fpn_stages;
    std::vector<Layer*>* fpn_routes;
    std::vector<std::vector<Layer*>*>* pan_stages;
    std::vector<Layer*>* pan_routes;
    Concat* concat;
    Interp* interp;

    std::vector<int>* in_channels;
    std::vector<int>* out_channels;
    char* act_name;
    int stage_num;
    int block_num;
    bool drop_block;
    int block_size;
    float keep_prob;
    bool spp;
    float width_mult;
    float depth_mult;
    int num_blocks;

    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_CUSTOM_PAN_H__
