#ifndef __MIEMIEDET_PICO_HEAD_H__
#define __MIEMIEDET_PICO_HEAD_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class PicoHeadConvNormLayer : public Layer
{
public:
    PicoHeadConvNormLayer(int ch_in, int ch_out, int filter_size, int stride, int groups=1, char* norm_type="bn", float norm_decay=0.f, int norm_groups=32, bool use_dcn=false, bool bias_on=false);
    ~PicoHeadConvNormLayer();

    Conv2d* conv;
    Layer* norm;

    char* norm_type;
    bool use_dcn;

    bool fuse_conv_bn;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

class PicoSE : public Layer
{
public:
    PicoSE(int feat_channels);
    ~PicoSE();

    Conv2d* fc;
    PicoHeadConvNormLayer* conv;

    int feat_channels;
    char* act_name;

    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2);
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2);
private:
};

class PicoFeat : public Layer
{
public:
    PicoFeat(int feat_in=256, int feat_out=96, int num_fpn_stride=3, int num_convs=2, char* norm_type="bn", bool share_cls_reg=false, char* act_name="hardswish", bool use_se=false);
    ~PicoFeat();

    LayerList* cls_conv_dw0;
    LayerList* cls_conv_pw0;
    LayerList* cls_conv_dw1;
    LayerList* cls_conv_pw1;
    LayerList* cls_conv_dw2;
    LayerList* cls_conv_pw2;
    LayerList* cls_conv_dw3;
    LayerList* cls_conv_pw3;
    LayerList* se;
    Activation* act;
    Reduce* global_avgpool;

    bool share_cls_reg;
    bool use_se;
    int num_convs;

    std::vector<Tensor*>* create_tensors(Tensor* fpn_feat, int stage_idx);
    std::vector<Tensor*>* feed_forward(Tensor* fpn_feat, int stage_idx);
private:
};

class PicoHeadV2 : public Layer
{
public:
    PicoHeadV2(PicoFeat* conv_feat, int num_classes=80, std::vector<float>* fpn_stride=nullptr, bool use_align_head=true, int reg_max=16, int feat_in_chan=96, float cell_offset=0.f, char* act_name="hardswish", float grid_cell_scale=5.f);
    ~PicoHeadV2();

    PicoFeat* conv_feat;
    Conv2d* head_cls0;
    Conv2d* head_reg0;
    Conv2d* head_cls1;
    Conv2d* head_reg1;
    Conv2d* head_cls2;
    Conv2d* head_reg2;
    Conv2d* head_cls3;
    Conv2d* head_reg3;
    LayerList* cls_align;
    Conv2d* distribution_project;
    Concat* concat1;
    Concat* concat2;

    int num_classes;
    std::vector<float>* fpn_stride;
    bool use_align_head;
    int reg_max;
    int feat_in_chan;
    float cell_offset;
    char* act_name;
    float grid_cell_scale;
    int cls_out_channels;

    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_PICO_HEAD_H__
