#ifndef __MIEMIEDET_PPYOLOE_HEAD_H__
#define __MIEMIEDET_PPYOLOE_HEAD_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class ConvBNLayer;

class ESEAttn : public Layer
{
public:
    ESEAttn(int feat_channels, char* act_name="swish");
    ~ESEAttn();

    Conv2d* fc;
    ConvBNLayer* conv;

    int feat_channels;
    char* act_name;

    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2);
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2);
private:
};

class PPYOLOEHead : public Layer
{
public:
    PPYOLOEHead(std::vector<int>* in_channels, int num_classes=80, char* act_name="swish", std::vector<float>* fpn_strides=nullptr, float grid_cell_scale=5.f, float grid_cell_offset=0.5f, int reg_max=16, int static_assigner_epoch=4, bool use_varifocal_loss=true);
    ~PPYOLOEHead();

    LayerList* stem_cls;
    LayerList* stem_reg;
    LayerList* pred_cls;
    LayerList* pred_reg;
    LayerList* global_avgpools;
    Conv2d* proj_conv;
    Concat* concat1;
    Concat* concat2;

    std::vector<int>* in_channels;
    int num_classes;
    char* act_name;
    std::vector<float>* fpn_strides;
    float grid_cell_scale;
    float grid_cell_offset;
    int reg_max;
    int static_assigner_epoch;
    bool use_varifocal_loss;

    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_PPYOLOE_HEAD_H__
