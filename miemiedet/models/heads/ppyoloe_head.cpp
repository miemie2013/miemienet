#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "ppyoloe_head.h"

namespace miemiedet {

ESEAttn::ESEAttn(int feat_channels, char* act_name)
{
    this->feat_channels = feat_channels;
    this->act_name = act_name;

    fc = new SNT Conv2d(feat_channels, feat_channels, 1, 1, 0, 1, 1, true);
    register_sublayer("fc", fc);

    this->conv = new SNT ConvBNLayer(feat_channels, feat_channels, 1, 1, 1, 0, act_name);
    register_sublayer("conv", this->conv);
}

ESEAttn::~ESEAttn()
{
    delete fc;
    delete conv;
}

Tensor* ESEAttn::create_tensors(Tensor* feat, Tensor* avg_feat)
{
    // 看PPYOLOEHead的代码，后面还会再次使用feat和avg_feat，所以不能用inpalce（省内存）的方式求输出。
    // avg_feat肯定不会被修改，因为Conv2d层一定会新建结果张量weight。
    // weight进行sigmoid激活可以就地修改，不新建张量。
    // weight 和 feat逐元素相乘，需要新建1个张量feat_weight保存结果。
    Tensor* weight = fc->create_tensors(avg_feat);

    std::vector<int>* _shape = feat->clone_shape();
    Tensor* feat_weight = new SNT Tensor(_shape, FP32, false, false);
    feat_weight->referenceCount++;
    temp_tensors->push_back(feat_weight);

    Tensor* y = conv->create_tensors(feat_weight);
    return y;
}

Tensor* ESEAttn::feed_forward(Tensor* feat, Tensor* avg_feat)
{
    // 看PPYOLOEHead的代码，后面还会再次使用feat和avg_feat，所以不能用inpalce（省内存）的方式求输出。
    // avg_feat肯定不会被修改，因为Conv2d层一定会新建结果张量weight。
    // weight进行sigmoid激活可以就地修改，不新建张量。
    // weight 和 feat逐元素相乘，需要新建1个张量feat_weight保存结果。
    Tensor* weight = fc->feed_forward(avg_feat);
    miemienet::functional::activation(weight, weight, "sigmoid", 0.f);

    Tensor* feat_weight = temp_tensors->at(0);
    miemienet::functional::elementwise(feat, weight, feat_weight, ELE_MUL);

    Tensor* y = conv->feed_forward(feat_weight);
    return y;
}

PPYOLOEHead::PPYOLOEHead(std::vector<int>* in_channels, int num_classes, char* act_name, std::vector<float>* fpn_strides, float grid_cell_scale, float grid_cell_offset, int reg_max, int static_assigner_epoch, bool use_varifocal_loss)
{
    this->in_channels = in_channels;
    this->num_classes = num_classes;
    this->act_name = act_name;
    this->fpn_strides = fpn_strides;
    this->grid_cell_scale = grid_cell_scale;
    this->grid_cell_offset = grid_cell_offset;
    this->reg_max = reg_max;
    this->static_assigner_epoch = static_assigner_epoch;
    this->use_varifocal_loss = use_varifocal_loss;

    this->stem_cls = new LayerList();
    this->stem_reg = new LayerList();
    for (int i = 0; i < in_channels->size(); i++)
    {
        int in_c = in_channels->at(i);
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name = new char[64];
        sprintf(layer_name, "%d", i);  // xxx
        ESEAttn* layer = new SNT ESEAttn(in_c, act_name);
        stem_cls->add_sublayer(layer_name, layer);

        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name2 = new char[64];
        sprintf(layer_name2, "%d", i);  // xxx
        ESEAttn* layer2 = new SNT ESEAttn(in_c, act_name);
        stem_reg->add_sublayer(layer_name2, layer2);
    }
    register_sublayer("stem_cls", stem_cls);
    register_sublayer("stem_reg", stem_reg);

    this->pred_cls = new LayerList();
    this->pred_reg = new LayerList();
    for (int i = 0; i < in_channels->size(); i++)
    {
        int in_c = in_channels->at(i);
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name = new char[64];
        sprintf(layer_name, "%d", i);  // xxx
        Conv2d* layer = new SNT Conv2d(in_c, num_classes, 3, 1, 1, 1, 1, true);
        pred_cls->add_sublayer(layer_name, layer);

        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name2 = new char[64];
        sprintf(layer_name2, "%d", i);  // xxx
        Conv2d* layer2 = new SNT Conv2d(in_c, 4 * (reg_max + 1), 3, 1, 1, 1, 1, true);
        pred_reg->add_sublayer(layer_name2, layer2);
    }
    register_sublayer("pred_cls", pred_cls);
    register_sublayer("pred_reg", pred_reg);
    this->global_avgpools = new LayerList();

    proj_conv = new SNT Conv2d(reg_max + 1, 1, 1, 1, 0, 1, 1, false);
    for (int k = 0; k < reg_max + 1; k++)
    {
        *(proj_conv->weight->data_fp32 + k) = (float)k;
    }
    register_sublayer("proj_conv", proj_conv);

    concat1 = new SNT Concat(1);
    register_sublayer("concat1", concat1);
    concat2 = new SNT Concat(1);
    register_sublayer("concat2", concat2);
}

PPYOLOEHead::~PPYOLOEHead()
{
    delete stem_cls;
    delete stem_reg;
    delete pred_cls;
    delete pred_reg;
    delete proj_conv;
    delete global_avgpools;
    delete concat1;
    delete concat2;
}

std::vector<Tensor*>* PPYOLOEHead::create_tensors(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* cls_score_list = temp_tensors;
    std::vector<Tensor*>* reg_dist_list = temp2_tensors;
    int gap_id = 0;
    miemienet::Config* cfg = miemienet::Config::getInstance();
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* feat = inputs->at(i);
        Reduce* reduce;
        if (cfg->image_data_format == NCHW)
        {
            reduce = new SNT Reduce(MMSHAPE2D(2, 3), true, RED_MEAN);
        }
        else if (cfg->image_data_format == NHWC)
        {
            reduce = new SNT Reduce(MMSHAPE2D(1, 2), true, RED_MEAN);
        }
        char* layer_name = new char[64];
        sprintf(layer_name, "%d", gap_id++);
        global_avgpools->add_sublayer(layer_name, reduce);
        Tensor* avg_feat = reduce->create_tensors(feat);

        Tensor* cls_score = stem_cls->at(i)->create_tensors(feat, avg_feat);
        cls_score = pred_cls->at(i)->create_tensors(cls_score);

        Tensor* reg_feat = stem_reg->at(i)->create_tensors(feat, avg_feat);
        reg_feat = pred_reg->at(i)->create_tensors(reg_feat);

        const int N = reg_feat->shape->at(0);
        reg_feat->reshape(MMSHAPE4D(N, -1, 4, reg_max + 1));
//        miemienet::functional::softmax(reg_feat, reg_feat, -1);
        Tensor* reg_dist = proj_conv->create_tensors(reg_feat);

        reg_dist->reshape(MMSHAPE3D(N, -1, 4));
        cls_score->reshape(MMSHAPE3D(N, -1, num_classes));

        cls_score->referenceCount++;
        reg_dist->referenceCount++;
        cls_score_list->push_back(cls_score);
        reg_dist_list->push_back(reg_dist);
    }
    register_sublayer("global_avgpools", global_avgpools);

    Tensor* scores;
    Tensor* regs;
    if (cls_score_list->size() == 3)
    {
        scores = concat1->create_tensors(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2));
        regs = concat2->create_tensors(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2));
    }

    std::vector<Tensor*>* outs = output_tensors;
    scores->referenceCount++;
    regs->referenceCount++;
    outs->push_back(scores);
    outs->push_back(regs);
    return outs;
}

std::vector<Tensor*>* PPYOLOEHead::feed_forward(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* cls_score_list = temp_tensors;
    std::vector<Tensor*>* reg_dist_list = temp2_tensors;
    int gap_id = 0;
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* feat = inputs->at(i);
        Layer* reduce = global_avgpools->at(gap_id++);
        Tensor* avg_feat = reduce->feed_forward(feat);

        Tensor* cls_score = stem_cls->at(i)->feed_forward(feat, avg_feat);
        miemienet::functional::elementwise(cls_score, feat, cls_score, ELE_ADD);
        cls_score = pred_cls->at(i)->feed_forward(cls_score);
        miemienet::functional::activation(cls_score, cls_score, "sigmoid", 0.f);

        Tensor* reg_feat = stem_reg->at(i)->feed_forward(feat, avg_feat);
        reg_feat = pred_reg->at(i)->feed_forward(reg_feat);

        const int N = reg_feat->shape->at(0);
        reg_feat->reshape(MMSHAPE4D(N, -1, 4, reg_max + 1));
        miemienet::functional::softmax(reg_feat, reg_feat, -1);

        Tensor* reg_dist = proj_conv->feed_forward(reg_feat);
        reg_dist->reshape(MMSHAPE3D(N, -1, 4));
        cls_score->reshape(MMSHAPE3D(N, -1, num_classes));

//        cls_score_list->push_back(cls_score);
//        reg_dist_list->push_back(reg_dist);
    }

    Tensor* scores;
    Tensor* regs;
    if (cls_score_list->size() == 3)
    {
        scores = concat1->feed_forward(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2));
        regs = concat2->feed_forward(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2));
    }

    std::vector<Tensor*>* outs = output_tensors;
//    outs->push_back(scores);
//    outs->push_back(regs);
    return outs;
}

}  // namespace miemiedet
