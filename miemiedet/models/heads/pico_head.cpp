#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "pico_head.h"

namespace miemiedet {

PicoHeadConvNormLayer::PicoHeadConvNormLayer(int ch_in, int ch_out, int filter_size, int stride, int groups, char* norm_type, float norm_decay, int norm_groups, bool use_dcn, bool bias_on)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;
    this->norm_type = norm_type;
    this->use_dcn = use_dcn;

    if (use_dcn)
    {
        printf("DCN not implemented!\n");
        exit(1);
    }
    else
    {
        if (!cfg->fuse_conv_bn)
        {
            conv = new SNT Conv2d(ch_in, ch_out, filter_size, stride, (filter_size - 1) / 2, 1, groups, bias_on);
        }
        else
        {
            conv = new SNT Conv2d(ch_in, ch_out, filter_size, stride, (filter_size - 1) / 2, 1, groups, true);
        }
        register_sublayer("conv", conv);
    }

    norm = nullptr;
    if (strcmp(norm_type, "bn") == 0 || strcmp(norm_type, "sync_bn") == 0)
    {
        if (!cfg->fuse_conv_bn)
        {
            printf("PicoHeadConvNormLayer BN not implemented!\n");
            exit(1);
        }
        else
        {
            norm = nullptr;
        }
    }
    else if (strcmp(norm_type, "gn") == 0)
    {
        printf("PicoHeadConvNormLayer norm_type \'%s\' not implemented!\n", norm_type);
        exit(1);
    }
    else
    {
        printf("PicoHeadConvNormLayer norm_type \'%s\' not implemented!\n", norm_type);
        exit(1);
    }
}

PicoHeadConvNormLayer::~PicoHeadConvNormLayer()
{
    delete conv;
    if (norm != nullptr)
    {
        delete norm;
    }
}

Tensor* PicoHeadConvNormLayer::create_tensors(Tensor* input)
{
    Tensor* out;
    out = conv->create_tensors(input);
    if (norm != nullptr)
    {
        out = norm->create_tensors(out);
    }
    return out;
}

Tensor* PicoHeadConvNormLayer::feed_forward(Tensor* input)
{
    Tensor* out;
    out = conv->feed_forward(input);
    if (norm != nullptr)
    {
        out = norm->feed_forward(out);
    }
    return out;
}

PicoSE::PicoSE(int feat_channels)
{
    this->feat_channels = feat_channels;

    fc = new SNT Conv2d(feat_channels, feat_channels, 1, 1, 0, 1, 1, true);
    register_sublayer("fc", fc);

    this->conv = new SNT PicoHeadConvNormLayer(feat_channels, feat_channels, 1, 1);
    register_sublayer("conv", this->conv);
}

PicoSE::~PicoSE()
{
    delete fc;
    delete conv;
}

Tensor* PicoSE::create_tensors(Tensor* feat, Tensor* avg_feat)
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

Tensor* PicoSE::feed_forward(Tensor* feat, Tensor* avg_feat)
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

PicoFeat::PicoFeat(int feat_in, int feat_out, int num_fpn_stride, int num_convs, char* norm_type, bool share_cls_reg, char* act_name, bool use_se)
{
    this->share_cls_reg = share_cls_reg;
    this->use_se = use_se;
    this->num_convs = num_convs;

    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act = nullptr;
        }
        else
        {
            this->act = new SNT Activation(act_name, 0.01f);
            register_sublayer("act", this->act);
        }
    }
    else
    {
        this->act = nullptr;
    }

    cls_conv_dw0 = new LayerList();
    cls_conv_pw0 = new LayerList();
    cls_conv_dw1 = new LayerList();
    cls_conv_pw1 = new LayerList();
    cls_conv_dw2 = new LayerList();
    cls_conv_pw2 = new LayerList();
    cls_conv_dw3 = new LayerList();
    cls_conv_pw3 = new LayerList();
    if (use_se)
    {
        se = new LayerList();
    }
    else
    {
        se = nullptr;
    }
    for (int stage_idx = 0; stage_idx < num_fpn_stride; stage_idx++)
    {
        for (int i = 0; i < num_convs; i++)
        {
            int in_c = i == 0 ? feat_in : feat_out;
            PicoHeadConvNormLayer* dw = new SNT PicoHeadConvNormLayer(in_c, feat_out, 5, 1, feat_out, norm_type, 0.f, 32, false, false);
            PicoHeadConvNormLayer* pw = new SNT PicoHeadConvNormLayer(in_c, feat_out, 1, 1, 1, norm_type, 0.f, 32, false, false);

            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* layer1_name = new char[64];
            sprintf(layer1_name, "%d", i);  // xxx
            char* layer2_name = new char[64];
            sprintf(layer2_name, "%d", i);  // xxx
            if (stage_idx == 0)
            {
                cls_conv_dw0->add_sublayer(layer1_name, dw);
                cls_conv_pw0->add_sublayer(layer2_name, pw);
            }
            else if (stage_idx == 1)
            {
                cls_conv_dw1->add_sublayer(layer1_name, dw);
                cls_conv_pw1->add_sublayer(layer2_name, pw);
            }
            else if (stage_idx == 2)
            {
                cls_conv_dw2->add_sublayer(layer1_name, dw);
                cls_conv_pw2->add_sublayer(layer2_name, pw);
            }
            else if (stage_idx == 3)
            {
                cls_conv_dw3->add_sublayer(layer1_name, dw);
                cls_conv_pw3->add_sublayer(layer2_name, pw);
            }

            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
        if (use_se)
        {
            PicoSE* layer = new SNT PicoSE(feat_out);

            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* layer_name = new char[64];
            sprintf(layer_name, "%d", stage_idx);  // xxx

            se->add_sublayer(layer_name, layer);
        }
    }
    register_sublayer("cls_conv_dw0", cls_conv_dw0);
    register_sublayer("cls_conv_pw0", cls_conv_pw0);
    register_sublayer("cls_conv_dw1", cls_conv_dw1);
    register_sublayer("cls_conv_pw1", cls_conv_pw1);
    register_sublayer("cls_conv_dw2", cls_conv_dw2);
    register_sublayer("cls_conv_pw2", cls_conv_pw2);
    register_sublayer("cls_conv_dw3", cls_conv_dw3);
    register_sublayer("cls_conv_pw3", cls_conv_pw3);
    register_sublayer("se", se);

    miemienet::Config* cfg = miemienet::Config::getInstance();
    if (cfg->image_data_format == NCHW)
    {
        global_avgpool = new SNT Reduce(MMSHAPE2D(2, 3), true, RED_MEAN);
    }
    else if (cfg->image_data_format == NHWC)
    {
        global_avgpool = new SNT Reduce(MMSHAPE2D(1, 2), true, RED_MEAN);
    }
    register_sublayer("global_avgpool", global_avgpool);
}

PicoFeat::~PicoFeat()
{
    delete cls_conv_dw0;
    delete cls_conv_pw0;
    delete cls_conv_dw1;
    delete cls_conv_pw1;
    delete cls_conv_dw2;
    delete cls_conv_pw2;
    delete cls_conv_dw3;
    delete cls_conv_pw3;
    delete se;
    if (act)
    {
        delete act;
    }
    delete global_avgpool;
}

std::vector<Tensor*>* PicoFeat::create_tensors(Tensor* fpn_feat, int stage_idx)
{
    Tensor* cls_feat = fpn_feat;
    Tensor* reg_feat = fpn_feat;

    if (stage_idx == 0)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw0->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw0->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 1)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw1->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw1->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 2)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw2->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw2->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 3)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw3->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw3->at(i)->create_tensors(cls_feat);
            cls_feat = act->create_tensors(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }

    if (use_se)
    {
        Tensor* avg_feat = global_avgpool->create_tensors(cls_feat);
        Tensor* se_feat = se->at(stage_idx)->create_tensors(cls_feat, avg_feat);
        se_feat = act->create_tensors(se_feat);

        cls_feat->referenceCount++;
        output_tensors->push_back(cls_feat);
        se_feat->referenceCount++;
        output_tensors->push_back(se_feat);
        return output_tensors;
    }
    cls_feat->referenceCount++;
    output_tensors->push_back(cls_feat);
    reg_feat->referenceCount++;
    output_tensors->push_back(reg_feat);
    return output_tensors;
}

std::vector<Tensor*>* PicoFeat::feed_forward(Tensor* fpn_feat, int stage_idx)
{
    Tensor* cls_feat = fpn_feat;
    Tensor* reg_feat = fpn_feat;

    if (stage_idx == 0)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw0->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw0->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 1)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw1->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw1->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 2)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw2->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw2->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }
    else if (stage_idx == 3)
    {
        for (int i = 0; i < num_convs; i++)
        {
            cls_feat = cls_conv_dw3->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }

            cls_feat = cls_conv_pw3->at(i)->feed_forward(cls_feat);
            cls_feat = act->feed_forward(cls_feat);
            reg_feat = cls_feat;
            if (!share_cls_reg)
            {
                printf("PicoFeat !share_cls_reg not implemented!\n");
                exit(1);
            }
        }
    }

    if (use_se)
    {
        Tensor* avg_feat = global_avgpool->feed_forward(cls_feat);
        Tensor* se_feat = se->at(stage_idx)->feed_forward(cls_feat, avg_feat);
        se_feat = act->feed_forward(se_feat);

        cls_feat->referenceCount++;
        output_tensors->push_back(cls_feat);
        se_feat->referenceCount++;
        output_tensors->push_back(se_feat);
        return output_tensors;
    }
    cls_feat->referenceCount++;
    output_tensors->push_back(cls_feat);
    reg_feat->referenceCount++;
    output_tensors->push_back(reg_feat);
    return output_tensors;
}

PicoHeadV2::PicoHeadV2(PicoFeat* conv_feat, int num_classes, std::vector<float>* fpn_stride, bool use_align_head, int reg_max, int feat_in_chan, float cell_offset, char* act_name, float grid_cell_scale)
{
    this->num_classes = num_classes;
    this->fpn_stride = fpn_stride;
    this->use_align_head = use_align_head;
    this->reg_max = reg_max;
    this->feat_in_chan = feat_in_chan;
    this->cell_offset = cell_offset;
    this->act_name = act_name;
    this->grid_cell_scale = grid_cell_scale;
    this->cls_out_channels = num_classes;

    this->conv_feat = conv_feat;
    register_sublayer("conv_feat", conv_feat);

    cls_align = new LayerList();
    for (int i = 0; i < fpn_stride->size(); i++)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name = new char[64];
        sprintf(layer_name, "head_cls%d", i);  // xxx
        Conv2d* head_cls = new SNT Conv2d(feat_in_chan, cls_out_channels, 1, 1, 0, 1, 1, true);

        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name2 = new char[64];
        sprintf(layer_name2, "head_reg%d", i);  // xxx
        Conv2d* head_reg = new SNT Conv2d(feat_in_chan, 4 * (reg_max + 1), 1, 1, 0, 1, 1, true);

        if (i == 0)
        {
            head_cls0 = head_cls;
            register_sublayer("head_cls0", head_cls0);
            head_reg0 = head_reg;
            register_sublayer("head_reg0", head_reg0);
        }
        else if (i == 1)
        {
            head_cls1 = head_cls;
            register_sublayer("head_cls1", head_cls1);
            head_reg1 = head_reg;
            register_sublayer("head_reg1", head_reg1);
        }
        else if (i == 2)
        {
            head_cls2 = head_cls;
            register_sublayer("head_cls2", head_cls2);
            head_reg2 = head_reg;
            register_sublayer("head_reg2", head_reg2);
        }
        else if (i == 3)
        {
            head_cls3 = head_cls;
            register_sublayer("head_cls3", head_cls3);
            head_reg3 = head_reg;
            register_sublayer("head_reg3", head_reg3);
        }

        if (use_align_head)
        {
            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* dp_layer_name = new char[64];
            sprintf(dp_layer_name, "%d", i);  // xxx
            DPModule* dp_layer = new SNT DPModule(feat_in_chan, 1, 5, 1, act_name, false);
            cls_align->add_sublayer(dp_layer_name, dp_layer);
        }
    }
    register_sublayer("cls_align", cls_align);

    distribution_project = new SNT Conv2d(reg_max + 1, 1, 1, 1, 0, 1, 1, false);
    for (int k = 0; k < reg_max + 1; k++)
    {
        *(distribution_project->weight->data_fp32 + k) = (float)k;
    }
    register_sublayer("distribution_project", distribution_project);

    concat1 = new SNT Concat(1);
    register_sublayer("concat1", concat1);
    concat2 = new SNT Concat(1);
    register_sublayer("concat2", concat2);
}

PicoHeadV2::~PicoHeadV2()
{
    delete conv_feat;
    delete head_cls0;
    delete head_reg0;
    delete head_cls1;
    delete head_reg1;
    delete head_cls2;
    delete head_reg2;
    delete head_cls3;
    delete head_reg3;
    delete cls_align;

    delete distribution_project;
    delete concat1;
    delete concat2;
}

std::vector<Tensor*>* PicoHeadV2::create_tensors(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* cls_score_list = temp_tensors;
    std::vector<Tensor*>* reg_dist_list = temp2_tensors;
    int gap_id = 0;
    miemienet::Config* cfg = miemienet::Config::getInstance();
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* fpn_feat = inputs->at(i);
        float stride = fpn_stride->at(i);
        const int b = fpn_feat->shape->at(0);
        const int h = fpn_feat->shape->at(1);
        const int w = fpn_feat->shape->at(2);

        std::vector<Tensor*>* conv_cls_feat_se_feat = conv_feat->create_tensors(fpn_feat, i);
        Tensor* conv_cls_feat = conv_cls_feat_se_feat->at(i * 2);
        Tensor* se_feat = conv_cls_feat_se_feat->at(i * 2 + 1);

        Tensor* cls_logit = nullptr;
        Tensor* reg_pred = nullptr;
        if (i == 0)
        {
            cls_logit = head_cls0->create_tensors(se_feat);
            reg_pred = head_reg0->create_tensors(se_feat);
        }
        else if (i == 1)
        {
            cls_logit = head_cls1->create_tensors(se_feat);
            reg_pred = head_reg1->create_tensors(se_feat);
        }
        else if (i == 2)
        {
            cls_logit = head_cls2->create_tensors(se_feat);
            reg_pred = head_reg2->create_tensors(se_feat);
        }
        else if (i == 3)
        {
            cls_logit = head_cls3->create_tensors(se_feat);
            reg_pred = head_reg3->create_tensors(se_feat);
        }

        Tensor* cls_score = nullptr;
        if (use_align_head)
        {
            cls_score = cls_align->at(i)->create_tensors(conv_cls_feat);
            cls_score = cls_logit;
        }
        else
        {
            cls_score = cls_logit;
        }

        const int l = h * w;

        reg_pred->reshape(MMSHAPE4D(b, -1, 4, reg_max + 1));
//        miemienet::functional::softmax(reg_pred, reg_pred, -1);
        Tensor* bbox_pred = distribution_project->create_tensors(reg_pred);

        bbox_pred->reshape(MMSHAPE3D(b, -1, 4));
        cls_score->reshape(MMSHAPE3D(b, -1, num_classes));

        cls_score->referenceCount++;
        bbox_pred->referenceCount++;
        cls_score_list->push_back(cls_score);
        reg_dist_list->push_back(bbox_pred);
    }

    Tensor* scores;
    Tensor* regs;
    if (cls_score_list->size() == 3)
    {
        scores = concat1->create_tensors(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2));
        regs = concat2->create_tensors(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2));
    }
    else if (cls_score_list->size() == 4)
    {
        scores = concat1->create_tensors(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2), cls_score_list->at(3));
        regs = concat2->create_tensors(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2), reg_dist_list->at(3));
    }

    std::vector<Tensor*>* outs = output_tensors;
    scores->referenceCount++;
    regs->referenceCount++;
    outs->push_back(scores);
    outs->push_back(regs);
    return outs;
}

std::vector<Tensor*>* PicoHeadV2::feed_forward(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* cls_score_list = temp_tensors;
    std::vector<Tensor*>* reg_dist_list = temp2_tensors;
    int gap_id = 0;
    miemienet::Config* cfg = miemienet::Config::getInstance();
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* fpn_feat = inputs->at(i);
        float stride = fpn_stride->at(i);
        const int b = fpn_feat->shape->at(0);
        const int h = fpn_feat->shape->at(1);
        const int w = fpn_feat->shape->at(2);

        std::vector<Tensor*>* conv_cls_feat_se_feat = conv_feat->feed_forward(fpn_feat, i);
        Tensor* conv_cls_feat = conv_cls_feat_se_feat->at(i * 2);
        Tensor* se_feat = conv_cls_feat_se_feat->at(i * 2 + 1);

        Tensor* cls_logit = nullptr;
        Tensor* reg_pred = nullptr;
        if (i == 0)
        {
            cls_logit = head_cls0->feed_forward(se_feat);
            reg_pred = head_reg0->feed_forward(se_feat);
        }
        else if (i == 1)
        {
            cls_logit = head_cls1->feed_forward(se_feat);
            reg_pred = head_reg1->feed_forward(se_feat);
        }
        else if (i == 2)
        {
            cls_logit = head_cls2->feed_forward(se_feat);
            reg_pred = head_reg2->feed_forward(se_feat);
        }
        else if (i == 3)
        {
            cls_logit = head_cls3->feed_forward(se_feat);
            reg_pred = head_reg3->feed_forward(se_feat);
        }

        Tensor* cls_score = nullptr;
        if (use_align_head)
        {
            cls_score = cls_align->at(i)->feed_forward(conv_cls_feat);
            miemienet::functional::activation(cls_score, cls_score, "sigmoid", 0.f);
            miemienet::functional::activation(cls_logit, cls_logit, "sigmoid", 0.f);
            miemienet::functional::elementwise(cls_logit, cls_score, cls_logit, ELE_MUL);
            miemienet::functional::activation(cls_logit, cls_logit, "sqrt", 1e-9);
            cls_score = cls_logit;
        }
        else
        {
            miemienet::functional::activation(cls_logit, cls_logit, "sigmoid", 0.f);
            cls_score = cls_logit;
        }

        const int l = h * w;

        reg_pred->reshape(MMSHAPE4D(b, -1, 4, reg_max + 1));
        miemienet::functional::softmax(reg_pred, reg_pred, -1);
        Tensor* bbox_pred = distribution_project->feed_forward(reg_pred);

        bbox_pred->reshape(MMSHAPE3D(b, -1, 4));
        cls_score->reshape(MMSHAPE3D(b, -1, num_classes));
    }

    Tensor* scores;
    Tensor* regs;
    if (cls_score_list->size() == 3)
    {
        scores = concat1->feed_forward(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2));
        regs = concat2->feed_forward(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2));
    }
    else if (cls_score_list->size() == 4)
    {
        scores = concat1->feed_forward(cls_score_list->at(0), cls_score_list->at(1), cls_score_list->at(2), cls_score_list->at(3));
        regs = concat2->feed_forward(reg_dist_list->at(0), reg_dist_list->at(1), reg_dist_list->at(2), reg_dist_list->at(3));
    }

    std::vector<Tensor*>* outs = output_tensors;
    return outs;
}

}  // namespace miemiedet
