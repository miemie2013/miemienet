#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "custom_pan.h"

namespace miemiedet {

CSPPANConvBNLayer::CSPPANConvBNLayer(int in_channel, int out_channel, int kernel_size, int stride, int groups, char* act_name)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;

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

    if (!cfg->fuse_conv_bn)
    {
        conv = new SNT Conv2d(in_channel, out_channel, kernel_size, stride, (kernel_size - 1) / 2, 1, groups, false);
    }
    else
    {
        conv = new SNT Conv2d(in_channel, out_channel, kernel_size, stride, (kernel_size - 1) / 2, 1, groups, true);
    }
    register_sublayer("conv", conv);
    // 初始化权重
    // xxx

    if (!cfg->fuse_conv_bn)
    {
//        float momentum = 0.1f;
//        bn = new SNT BatchNorm2d(ch_out, 1e-5, momentum, true, true);
//        register_sublayer("bn", bn);
        bn = nullptr;
    }
    else
    {
        bn = nullptr;
    }
}

CSPPANConvBNLayer::~CSPPANConvBNLayer()
{
    delete conv;
    if (!fuse_conv_bn)
    {
        delete bn;
    }
    if (act)
    {
        delete act;
    }
}

Tensor* CSPPANConvBNLayer::create_tensors(Tensor* input)
{
    Tensor* out;
    out = conv->create_tensors(input);
    if (!fuse_conv_bn)
    {
        out = bn->create_tensors(out);
    }
    if (act)
    {
        out = act->create_tensors(out);
    }
    return out;
}

Tensor* CSPPANConvBNLayer::feed_forward(Tensor* input)
{
    Tensor* out;
    out = conv->feed_forward(input);
    if (!fuse_conv_bn)
    {
        out = bn->feed_forward(out);
    }
    if (act)
    {
        out = act->feed_forward(out);
    }
    return out;
}

Channel_T::Channel_T(std::vector<int>* in_channels, int out_channels, char* act_name)
{
    this->convs = new LayerList();
    for (int i = 0; i < in_channels->size(); i++)
    {
        int ch_in = in_channels->at(i);
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* conv_name = new char[64];
        sprintf(conv_name, "%d", i);  // xxx
        CSPPANConvBNLayer* layer = new SNT CSPPANConvBNLayer(ch_in, out_channels, 1, 1, 1, act_name);
        convs->add_sublayer(conv_name, layer);
    }
    register_sublayer("convs", convs);
}

Channel_T::~Channel_T()
{
    delete convs;
}

std::vector<Tensor*>* Channel_T::create_tensors(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* outs = output_tensors;
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* xi = inputs->at(i);
        Layer* layer = convs->at(i);
        xi = layer->create_tensors(xi);

        xi->referenceCount++;
        outs->push_back(xi);
    }
    return outs;
}

std::vector<Tensor*>* Channel_T::feed_forward(std::vector<Tensor*>* inputs, char miemie2013)
{
    std::vector<Tensor*>* outs = output_tensors;
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* xi = inputs->at(i);
        Layer* layer = convs->at(i);
        xi = layer->feed_forward(xi);
    }
    return outs;
}

DPModule::DPModule(int in_channel, int out_channel, int kernel_size, int stride, char* act_name, bool use_act_in_out)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;
    this->use_act_in_out = use_act_in_out;

    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act1 = nullptr;
            this->act2 = nullptr;
        }
        else
        {
            this->act1 = new SNT Activation(act_name, 0.01f);
            register_sublayer("act1", this->act1);
            this->act2 = new SNT Activation(act_name, 0.01f);
            register_sublayer("act2", this->act2);
        }
    }
    else
    {
        this->act1 = nullptr;
        this->act2 = nullptr;
    }

    if (!cfg->fuse_conv_bn)
    {
        dwconv = new SNT Conv2d(in_channel, out_channel, kernel_size, stride, (kernel_size - 1) / 2, 1, out_channel, false);
        pwconv = new SNT Conv2d(out_channel, out_channel, 1, 1, 0, 1, 1, false);
    }
    else
    {
        dwconv = new SNT Conv2d(in_channel, out_channel, kernel_size, stride, (kernel_size - 1) / 2, 1, out_channel, true);
        pwconv = new SNT Conv2d(out_channel, out_channel, 1, 1, 0, 1, 1, true);
    }
    register_sublayer("dwconv", dwconv);
    register_sublayer("pwconv", pwconv);
    // 初始化权重
    // xxx

    if (!cfg->fuse_conv_bn)
    {
//        float momentum = 0.1f;
//        bn = new SNT BatchNorm2d(ch_out, 1e-5, momentum, true, true);
//        register_sublayer("bn", bn);
        bn1 = nullptr;
        bn2 = nullptr;
    }
    else
    {
        bn1 = nullptr;
        bn2 = nullptr;
    }
}

DPModule::~DPModule()
{
    delete dwconv;
    delete pwconv;
    if (!fuse_conv_bn)
    {
        delete bn1;
        delete bn2;
    }
    delete act1;
    delete act2;
}

Tensor* DPModule::create_tensors(Tensor* input)
{
    Tensor* x = dwconv->create_tensors(input);
    if (!fuse_conv_bn)
    {
        x = bn1->create_tensors(x);
    }
    x = act1->create_tensors(x);

    x = pwconv->create_tensors(x);
    if (!fuse_conv_bn)
    {
        x = bn2->create_tensors(x);
    }
    if (use_act_in_out)
    {
        x = act2->create_tensors(x);
    }
    return x;
}

Tensor* DPModule::feed_forward(Tensor* input)
{
    Tensor* x = dwconv->feed_forward(input);
    if (!fuse_conv_bn)
    {
        x = bn1->feed_forward(x);
    }
    x = act1->feed_forward(x);

    x = pwconv->feed_forward(x);
    if (!fuse_conv_bn)
    {
        x = bn2->feed_forward(x);
    }
    if (use_act_in_out)
    {
        x = act2->feed_forward(x);
    }
    return x;
}

LCPAN::LCPAN(std::vector<int>* in_channels, int out_channels, int kernel_size, int num_features, bool use_depthwise, char* act_name, std::vector<float>* spatial_scales)
{
    this->in_channels = out_channels;
    this->num_features = num_features;
    this->spatial_scales = spatial_scales;

    conv_t = new SNT Channel_T(in_channels, out_channels, act_name);
    register_sublayer("conv_t", conv_t);

    if (num_features == 4)
    {
        if (use_depthwise)
        {
            first_top_conv = new SNT DPModule(out_channels, out_channels, kernel_size, 2, act_name, true);
            second_top_conv = new SNT DPModule(out_channels, out_channels, kernel_size, 2, act_name, true);
        }
        else
        {
            first_top_conv = new SNT CSPPANConvBNLayer(out_channels, out_channels, kernel_size, 2, 1, act_name);
            second_top_conv = new SNT CSPPANConvBNLayer(out_channels, out_channels, kernel_size, 2, 1, act_name);
        }
        register_sublayer("first_top_conv", first_top_conv);
        register_sublayer("second_top_conv", second_top_conv);
    }

    upsample = new SNT Interp(0, 0, 2.f, 2.f);
    register_sublayer("upsample", upsample);

    this->top_down_blocks = new LayerList();
    int i = 0;
    for (int idx = spatial_scales->size() - 1; idx >= 0; idx--)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name = new char[64];
        sprintf(layer_name, "%d", i++);  // xxx


        Sequential* sequ = new Sequential();
        Layer* layer1 = new SNT DepthwiseSeparable(out_channels * 2, out_channels * 2, 1, kernel_size, false);
        Layer* layer2 = new SNT DepthwiseSeparable(out_channels * 2, out_channels, 1, kernel_size, false);
        sequ->add_sublayer("0", layer1);
        sequ->add_sublayer("1", layer2);
//        register_sublayer("sequ", sequ);   // LayerList成员，里面装Sequential对象时，Sequential对象不需要register_sublayer()，LayerList成员register_sublayer()即可。
        // LayerList成员，里面装其他Layer子类对象时，其他Layer子类对象不需要register_sublayer()，LayerList成员register_sublayer()即可。PPYOLOEHead的stem_reg就是一个例子。

        top_down_blocks->add_sublayer(layer_name, sequ);
    }
    register_sublayer("top_down_blocks", top_down_blocks);


    this->downsamples = new LayerList();
    this->bottom_up_blocks = new LayerList();
    for (int idx = 0; idx < spatial_scales->size() - 1; idx++)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name = new char[64];
        sprintf(layer_name, "%d", idx);  // xxx


        Sequential* sequ = new Sequential();
        Layer* layer1 = new SNT DepthwiseSeparable(out_channels * 2, out_channels * 2, 1, kernel_size, false);
        Layer* layer2 = new SNT DepthwiseSeparable(out_channels * 2, out_channels, 1, kernel_size, false);
        sequ->add_sublayer("0", layer1);
        sequ->add_sublayer("1", layer2);
//        register_sublayer("sequ", sequ);   // LayerList成员，里面装Sequential对象时，Sequential对象不需要register_sublayer()，LayerList成员register_sublayer()即可。
        // LayerList成员，里面装其他Layer子类对象时，其他Layer子类对象不需要register_sublayer()，LayerList成员register_sublayer()即可。PPYOLOEHead的stem_reg就是一个例子。

        bottom_up_blocks->add_sublayer(layer_name, sequ);


        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* layer_name2 = new char[64];
        sprintf(layer_name2, "%d", idx);  // xxx
        Layer* layer3 = nullptr;
        if (use_depthwise)
        {
            layer3 = new SNT DPModule(out_channels, out_channels, kernel_size, 2, act_name, true);
        }
        else
        {
            layer3 = new SNT CSPPANConvBNLayer(out_channels, out_channels, kernel_size, 2, 1, act_name);
        }
        downsamples->add_sublayer(layer_name2, layer3);
    }
    register_sublayer("downsamples", downsamples);
    register_sublayer("bottom_up_blocks", bottom_up_blocks);

    miemienet::Config* cfg = miemienet::Config::getInstance();
    if (cfg->image_data_format == NCHW)
    {
        concat = new SNT Concat(1);
    }
    else if (cfg->image_data_format == NHWC)
    {
        concat = new SNT Concat(-1);
    }
    register_sublayer("concat", concat);
}

LCPAN::~LCPAN()
{
    delete conv_t;
    if (num_features == 4)
    {
        delete first_top_conv;
        delete second_top_conv;
    }
    delete upsample;
    delete top_down_blocks;
    delete downsamples;
    delete bottom_up_blocks;
    delete concat;
}

std::vector<Tensor*>* LCPAN::create_tensors(std::vector<Tensor*>* inputs, char miemie2013)
{
    inputs = conv_t->create_tensors(inputs, miemie2013);

    // top-down path
    std::vector<Tensor*>* inner_outs = temp_tensors;
    inputs->at(inputs->size() - 1)->referenceCount++;
    inner_outs->push_back(inputs->at(inputs->size() - 1));
    int i = 0;
    for (int idx = inputs->size() - 1; idx > 0; idx--)
    {
        Tensor* feat_heigh = inner_outs->at(0);
        Tensor* feat_low = inputs->at(idx - 1);

        Tensor* upsample_feat = upsample->create_tensors(feat_heigh);

        Tensor* inner_out = concat->create_tensors(upsample_feat, feat_low);
        inner_out = top_down_blocks->at(i++)->create_tensors(inner_out);

        inner_out->referenceCount++;
        inner_outs->insert(inner_outs->begin(), inner_out);
    }

    // bottom-up path
    std::vector<Tensor*>* outs = output_tensors;
    inner_outs->at(0)->referenceCount++;
    outs->push_back(inner_outs->at(0));
    for (int idx = 0; idx < inputs->size() - 1; idx++)
    {
        Tensor* feat_low = outs->at(outs->size() - 1);
        Tensor* feat_height = inner_outs->at(idx + 1);

        Tensor* downsample_feat = downsamples->at(idx)->create_tensors(feat_low);

        Tensor* out = concat->create_tensors(downsample_feat, feat_height);
        out = bottom_up_blocks->at(idx)->create_tensors(out);

        out->referenceCount++;
        outs->push_back(out);
    }

    if (num_features == 4)
    {
        Tensor* top_features = first_top_conv->create_tensors(inputs->at(inputs->size() - 1));
        Tensor* short_cut = second_top_conv->create_tensors(outs->at(outs->size() - 1));
//        miemienet::functional::elementwise(top_features, short_cut, top_features, ELE_ADD);

        top_features->referenceCount++;
        outs->push_back(top_features);
    }

    return outs;
}

std::vector<Tensor*>* LCPAN::feed_forward(std::vector<Tensor*>* inputs, char miemie2013)
{
    inputs = conv_t->feed_forward(inputs, miemie2013);

    // top-down path
    std::vector<Tensor*>* inner_outs = temp_tensors;
//    inputs->at(inputs->size() - 1)->referenceCount++;
//    inner_outs->push_back(inputs->at(inputs->size() - 1));
    int i = 0;
    for (int idx = inputs->size() - 1; idx > 0; idx--)
    {
        Tensor* feat_heigh = inner_outs->at(idx);   // inner_outs在create_tensors()阶段已经插入过，所以这里索引由0改成idx
        Tensor* feat_low = inputs->at(idx - 1);

        Tensor* upsample_feat = upsample->feed_forward(feat_heigh);

        Tensor* inner_out = concat->feed_forward(upsample_feat, feat_low);
        inner_out = top_down_blocks->at(i++)->feed_forward(inner_out);

//        inner_out->referenceCount++;
//        inner_outs->insert(inner_outs->begin(), inner_out);
    }

    // bottom-up path
    std::vector<Tensor*>* outs = output_tensors;
//    inner_outs->at(0)->referenceCount++;
//    outs->push_back(inner_outs->at(0));
    for (int idx = 0; idx < inputs->size() - 1; idx++)
    {
        Tensor* feat_low = outs->at(idx);   // outs在create_tensors()阶段已经插入过，所以这里索引由 outs->size() - 1 改成idx
        Tensor* feat_height = inner_outs->at(idx + 1);

        Tensor* downsample_feat = downsamples->at(idx)->feed_forward(feat_low);

        Tensor* out = concat->feed_forward(downsample_feat, feat_height);
        out = bottom_up_blocks->at(idx)->feed_forward(out);

//        out->referenceCount++;
//        outs->push_back(out);
    }

    if (num_features == 4)
    {
        Tensor* top_features = first_top_conv->feed_forward(inputs->at(inputs->size() - 1));
        Tensor* short_cut = second_top_conv->feed_forward(outs->at(outs->size() - 2));   // outs在create_tensors()阶段已经插入过，所以这里索引由 outs->size() - 1 改成 outs->size() - 2
        miemienet::functional::elementwise(top_features, short_cut, top_features, ELE_ADD);

//        top_features->referenceCount++;
//        outs->push_back(top_features);
    }

    return outs;
}

}  // namespace miemiedet
