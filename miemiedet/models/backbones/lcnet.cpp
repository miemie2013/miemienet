#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "lcnet.h"

namespace miemiedet {

LCNetConvBNLayer::LCNetConvBNLayer(int num_channels, int filter_size, int num_filters, int stride, int groups)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;

    this->act = new SNT Activation("hardswish", 0.f);
    register_sublayer("act", this->act);

    if (!cfg->fuse_conv_bn)
    {
        conv = new SNT Conv2d(num_channels, num_filters, filter_size, stride, (filter_size - 1) / 2, 1, groups, false);
    }
    else
    {
        conv = new SNT Conv2d(num_channels, num_filters, filter_size, stride, (filter_size - 1) / 2, 1, groups, true);
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

LCNetConvBNLayer::~LCNetConvBNLayer()
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

Tensor* LCNetConvBNLayer::create_tensors(Tensor* input)
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

Tensor* LCNetConvBNLayer::feed_forward(Tensor* input)
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

SEModule::SEModule(int channel, int reduction)
{
    this->relu = new SNT Activation("relu", 0.f);
    register_sublayer("relu", this->relu);
    this->hardsigmoid = new SNT Activation("hardsigmoid", 0.f);
    register_sublayer("hardsigmoid", this->hardsigmoid);
    conv1 = new SNT Conv2d(channel, channel / reduction, 1, 1, 0, 1, 1, true);
    conv2 = new SNT Conv2d(channel / reduction, channel, 1, 1, 0, 1, 1, true);
    register_sublayer("conv1", conv1);
    register_sublayer("conv2", conv2);
    avg_pool = new SNT Reduce(MMSHAPE2D(1, 2), true, RED_MEAN);
    register_sublayer("avg_pool", avg_pool);
}

SEModule::~SEModule()
{
    delete conv1;
    delete conv2;
    delete relu;
    delete hardsigmoid;
    delete avg_pool;
}

Tensor* SEModule::create_tensors(Tensor* input)
{
    Tensor* x = avg_pool->create_tensors(input);
    x = conv1->create_tensors(x);
    x = relu->create_tensors(x);
    x = conv2->create_tensors(x);
    x = hardsigmoid->create_tensors(x);

    return input;
}

Tensor* SEModule::feed_forward(Tensor* input)
{
    Tensor* x = avg_pool->feed_forward(input);
    x = conv1->feed_forward(x);
    x = relu->feed_forward(x);
    x = conv2->feed_forward(x);
    x = hardsigmoid->feed_forward(x);

    miemienet::functional::elementwise(input, x, input, ELE_MUL);
    return input;
}

DepthwiseSeparable::DepthwiseSeparable(int num_channels, int num_filters, int stride, int dw_size, bool use_se)
{
    this->use_se = use_se;

    this->dw_conv = new SNT LCNetConvBNLayer(num_channels, dw_size, num_channels, stride, num_channels);
    register_sublayer("dw_conv", this->dw_conv);
    this->pw_conv = new SNT LCNetConvBNLayer(num_channels, 1, num_filters, 1, 1);
    register_sublayer("pw_conv", this->pw_conv);
    if (use_se)
    {
        this->se = new SNT SEModule(num_channels);
        register_sublayer("se", this->se);
    }
}

DepthwiseSeparable::~DepthwiseSeparable()
{
    delete dw_conv;
    delete pw_conv;
    if (use_se)
    {
        delete se;
    }
}

Tensor* DepthwiseSeparable::create_tensors(Tensor* input)
{
    Tensor* x = dw_conv->create_tensors(input);
    if (use_se)
    {
        x = se->create_tensors(x);
    }
    x = pw_conv->create_tensors(x);
    return x;
}

Tensor* DepthwiseSeparable::feed_forward(Tensor* input)
{
    Tensor* x = dw_conv->feed_forward(input);
    if (use_se)
    {
        x = se->feed_forward(x);
    }
    x = pw_conv->feed_forward(x);
    return x;
}

LCNet::LCNet(float scale, std::vector<int>* feature_maps)
{
    this->scale = scale;
    this->feature_maps = feature_maps;

    this->conv1 = new SNT LCNetConvBNLayer(3, 3, make_divisible(16 * scale), 2, 1);
    register_sublayer("conv1", this->conv1);

    this->blocks2 = new Sequential();
    this->blocks3 = new Sequential();
    this->blocks4 = new Sequential();
    this->blocks5 = new Sequential();
    this->blocks6 = new Sequential();

    DepthwiseSeparable* layer;

    layer = new SNT DepthwiseSeparable(make_divisible(16 * scale), make_divisible(32 * scale), 1, 3, false);
    blocks2->add_sublayer("0", layer);
    register_sublayer("blocks2", blocks2);

    layer = new SNT DepthwiseSeparable(make_divisible(32 * scale), make_divisible(64 * scale), 2, 3, false);
    blocks3->add_sublayer("0", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(64 * scale), make_divisible(64 * scale), 1, 3, false);
    blocks3->add_sublayer("1", layer);
    register_sublayer("blocks3", blocks3);

    layer = new SNT DepthwiseSeparable(make_divisible(64 * scale), make_divisible(128 * scale), 2, 3, false);
    blocks4->add_sublayer("0", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(128 * scale), make_divisible(128 * scale), 1, 3, false);
    blocks4->add_sublayer("1", layer);
    register_sublayer("blocks4", blocks4);

    layer = new SNT DepthwiseSeparable(make_divisible(128 * scale), make_divisible(256 * scale), 2, 3, false);
    blocks5->add_sublayer("0", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(256 * scale), 1, 5, false);
    blocks5->add_sublayer("1", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(256 * scale), 1, 5, false);
    blocks5->add_sublayer("2", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(256 * scale), 1, 5, false);
    blocks5->add_sublayer("3", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(256 * scale), 1, 5, false);
    blocks5->add_sublayer("4", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(256 * scale), 1, 5, false);
    blocks5->add_sublayer("5", layer);
    register_sublayer("blocks5", blocks5);

    layer = new SNT DepthwiseSeparable(make_divisible(256 * scale), make_divisible(512 * scale), 2, 5, true);
    blocks6->add_sublayer("0", layer);
    layer = new SNT DepthwiseSeparable(make_divisible(512 * scale), make_divisible(512 * scale), 1, 5, true);
    blocks6->add_sublayer("1", layer);
    register_sublayer("blocks6", blocks6);

}

LCNet::~LCNet()
{
    delete conv1;
    delete blocks2;
    delete blocks3;
    delete blocks4;
    delete blocks5;
    delete blocks6;
}

int LCNet::make_divisible(float v, int divisor, int min_value)
{
    int min_value_ = min_value;
    if (min_value < 0)
    {
        min_value_ = divisor;
    }
    float temp = v + divisor / 2;
    temp /= divisor;
    int new_v = (int)temp;
    new_v *= divisor;
    int new_v_ = std::max(min_value, new_v);
    if (new_v_ < 0.9f * v)
    {
        new_v_ += divisor;
    }
    return new_v_;
}

std::vector<Tensor*>* LCNet::create_tensors(Tensor* input, char miemie2013)
{
    Tensor* x = input;
    x = conv1->create_tensors(x);
    x = blocks2->create_tensors(x);
    x = blocks3->create_tensors(x);
    if (std::find(feature_maps->begin(), feature_maps->end(), 2) != feature_maps->end())
    {
        x->referenceCount++;
        output_tensors->push_back(x);
    }
    x = blocks4->create_tensors(x);
    if (std::find(feature_maps->begin(), feature_maps->end(), 3) != feature_maps->end())
    {
        x->referenceCount++;
        output_tensors->push_back(x);
    }
    x = blocks5->create_tensors(x);
    if (std::find(feature_maps->begin(), feature_maps->end(), 4) != feature_maps->end())
    {
        x->referenceCount++;
        output_tensors->push_back(x);
    }
    x = blocks6->create_tensors(x);
    if (std::find(feature_maps->begin(), feature_maps->end(), 5) != feature_maps->end())
    {
        x->referenceCount++;
        output_tensors->push_back(x);
    }
    return output_tensors;
}

std::vector<Tensor*>* LCNet::feed_forward(Tensor* input, char miemie2013)
{
    Tensor* x = input;
    x = conv1->feed_forward(x);
    x = blocks2->feed_forward(x);
    x = blocks3->feed_forward(x);
    x = blocks4->feed_forward(x);
    x = blocks5->feed_forward(x);
    x = blocks6->feed_forward(x);
    return output_tensors;
}


}  // namespace miemiedet
