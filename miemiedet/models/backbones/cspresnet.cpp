#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "cspresnet.h"

namespace miemiedet {

ConvBNLayer::ConvBNLayer(int ch_in, int ch_out, int filter_size, int stride, int groups, int padding, char* act_name)
{
    miemienet::Config* cfg = miemienet::Config::getInstance();
    this->fuse_conv_bn = cfg->fuse_conv_bn;
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->filter_size = filter_size;
    this->stride = stride;
    this->groups = groups;
    this->padding = padding;
    this->act_name = act_name;
    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act = nullptr;
        }
        else
        {
            this->act = new SNT Activation(act_name, 0.f);
            register_sublayer("act", this->act);
        }
    }
    else
    {
        this->act = nullptr;
    }

    if (!cfg->fuse_conv_bn)
    {
        conv = new SNT Conv2d(ch_in, ch_out, filter_size, stride, padding, 1, groups, false);
    }
    else
    {
        conv = new SNT Conv2d(ch_in, ch_out, filter_size, stride, padding, 1, groups, true);
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

ConvBNLayer::~ConvBNLayer()
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

Tensor* ConvBNLayer::create_tensors(Tensor* input)
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

Tensor* ConvBNLayer::feed_forward(Tensor* input)
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

RepVggBlock::RepVggBlock(int ch_in, int ch_out, char* act_name)
{
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->act_name = act_name;

    this->conv1 = new SNT ConvBNLayer(ch_in, ch_out, 3, 1, 1, 1, nullptr);
    register_sublayer("conv1", this->conv1);
    this->conv2 = new SNT ConvBNLayer(ch_in, ch_out, 1, 1, 1, 0, nullptr);
    register_sublayer("conv2", this->conv2);
    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act = nullptr;
        }
        else
        {
            this->act = new SNT Activation(act_name, 0.f);
            register_sublayer("act", this->act);
        }
    }
}

RepVggBlock::~RepVggBlock()
{
    delete conv1;
    delete conv2;
    if (act)
    {
        delete act;
    }
}

Tensor* RepVggBlock::create_tensors(Tensor* input)
{
    Tensor* x1 = conv1->create_tensors(input);
    Tensor* x2 = conv2->create_tensors(input);
    if (act)
    {
        x1 = act->create_tensors(x1);
    }
    return x1;
}

Tensor* RepVggBlock::feed_forward(Tensor* input)
{
    Tensor* x1 = conv1->feed_forward(input);
    Tensor* x2 = conv2->feed_forward(input);
    miemienet::functional::elementwise(x1, x2, x1, ELE_ADD);
    if (act)
    {
        x1 = act->feed_forward(x1);
    }
    return x1;
}

BasicBlock::BasicBlock(int ch_in, int ch_out, char* act_name, bool shortcut)
{
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->act_name = act_name;
    this->shortcut = shortcut;

    this->conv1 = new SNT ConvBNLayer(ch_in, ch_out, 3, 1, 1, 1, act_name);
    register_sublayer("conv1", this->conv1);
    this->conv2 = new SNT RepVggBlock(ch_out, ch_out, act_name);
    register_sublayer("conv2", this->conv2);
}

BasicBlock::~BasicBlock()
{
    delete conv1;
    delete conv2;
}

Tensor* BasicBlock::create_tensors(Tensor* input)
{
    Tensor* y = conv1->create_tensors(input);
    y = conv2->create_tensors(y);
    return y;
}

Tensor* BasicBlock::feed_forward(Tensor* input)
{
    Tensor* y = conv1->feed_forward(input);
    y = conv2->feed_forward(y);
    if (shortcut)
    {
        miemienet::functional::elementwise(y, input, y, ELE_ADD);
    }
    return y;
}

EffectiveSELayer::EffectiveSELayer(int channels, char* act_name)
{
    this->channels = channels;
    this->act_name = act_name;
    if (act_name)
    {
        if (act_name == nullptr)
        {
            this->act = nullptr;
        }
        else
        {
            this->act = new SNT Activation(act_name, 0.f);
            register_sublayer("act", this->act);
        }
    }
    else
    {
        this->act = nullptr;
    }
    fc = new SNT Conv2d(channels, channels, 1, 1, 0, 1, 1, true);
    register_sublayer("fc", fc);

    Config* cfg = Config::getInstance();
    if (cfg->image_data_format == NCHW)
    {
        reduce_mean = new SNT Reduce(MMSHAPE2D(2, 3), true, RED_MEAN);
    }
    else if (cfg->image_data_format == NHWC)
    {
        reduce_mean = new SNT Reduce(MMSHAPE2D(1, 2), true, RED_MEAN);
    }
    register_sublayer("reduce_mean", reduce_mean);
}

EffectiveSELayer::~EffectiveSELayer()
{
    delete fc;
    delete reduce_mean;
    if (act)
    {
        delete act;
    }
}

Tensor* EffectiveSELayer::create_tensors(Tensor* input)
{
    Tensor* x_se = reduce_mean->create_tensors(input);
    x_se = fc->create_tensors(x_se);
    x_se = act->create_tensors(x_se);
    return input;
}

Tensor* EffectiveSELayer::feed_forward(Tensor* input)
{
    Tensor* x_se = reduce_mean->feed_forward(input);
    x_se = fc->feed_forward(x_se);
    x_se = act->feed_forward(x_se);
    miemienet::functional::elementwise(input, x_se, input, ELE_MUL);
    return input;
}

CSPResStage::CSPResStage(int ch_in, int ch_out, int n, int stride, char* act_name, bool use_attn)
{
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->n = n;
    this->stride = stride;
    this->act_name = act_name;
    this->use_attn = use_attn;

    int ch_mid = (ch_in + ch_out) / 2;
    if (stride == 2)
    {
        this->conv_down = new SNT ConvBNLayer(ch_in, ch_mid, 3, 2, 1, 1, act_name);
        register_sublayer("conv_down", this->conv_down);
    }
    else
    {
        this->conv_down = nullptr;
    }

    this->conv1 = new SNT ConvBNLayer(ch_mid, ch_mid / 2, 1, 1, 1, 0, act_name);
    register_sublayer("conv1", this->conv1);
    this->conv2 = new SNT ConvBNLayer(ch_mid, ch_mid / 2, 1, 1, 1, 0, act_name);
    register_sublayer("conv2", this->conv2);

    this->blocks = new Sequential();
    for (int i = 0; i < n; i++)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* conv_name = new char[64];
        sprintf(conv_name, "%d", i);  // xxx
        BasicBlock* layer = new SNT BasicBlock(ch_mid / 2, ch_mid / 2, act_name, true);
        blocks->add_sublayer(conv_name, layer);
    }
    register_sublayer("blocks", blocks);
    if (use_attn)
    {
        this->attn = new SNT EffectiveSELayer(ch_mid, "hardsigmoid");
        register_sublayer("attn", this->attn);
    }
    else
    {
        this->attn = nullptr;
    }
    this->conv3 = new SNT ConvBNLayer(ch_mid, ch_out, 1, 1, 1, 0, act_name);
    register_sublayer("conv3", this->conv3);

    Config* cfg = Config::getInstance();
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

CSPResStage::~CSPResStage()
{
    delete conv_down;
    delete conv1;
    delete conv2;
    delete attn;
    delete conv3;
    delete blocks;
    delete concat;
}

Tensor* CSPResStage::create_tensors(Tensor* input)
{
    Tensor* x;
    if (stride == 2)
    {
        x = conv_down->create_tensors(input);
    }
    else
    {
        x = input;
    }
    Tensor* y1 = conv1->create_tensors(x);
    Tensor* y2 = conv2->create_tensors(x);
    y2 = blocks->create_tensors(y2);
    y2 = concat->create_tensors(y1, y2);

    if (use_attn)
    {
        y2 = attn->create_tensors(y2);
    }
    y2 = conv3->create_tensors(y2);
    return y2;
}

Tensor* CSPResStage::feed_forward(Tensor* input)
{
    Tensor* x;
    if (stride == 2)
    {
        x = conv_down->feed_forward(input);
    }
    else
    {
        x = input;
    }
    Tensor* y1 = conv1->feed_forward(x);
    Tensor* y2 = conv2->feed_forward(x);
    y2 = blocks->feed_forward(y2);
    y2 = concat->feed_forward(y1, y2);

    if (use_attn)
    {
        y2 = attn->feed_forward(y2);
    }
    y2 = conv3->feed_forward(y2);
    return y2;
}

CSPResNet::CSPResNet(std::vector<int>* layers, std::vector<int>* channels, char* act_name, std::vector<int>* return_idx, bool depth_wise, bool use_large_stem, float width_mult, float depth_mult, int freeze_at)
{
    this->layers = layers;
    this->channels = channels;
    this->act_name = act_name;
    this->return_idx = return_idx;
    this->depth_wise = depth_wise;
    this->use_large_stem = use_large_stem;
    this->width_mult = width_mult;
    this->depth_mult = depth_mult;
    this->freeze_at = freeze_at;

    for (int i = channels->size() - 1; i >= 0; i--)
    {
        int cn = channels->at(i);
        cn = std::max(1, (int)(cn * width_mult + 0.5f));
        channels->at(i) = cn;
    }
    for (int i = layers->size() - 1; i >= 0; i--)
    {
        int ly = layers->at(i);
        ly = std::max(1, (int)(ly * depth_mult + 0.5f));
        layers->at(i) = ly;
    }

    this->stem = new Sequential();
    if (use_large_stem)
    {
        ConvBNLayer* conv1 = new SNT ConvBNLayer(3, channels->at(0) / 2, 3, 2, 1, 1, act_name);
        ConvBNLayer* conv2 = new SNT ConvBNLayer(channels->at(0) / 2, channels->at(0) / 2, 3, 1, 1, 1, act_name);
        ConvBNLayer* conv3 = new SNT ConvBNLayer(channels->at(0) / 2, channels->at(0), 3, 1, 1, 1, act_name);

        stem->add_sublayer("conv1", conv1);
        stem->add_sublayer("conv2", conv2);
        stem->add_sublayer("conv3", conv3);
    }
    else
    {
        ConvBNLayer* conv1 = new SNT ConvBNLayer(3, channels->at(0) / 2, 3, 2, 1, 1, act_name);
        ConvBNLayer* conv2 = new SNT ConvBNLayer(channels->at(0) / 2, channels->at(0), 3, 1, 1, 1, act_name);

        stem->add_sublayer("conv1", conv1);
        stem->add_sublayer("conv2", conv2);
    }
    register_sublayer("stem", stem);

    this->stages = new LayerList();
    for (int i = 0; i < channels->size() - 1; i++)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* conv_name = new char[64];
        sprintf(conv_name, "%d", i);  // xxx
        CSPResStage* stage = new SNT CSPResStage(channels->at(i), channels->at(i+1), layers->at(i), 2, act_name);
        stages->add_sublayer(conv_name, stage);
    }
    register_sublayer("stages", stages);
}

CSPResNet::~CSPResNet()
{
    delete stem;
    delete stages;
}

std::vector<Tensor*>* CSPResNet::create_tensors(Tensor* input, char miemie2013)
{
    Tensor* x = stem->create_tensors(input);
    for (int idx = 0; idx < stages->size(); idx++)
    {
        Layer* layer = stages->at(idx);
        x = layer->create_tensors(x);
        if (std::find(return_idx->begin(), return_idx->end(), idx) != return_idx->end())
        {
            x->referenceCount++;
            output_tensors->push_back(x);
        }
    }
    return output_tensors;
}

std::vector<Tensor*>* CSPResNet::feed_forward(Tensor* input, char miemie2013)
{
    Tensor* x = stem->feed_forward(input);
    for (int idx = 0; idx < stages->size(); idx++)
    {
        Layer* layer = stages->at(idx);
        x = layer->feed_forward(x);
    }
    return output_tensors;
}


}  // namespace miemiedet
