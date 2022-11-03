#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "custom_pan.h"

namespace miemiedet {

SPP::SPP(int ch_in, int ch_out, int k, char* act_name)
{
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->k = k;
    this->act_name = act_name;

    this->conv = new SNT ConvBNLayer(ch_in, ch_out, k, 1, 1, k / 2, act_name);
    register_sublayer("conv", this->conv);

    this->maxpools = new LayerList();
    MaxPool2d* maxpool1 = new SNT MaxPool2d(5, 1, 2, false);
    MaxPool2d* maxpool2 = new SNT MaxPool2d(9, 1, 4, false);
    MaxPool2d* maxpool3 = new SNT MaxPool2d(13, 1, 6, false);
    maxpools->add_sublayer("1", maxpool1);
    maxpools->add_sublayer("2", maxpool2);
    maxpools->add_sublayer("3", maxpool3);
    register_sublayer("maxpools", maxpools);

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

SPP::~SPP()
{
    delete conv;
    delete maxpools;
    delete concat;
}

Tensor* SPP::create_tensors(Tensor* input)
{
    Tensor* x1 = maxpools->at(0)->create_tensors(input);
    Tensor* x2 = maxpools->at(1)->create_tensors(input);
    Tensor* x3 = maxpools->at(2)->create_tensors(input);
    Tensor* y = concat->create_tensors(input, x1, x2, x3);
    y = conv->create_tensors(y);
    return y;
}

Tensor* SPP::feed_forward(Tensor* input)
{
    Tensor* x1 = maxpools->at(0)->feed_forward(input);
    Tensor* x2 = maxpools->at(1)->feed_forward(input);
    Tensor* x3 = maxpools->at(2)->feed_forward(input);
    Tensor* y = concat->feed_forward(input, x1, x2, x3);
    y = conv->feed_forward(y);
    return y;
}

CSPStage::CSPStage(int ch_in, int ch_out, int n, char* act_name, bool spp)
{
    this->ch_in = ch_in;
    this->ch_out = ch_out;
    this->n = n;
    this->act_name = act_name;
    this->spp = spp;

    int ch_mid = ch_out / 2;
    this->conv1 = new SNT ConvBNLayer(ch_in, ch_mid, 1, 1, 1, 0, act_name);
    register_sublayer("conv1", this->conv1);
    this->conv2 = new SNT ConvBNLayer(ch_in, ch_mid, 1, 1, 1, 0, act_name);
    register_sublayer("conv2", this->conv2);
    this->convs = new Sequential();
    for (int i = 0; i < n; i++)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* conv_name = new char[64];
        sprintf(conv_name, "%d", i);  // xxx
        BasicBlock* layer = new SNT BasicBlock(ch_mid, ch_mid, act_name, false);
        convs->add_sublayer(conv_name, layer);
        if (i == (n - 1) / 2 && spp)
        {
            char* spp_name = new char[64];
            sprintf(spp_name, "spp");  // xxx
            SPP* spp_layer = new SNT SPP(ch_mid * 4, ch_mid, 1, act_name);
            convs->add_sublayer(spp_name, spp_layer);
        }
    }
    register_sublayer("convs", convs);
    this->conv3 = new SNT ConvBNLayer(ch_mid * 2, ch_out, 1, 1, 1, 0, act_name);
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

CSPStage::~CSPStage()
{
    delete conv1;
    delete conv2;
    delete conv3;
    delete convs;
    delete concat;
}

Tensor* CSPStage::create_tensors(Tensor* input)
{
    Tensor* y1 = conv1->create_tensors(input);
    Tensor* y2 = conv2->create_tensors(input);
    y2 = convs->create_tensors(y2);
    y1 = concat->create_tensors(y1, y2);
    y1 = conv3->create_tensors(y1);
    return y1;
}

Tensor* CSPStage::feed_forward(Tensor* input)
{
    Tensor* y1 = conv1->feed_forward(input);
    Tensor* y2 = conv2->feed_forward(input);
    y2 = convs->feed_forward(y2);
    y1 = concat->feed_forward(y1, y2);
    y1 = conv3->feed_forward(y1);
    return y1;
}

CustomCSPPAN::CustomCSPPAN(std::vector<int>* in_channels, std::vector<int>* out_channels, char* act_name, int stage_num, int block_num, bool drop_block, int block_size, float keep_prob, bool spp, float width_mult, float depth_mult)
{
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->act_name = act_name;
    this->stage_num = stage_num;
    this->drop_block = drop_block;
    this->block_size = block_size;
    this->keep_prob = keep_prob;
    this->spp = spp;
    this->width_mult = width_mult;
    this->depth_mult = depth_mult;

    for (int i = out_channels->size() - 1; i >= 0; i--)
    {
        int cn = out_channels->at(i);
        cn = std::max(1, (int)(cn * width_mult + 0.5f));
        out_channels->at(i) = cn;
    }
    block_num = std::max(1, (int)(block_num * depth_mult + 0.5f));
    this->block_num = block_num;
    this->num_blocks = in_channels->size();

    this->fpn_stages = new std::vector<std::vector<Layer*>*>;
    this->fpn_routes = new std::vector<Layer*>;
    int ch_pre = 0;
    int fpn_routes_id = 0;
    for (int i = 0; i < num_blocks; i++)
    {
        int ch_in = in_channels->at(num_blocks - 1 - i);
        int ch_out = out_channels->at(i);
        if (i > 0)
        {
            ch_in += ch_pre / 2;
        }
        std::vector<Layer*>* stage = new std::vector<Layer*>;
        for (int j = 0; j < stage_num; j++)
        {
            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* conv_name = new char[64];
            sprintf(conv_name, "fpn_stages.%d.%d", i, j);  // xxx
            CSPStage* layer = new SNT CSPStage(j == 0 ? ch_in : ch_out, ch_out, block_num, act_name, spp && i == 0);
            register_sublayer(conv_name, layer);
            stage->push_back(layer);
        }
        if (drop_block)
        {
            ;   // not impl
        }
        fpn_stages->push_back(stage);
        if (i < num_blocks - 1)
        {
            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* conv_name = new char[64];
            sprintf(conv_name, "fpn_routes.%d", fpn_routes_id);  // xxx
            ConvBNLayer* conv = new SNT ConvBNLayer(ch_out, ch_out / 2, 1, 1, 1, 0, act_name);
            register_sublayer(conv_name, conv);
            fpn_routes->push_back(conv);
            fpn_routes_id++;
        }
        ch_pre = ch_out;
    }

    this->pan_stages = new std::vector<std::vector<Layer*>*>;
    this->pan_routes = new std::vector<Layer*>;
    int pan_routes_id = num_blocks - 2;
    for (int i = num_blocks - 2; i >= 0; i--)
    {
        // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
        char* conv_name = new char[64];
        sprintf(conv_name, "pan_routes.%d", pan_routes_id);  // xxx
        ConvBNLayer* conv = new SNT ConvBNLayer(out_channels->at(i+1), out_channels->at(i+1), 3, 2, 1, 1, act_name);
        register_sublayer(conv_name, conv);
        pan_routes->push_back(conv);
        pan_routes_id--;

        int ch_in = out_channels->at(i) + out_channels->at(i+1);
        int ch_out = out_channels->at(i);
        std::vector<Layer*>* stage = new std::vector<Layer*>;
        for (int j = 0; j < stage_num; j++)
        {
            // 作为持久的字符串，应该是堆空间上而不是栈空间上，即 char conv_name[64];是错误的！
            char* conv_name2 = new char[64];
            sprintf(conv_name2, "pan_stages.%d.%d", i, j);  // xxx
            CSPStage* layer = new SNT CSPStage(j == 0 ? ch_in : ch_out, ch_out, block_num, act_name, false);
            register_sublayer(conv_name2, layer);
            stage->push_back(layer);
        }
        if (drop_block)
        {
            ;   // not impl
        }
        pan_stages->push_back(stage);
    }
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
    interp = new SNT Interp(0, 0, 2.f, 2.f);
    register_sublayer("interp", interp);
}

CustomCSPPAN::~CustomCSPPAN()
{
    // 倒序遍历，可以边遍历边删除
    for (int j = fpn_stages->size() - 1; j >= 0; j--)
    {
        for (int i = fpn_stages->at(j)->size() - 1; i >= 0; i--)
        {
            Layer* layer = fpn_stages->at(j)->at(i);
            delete layer;
            fpn_stages->at(j)->erase(fpn_stages->at(j)->begin() + i);
        }
        delete fpn_stages->at(j);
    }
    delete fpn_stages;
    // 倒序遍历，可以边遍历边删除
    for (int i = fpn_routes->size() - 1; i >= 0; i--)
    {
        Layer* layer = fpn_routes->at(i);
        delete layer;
        fpn_routes->erase(fpn_routes->begin() + i);
    }
    delete fpn_routes;
    // 倒序遍历，可以边遍历边删除
    for (int j = pan_stages->size() - 1; j >= 0; j--)
    {
        for (int i = pan_stages->at(j)->size() - 1; i >= 0; i--)
        {
            Layer* layer = pan_stages->at(j)->at(i);
            delete layer;
            pan_stages->at(j)->erase(pan_stages->at(j)->begin() + i);
        }
        delete pan_stages->at(j);
    }
    delete pan_stages;
    // 倒序遍历，可以边遍历边删除
    for (int i = pan_routes->size() - 1; i >= 0; i--)
    {
        Layer* layer = pan_routes->at(i);
        delete layer;
        pan_routes->erase(pan_routes->begin() + i);
    }
    delete pan_routes;
    delete concat;
    delete interp;
}

std::vector<Tensor*>* CustomCSPPAN::create_tensors(std::vector<Tensor*>* inputs, char miemie2013)
{
    Tensor* route;
    std::vector<Tensor*>* fpn_feats = temp_tensors;
    miemienet::Config* cfg = miemienet::Config::getInstance();
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* block = inputs->at(inputs->size() - 1 - i);
        if (i > 0)
        {
            block = concat->create_tensors(route, block);
        }
        route = block;
        for (int j = 0; j < fpn_stages->at(i)->size(); j++)
        {
            Layer* layer = fpn_stages->at(i)->at(j);
            route = layer->create_tensors(route);
        }
        route->referenceCount++;
        fpn_feats->push_back(route);
        if (i < num_blocks - 1)
        {
            route = fpn_routes->at(i)->create_tensors(route);
            route = interp->create_tensors(route);
        }
    }

    std::vector<Tensor*>* pan_feats = output_tensors;
    route = fpn_feats->at(fpn_feats->size() - 1);
    route->referenceCount++;
    pan_feats->push_back(route);

    for (int i = num_blocks - 2; i >= 0; i--)
    {
        Tensor* block = fpn_feats->at(i);
        route = pan_routes->at(num_blocks - 2 - i)->create_tensors(route);

        block = concat->create_tensors(route, block);

        route = block;
        for (int j = 0; j < pan_stages->at(num_blocks - 2 - i)->size(); j++)
        {
            Layer* layer = pan_stages->at(num_blocks - 2 - i)->at(j);
            route = layer->create_tensors(route);
        }
        route->referenceCount++;
        pan_feats->insert(pan_feats->begin(), route);
    }

    return pan_feats;
}

std::vector<Tensor*>* CustomCSPPAN::feed_forward(std::vector<Tensor*>* inputs, char miemie2013)
{
    Tensor* route;
    std::vector<Tensor*>* fpn_feats = temp_tensors;
    for (int i = 0; i < inputs->size(); i++)
    {
        Tensor* block = inputs->at(inputs->size() - 1 - i);
        if (i > 0)
        {
            block = concat->feed_forward(route, block);
        }
        route = block;
        for (int j = 0; j < fpn_stages->at(i)->size(); j++)
        {
            Layer* layer = fpn_stages->at(i)->at(j);
            route = layer->feed_forward(route);
        }
//        fpn_feats->push_back(route);
        if (i < num_blocks - 1)
        {
            route = fpn_routes->at(i)->feed_forward(route);
            route = interp->feed_forward(route);
        }
    }

    std::vector<Tensor*>* pan_feats = output_tensors;
    route = fpn_feats->at(fpn_feats->size() - 1);
//    pan_feats->push_back(route);

    for (int i = num_blocks - 2; i >= 0; i--)
    {
        Tensor* block = fpn_feats->at(i);
        route = pan_routes->at(num_blocks - 2 - i)->feed_forward(route);

        block = concat->feed_forward(route, block);

        route = block;
        for (int j = 0; j < pan_stages->at(num_blocks - 2 - i)->size(); j++)
        {
            Layer* layer = pan_stages->at(num_blocks - 2 - i)->at(j);
            route = layer->feed_forward(route);
        }
//        pan_feats->insert(pan_feats->begin(), route);
    }

    return pan_feats;
}

}  // namespace miemiedet
