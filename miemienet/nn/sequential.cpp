#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "sequential.h"
#include "../framework/config.h"

NS_MM_BEGIN

Sequential::Sequential()
{
    this->layers = new std::vector<Layer*>;
}

Sequential::~Sequential()
{
    // 倒序遍历，可以边遍历边删除
    for (int i = layers->size() - 1; i >= 0; i--)
    {
        Layer* layer = layers->at(i);
        delete layer;
        layers->erase(layers->begin() + i);
    }
    delete layers;
}

void Sequential::add_sublayer(char* layer_name, Layer* layer)
{
    register_sublayer(layer_name, layer);
    layers->push_back(layer);
}

Tensor* Sequential::create_tensors(Tensor* input)
{
    Tensor* x = input;
    for (int i = 0; i < layers->size(); i++)
    {
        Layer* layer = layers->at(i);
        x = layer->create_tensors(x);
    }
    return x;
}

Tensor* Sequential::feed_forward(Tensor* input)
{
    Tensor* x = input;
    for (int i = 0; i < layers->size(); i++)
    {
        Layer* layer = layers->at(i);
        x = layer->feed_forward(x);
    }
    return x;
}

NS_MM_END
