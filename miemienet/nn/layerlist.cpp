#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "layerlist.h"
#include "../framework/config.h"

NS_MM_BEGIN

LayerList::LayerList()
{
    this->layers = new std::vector<Layer*>;
}

LayerList::~LayerList()
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

void LayerList::add_sublayer(char* layer_name, Layer* layer)
{
    register_sublayer(layer_name, layer);
    layers->push_back(layer);
}

Layer* LayerList::at(int i)
{
    return layers->at(i);
}

size_t LayerList::size()
{
    return layers->size();
}

NS_MM_END
