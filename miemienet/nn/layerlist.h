#ifndef __LAYERLIST_H__
#define __LAYERLIST_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class LayerList : public Layer
{
public:
    LayerList();
    ~LayerList();

    std::vector<Layer*>* layers;

    void add_sublayer(char* layer_name, Layer* layer);
    Layer* at(int i);
    size_t size();
private:
};

NS_MM_END

#endif // __LAYERLIST_H__
