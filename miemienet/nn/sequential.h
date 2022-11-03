#ifndef __SEQUENTIAL_H__
#define __SEQUENTIAL_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Sequential : public Layer
{
public:
    Sequential();
    ~Sequential();

    std::vector<Layer*>* layers;

    void add_sublayer(char* layer_name, Layer* layer);
    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __SEQUENTIAL_H__
