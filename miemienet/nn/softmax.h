#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Softmax : public Layer
{
public:
    Softmax(int dim=-1, bool inplace=false);
    ~Softmax();

    int dim;
    bool inplace;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __SOFTMAX_H__
