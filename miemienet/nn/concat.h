#ifndef __CONCAT_H__
#define __CONCAT_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Concat : public Layer
{
public:
    Concat(int dim=-1);
    ~Concat();

    int dim;

    Concat* son;

    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2);
    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2, Tensor* input3);
    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4);
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2);
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2, Tensor* input3);
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4);
private:
};

NS_MM_END

#endif // __CONCAT_H__
