#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Transpose : public Layer
{
public:
    Transpose(int transpose_type);
    ~Transpose();

    int transpose_type;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __TRANSPOSE_H__
