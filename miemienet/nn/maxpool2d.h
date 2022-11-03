#ifndef __MAXPOOL2D_H__
#define __MAXPOOL2D_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class MaxPool2d : public Layer
{
public:
    MaxPool2d(int kernel_size=1, int stride=1, int padding=0, bool ceil_mode=false);
    MaxPool2d(int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, bool ceil_mode=false);
    ~MaxPool2d();

    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    bool ceil_mode;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __MAXPOOL2D_H__
