#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Conv2d : public Layer
{
public:
    Conv2d(int in_channels, int out_channels, int kernel_size=1, int stride=1, int padding=0, int dilation=1, int groups=1, bool use_bias=true, bool create_weights=true);
    Conv2d(int in_channels, int out_channels, int kernel_h=1, int kernel_w=1, int stride_h=1, int stride_w=1, int padding_h=0, int padding_w=0, int dilation_h=1, int dilation_w=1, int groups=1, bool use_bias=true, bool create_weights=true);
    ~Conv2d();
    void reset_parameters();

    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
    int groups;
    bool use_bias;

    Conv2d* son;

    Tensor* weight;
    Tensor* bias;
    Tensor* group_weights;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
    virtual void print_msg(char* name);
private:
};

NS_MM_END

#endif // __CONV2D_H__
