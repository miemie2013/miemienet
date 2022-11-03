#ifndef __INTERP_H__
#define __INTERP_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Interp : public Layer
{
public:
    Interp(int size_h=0, int size_w=0, float scale_h=-1.f, float scale_w=-1.f, char* mode="nearest", bool align_corners=false, bool recompute_scale_factor=false);
    ~Interp();

    int size_h;
    int size_w;
    float scale_h;
    float scale_w;
    char* mode;
    bool align_corners;
    bool recompute_scale_factor;

    Interp* son;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __INTERP_H__
