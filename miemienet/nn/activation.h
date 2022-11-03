#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Activation : public Layer
{
public:
    Activation(char* type, float alpha=0.f, bool inplace=false);
    ~Activation();

    char* type;
    float alpha;
    bool inplace;

    Activation* son;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __ACTIVATION_H__
