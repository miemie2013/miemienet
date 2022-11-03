#ifndef __REDUCE_H__
#define __REDUCE_H__

#include "../macros.h"
#include "../framework/layer.h"

NS_MM_BEGIN

class Reduce : public Layer
{
public:
    Reduce(std::vector<int>* dims, bool keepdim, int op_type);
    ~Reduce();

    std::vector<int>* dims;
    bool keepdim;
    int op_type;

    Reduce* son;

    virtual Tensor* create_tensors(Tensor* input);
    virtual Tensor* feed_forward(Tensor* input);
private:
};

NS_MM_END

#endif // __REDUCE_H__
