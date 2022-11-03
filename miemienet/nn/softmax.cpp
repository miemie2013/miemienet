#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "softmax.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/softmax_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/softmax_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Softmax::Softmax(int dim, bool inplace)
{
    this->dim = dim;
    this->inplace = inplace;
}

Softmax::~Softmax()
{
}

Tensor* Softmax::create_tensors(Tensor* input)
{
    input->referenceCount++;
    input_tensors->push_back(input);

    if (inplace)
    {
        input->referenceCount++;
        output_tensors->push_back(input);
        return input;
    }
    else
    {
        std::vector<int>* output_shape = input->clone_shape();
        Tensor* output = new SNT Tensor(output_shape, FP32, false, false);
        output->referenceCount++;
        output_tensors->push_back(output);
        return output;
    }
}

Tensor* Softmax::feed_forward(Tensor* input)
{
    Tensor* output = output_tensors->at(0);
    miemienet::functional::softmax(input, output, dim);
    return output;
}

NS_MM_END
