#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "activation.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/activation_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/activation_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Activation::Activation(char* type, float alpha, bool inplace)
{
    this->type = type;
    this->alpha = alpha;
    this->inplace = inplace;
    this->son = nullptr;
}

Activation::~Activation()
{
    if (son != nullptr)
    {
        delete son;
    }
}

Tensor* Activation::create_tensors(Tensor* input)
{
    if (input_tensors->size() == 0)
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
    else
    {
        if (son == nullptr)
        {
            son = new SNT Activation(type, alpha, inplace);
        }
        return son->create_tensors(input);
    }
}

Tensor* Activation::feed_forward(Tensor* input)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::activation(input, output, type, alpha);
        return output;
    }
    else
    {
        return son->feed_forward(input);
    }
}

NS_MM_END
