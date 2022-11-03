#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "reduce.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/reduce_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/reduce_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Reduce::Reduce(std::vector<int>* dims, bool keepdim, int op_type)
{
    this->dims = dims;
    this->keepdim = keepdim;
    this->op_type = op_type;
    this->son = nullptr;
}

Reduce::~Reduce()
{
    if (son != nullptr)
    {
        delete son;
    }
}

Tensor* Reduce::create_tensors(Tensor* input)
{
    if (input_tensors->size() == 0)
    {
        input->referenceCount++;
        input_tensors->push_back(input);

        Tensor* output;
        const int tensor_dims = input->dims;
        if (tensor_dims == 4 && dims->at(0) == 1 && dims->at(1) == 2)
        {
            const int N = input->shape->at(0);
            const int C = input->shape->at(1);
            const int H = input->shape->at(2);
            const int W = input->shape->at(3);
            output = new SNT Tensor(MMSHAPE4D(N, 1, 1, W), FP32, false, false);
        }
        else
        {
            input->print_msg("input");
            printf("Reduce op type tensor_dims == %d && dim == %d not implemented!\n", tensor_dims, 333);
            exit(1);
        }
        output->referenceCount++;
        output_tensors->push_back(output);
        return output;
    }
    else
    {
        if (son == nullptr)
        {
            son = new SNT Reduce(dims, keepdim, op_type);
        }
        return son->create_tensors(input);
    }
}

Tensor* Reduce::feed_forward(Tensor* input)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::reduce(input, output, dims, keepdim, op_type);
        return output;
    }
    else
    {
        return son->feed_forward(input);
    }
}

NS_MM_END
