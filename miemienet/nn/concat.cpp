#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include "concat.h"
#include "../framework/config.h"

#if BACKEND_X86
#include "common/concat_common.h"
#endif // BACKEND_X86

#if BACKEND_ARM
#include "common/concat_common.h"
#endif // BACKEND_ARM

NS_MM_BEGIN

Concat::Concat(int dim)
{
    this->dim = dim;
    this->son = nullptr;
}

Concat::~Concat()
{
    if (son != nullptr)
    {
        delete son;
    }
}

Tensor* Concat::create_tensors(Tensor* input1, Tensor* input2)
{
    if (input_tensors->size() == 0)
    {
        input1->referenceCount++;
        input_tensors->push_back(input1);
        input2->referenceCount++;
        input_tensors->push_back(input2);

        const int dims = input1->dims;
        int positive_dim = dim < 0 ? dims + dim : dim;
        if (positive_dim < 0 || positive_dim >= dims)
        {
            printf("Error from concat op (2 tensors), invalid arg dim=%d.\n", dim);
            exit(1);
        }

        Tensor* output;
        if (dims == 4 && positive_dim == 3)
        {
            const int N = input1->shape->at(0);
            const int C = input1->shape->at(1);
            const int H = input1->shape->at(2);
            const int W1 = input1->shape->at(3);
            const int W2 = input2->shape->at(3);
            output = new SNT Tensor(MMSHAPE4D(N, C, H, W1+W2), FP32, false, false);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
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
            son = new SNT Concat(dim);
        }
        return son->create_tensors(input1, input2);
    }
}

Tensor* Concat::create_tensors(Tensor* input1, Tensor* input2, Tensor* input3)
{
    if (input_tensors->size() == 0)
    {
        input1->referenceCount++;
        input_tensors->push_back(input1);
        input2->referenceCount++;
        input_tensors->push_back(input2);
        input3->referenceCount++;
        input_tensors->push_back(input3);

        const int dims = input1->dims;
        int positive_dim = dim < 0 ? dims + dim : dim;
        if (positive_dim < 0 || positive_dim >= dims)
        {
            printf("Error from concat op (3 tensors), invalid arg dim=%d.\n", dim);
            exit(1);
        }

        Tensor* output;
        if (dims == 3 && positive_dim == 1)
        {
            const int N = input1->shape->at(0);
            const int H1 = input1->shape->at(1);
            const int H2 = input2->shape->at(1);
            const int H3 = input3->shape->at(1);
            const int W = input1->shape->at(2);
            output = new SNT Tensor(MMSHAPE3D(N, H1+H2+H3, W), FP32, false, false);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
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
            son = new SNT Concat(dim);
        }
        return son->create_tensors(input1, input2, input3);
    }
}

Tensor* Concat::create_tensors(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4)
{
    if (input_tensors->size() == 0)
    {
        input1->referenceCount++;
        input_tensors->push_back(input1);
        input2->referenceCount++;
        input_tensors->push_back(input2);
        input3->referenceCount++;
        input_tensors->push_back(input3);
        input4->referenceCount++;
        input_tensors->push_back(input4);

        const int dims = input1->dims;
        int positive_dim = dim < 0 ? dims + dim : dim;
        if (positive_dim < 0 || positive_dim >= dims)
        {
            printf("Error from concat op (4 tensors), invalid arg dim=%d.\n", dim);
            exit(1);
        }

        Tensor* output;
        if (dims == 4 && positive_dim == 3)
        {
            const int N = input1->shape->at(0);
            const int C = input1->shape->at(1);
            const int H = input1->shape->at(2);
            const int W1 = input1->shape->at(3);
            const int W2 = input2->shape->at(3);
            const int W3 = input3->shape->at(3);
            const int W4 = input4->shape->at(3);
            output = new SNT Tensor(MMSHAPE4D(N, C, H, W1+W2+W3+W4), FP32, false, false);
        }
        else if (dims == 3 && positive_dim == 1)
        {
            const int N = input1->shape->at(0);
            const int H1 = input1->shape->at(1);
            const int H2 = input2->shape->at(1);
            const int H3 = input3->shape->at(1);
            const int H4 = input4->shape->at(1);
            const int W = input1->shape->at(2);
            output = new SNT Tensor(MMSHAPE3D(N, H1+H2+H3+H4, W), FP32, false, false);
        }
        else
        {
            printf("concat op type dims == %d && dim == %d not implemented!\n", dims, dim);
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
            son = new SNT Concat(dim);
        }
        return son->create_tensors(input1, input2, input3, input4);
    }
}

Tensor* Concat::feed_forward(Tensor* input1, Tensor* input2)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input1->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::concat(input1, input2, output, dim);
        return output;
    }
    else
    {
        return son->feed_forward(input1, input2);
    }
}

Tensor* Concat::feed_forward(Tensor* input1, Tensor* input2, Tensor* input3)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input1->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::concat(input1, input2, input3, output, dim);
        return output;
    }
    else
    {
        return son->feed_forward(input1, input2, input3);
    }
}

Tensor* Concat::feed_forward(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4)
{
    Tensor* input_ = input_tensors->at(0);
    if (input_->id == input1->id)
    {
        Tensor* output = output_tensors->at(0);
        miemienet::functional::concat(input1, input2, input3, input4, output, dim);
        return output;
    }
    else
    {
        return son->feed_forward(input1, input2, input3, input4);
    }
}

NS_MM_END
