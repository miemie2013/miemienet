#ifndef __LAYER_H__
#define __LAYER_H__

#include <vector>
#include "../macros.h"
#include "tensor.h"

NS_MM_BEGIN

class Tensor;

class Layer
{
public:
    Layer();
    virtual ~Layer();

    virtual Tensor* create_tensors(Tensor* input) { return nullptr; }
    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2) { return nullptr; }
    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2, Tensor* input3) { return nullptr; }
    virtual Tensor* create_tensors(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4) { return nullptr; }
    virtual std::vector<Tensor*>* create_tensors(Tensor* input, char miemie2013) { return nullptr; }
    virtual Tensor* create_tensors(std::vector<Tensor*>* inputs) { return nullptr; }
    virtual std::vector<Tensor*>* create_tensors(std::vector<Tensor*>* inputs, char miemie2013) { return nullptr; }

    virtual Tensor* feed_forward(Tensor* input) { return nullptr; }
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2) { return nullptr; }
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2, Tensor* input3) { return nullptr; }
    virtual Tensor* feed_forward(Tensor* input1, Tensor* input2, Tensor* input3, Tensor* input4) { return nullptr; }
    virtual std::vector<Tensor*>* feed_forward(Tensor* input, char miemie2013) { return nullptr; }
    virtual Tensor* feed_forward(std::vector<Tensor*>* inputs) { return nullptr; }
    virtual std::vector<Tensor*>* feed_forward(std::vector<Tensor*>* inputs, char miemie2013) { return nullptr; }

    std::vector<Tensor*>* input_tensors;
    std::vector<Tensor*>* temp_tensors;
    std::vector<Tensor*>* temp2_tensors;
    std::vector<Tensor*>* output_tensors;

    bool training;
    bool inplace;
    void register_sublayer(char* name, Layer* layer);
    void train();
    void eval();
    virtual void print_msg(char* name) {}
    std::vector<char*> param_names;
    std::vector<Tensor*> params;
    Tensor* register_buffer(char* name, std::vector<int>* shape, int dtype, bool init, float init_value=0.f);
    Tensor* create_parameter(char* name, std::vector<int>* shape, int dtype, bool init, float init_value=0.f);
    void parameters(std::vector<Tensor*>* params);
    void named_parameters(std::vector<char*>* param_names, std::vector<Tensor*>* params, char* prefix=nullptr);
    void load_state_dict(char* name, char* param_prefix_name=nullptr);
private:
    std::vector<Layer*> sublayers;
    std::vector<char*> sublayer_names;
};

NS_MM_END

#endif // __LAYER_H__
