#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include "../macros.h"
#include "layer.h"

NS_MM_BEGIN

class Tensor
{
    friend class Layer;
public:
    Tensor(std::vector<int>* shape, int dtype, bool is_param, bool init, float init_value=0.f);
    ~Tensor();
    void load_from_txt(char* name);
    void set_data_fp32(float* new_data);
    void set_data_fp32(float val);
    void set_data_int32(int* new_data);
    void set_data_int32(int val);
    void normal_init(float mean=0.f, float std=1.f, int seed=0);
    void save_as_bin(char* name);
    float* get_data_fp32();
    int* get_data_int32();
    void print_data(int max_num=-1);
    void reshape(std::vector<int>* shape);
    void restore_shape();
    void squeeze(int dim);
    void unsqueeze(int dim);
    void print_msg(char* name);
    std::vector<int>* clone_shape();

    int id;
    int dtype;
    float* data_fp32;
    int* data_int32;

    std::vector<int>* shape;
    int times_as_input;
    int dims;
    int ori_D0;
    int ori_D1;
    int ori_D2;
    int ori_D3;
    int numel;
    bool is_param;
    bool is_buffer;
    bool is_zero;
    int referenceCount;
private:
};

NS_MM_END

#endif // __TENSOR_H__
