#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include "config.h"
#include "tensor.h"
#include "memoryallocator.h"
#include "tensoridmanager.h"

#if BACKEND_X86
#include <immintrin.h>
#endif // BACKEND_X86

#if BACKEND_ARM
//#include <arm_neon.h>
#endif // BACKEND_ARM

NS_MM_BEGIN

Tensor::Tensor(std::vector<int>* shape, int dtype, bool is_param, bool init, float init_value)
{
    this->shape = shape;
    this->dtype = dtype;
    this->is_param = is_param;
    this->is_buffer = false;
    this->is_zero = false;
    this->times_as_input = 0;
    this->referenceCount = 0;

    if (is_param)
    {
        this->id = TensorIdManager::getInstance()->assign_param_id();
    }
    else
    {
        this->id = TensorIdManager::getInstance()->assign_tensor_id();
    }

    numel = 1;
    dims = shape->size();
    if (dims == 4)
    {
        ori_D0 = shape->at(0);
        ori_D1 = shape->at(1);
        ori_D2 = shape->at(2);
        ori_D3 = shape->at(3);
    }
    for (int ILoveC = 0; ILoveC < shape->size(); ILoveC++)
    {
        if (shape->at(ILoveC) < 1)
        {
            printf("shape invalid.");
            exit(1);
        }
        numel *= shape->at(ILoveC);
    }
    if (dtype == FP32)
    {
        const int bytes = sizeof(float) * numel;
        if (is_param)
        {
            data_fp32 = (float*) malloc(bytes);
        }
        else
        {
            data_fp32 = MemoryAllocator::getInstance()->assign_fp32_memory(bytes);
        }
        if (init)
        {
            this->set_data_fp32(init_value);
        }
    }
    else if (dtype == INT32)
    {
        const int bytes = sizeof(int) * numel;
        if (is_param)
        {
            data_int32 = (int*) malloc(bytes);
        }
        else
        {
            data_int32 = MemoryAllocator::getInstance()->assign_int32_memory(bytes);
        }
        if (init)
        {
            this->set_data_int32((int)(init_value));
        }
    }
    else
    {
        printf("DTYPE::xxx not implemented.");
        exit(1);
    }
#ifdef DEBUG
    if (is_param)
    {
//        printf("create Param, id=%d, dims=%d, numel=%d\n", id, dims, numel);
    }
    else
    {
//        printf("create Tensor, id=%d, dims=%d, numel=%d\n", id, dims, numel);
    }
#endif
}

Tensor::~Tensor()
{
//    if (is_param)
//    {
        if (dtype == FP32)
        {
            free(data_fp32);
            data_fp32 = nullptr;
        }
        else if (dtype == INT32)
        {
            free(data_int32);
            data_int32 = nullptr;
        }
//    }

#ifdef DEBUG
    if (is_param)
    {
//        printf("delete Param, id=%d, dims=%d, numel=%d\n", id, dims, numel);
    }
    else
    {
//        printf("delete Tensor, id=%d, dims=%d, numel=%d\n", id, dims, numel);
    }
#endif
    delete shape;
    shape = nullptr;
}

void Tensor::load_from_txt(char* name)
{
    FILE* fp = fopen(name, "r");
    if (!fp)
    {
        printf("file %s not exist.\n", name);
        exit(1);
    }

    int bytes = 0;
    if (dtype == FP32)
    {
        bytes = sizeof(float) * numel;
        float* temp = (float*) malloc(bytes);
        const int N = 36;
        char buf[N];
        for (int i = 0; i < numel; i++)
        {
            fgets(buf, N, fp);
            float value = atof(buf);
            temp[i] = value;
        }
        this->set_data_fp32(temp);
        free(temp);
        temp = nullptr;
    }
    else if (dtype == INT32)
    {
        bytes = sizeof(int) * numel;
        int* temp = (int*) malloc(bytes);
        const int N = 36;
        char buf[N];
        for (int i = 0; i < numel; i++)
        {
            fgets(buf, N, fp);
            int value = atoi(buf);
            temp[i] = value;
        }
        this->set_data_int32(temp);
        free(temp);
        temp = nullptr;
    }
    fclose(fp);
}

void Tensor::set_data_fp32(float* new_data)
{
    // data_fp32 如果是从 MemoryAllocator 申请的，那么data_fp32其实是 MemoryAllocator 的 mem_fp32 加上1个偏移的指针。
    // 此时用 memcpy() 给 data_fp32 赋值会报错。
//    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
//    for (int i = 0; i < numel; i++)
//    {
//        *(data_fp32 + i) = new_data[i];
//    }

#if BACKEND_X86
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;
    const int elempack = 8;
    const int num_packs = numel / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        const float* x_ptr = new_data + pid * elempack;
        float* y_ptr = data_fp32 + pid * elempack;
        __m256 _x = _mm256_loadu_ps(x_ptr);
        _mm256_storeu_ps(y_ptr, _x);
    }
    int offset_ = num_packs * elempack;
    if (numel - offset_ >= 4)
    {
        const float* x_ptr = new_data + offset_;
        float* y_ptr = data_fp32 + offset_;
        __m128 _x = _mm_load_ps(x_ptr);
        _mm_store_ps(y_ptr, _x);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < numel; i++) {
        data_fp32[i] = new_data[i];
    }
#endif // BACKEND_X86

#if BACKEND_ARM
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        *(data_fp32 + i) = new_data[i];
    }
#endif // BACKEND_ARM

}

void Tensor::set_data_fp32(float val)
{
//    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
//    for (int i = 0; i < numel; i++)
//    {
//        *(data_fp32 + i) = val;
//    }

#if BACKEND_X86
    Config* cfg = Config::getInstance();
    const int num_threads_ = cfg->num_threads;
    const int elempack = 8;
    const int num_packs = numel / elempack;
    #pragma omp parallel for num_threads(num_threads_)
    for (int pid = 0; pid < num_packs; pid++) {
        float* y_ptr = data_fp32 + pid * elempack;
        __m256 _x = _mm256_broadcast_ss(&val);
        _mm256_storeu_ps(y_ptr, _x);
    }
    int offset_ = num_packs * elempack;
    if (numel - offset_ >= 4)
    {
        float* y_ptr = data_fp32 + offset_;
        __m128 _x = _mm_broadcast_ss(&val);
        _mm_store_ps(y_ptr, _x);
        offset_ += 4;
    }
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = offset_; i < numel; i++) {
        data_fp32[i] = val;
    }
#endif // BACKEND_X86

#if BACKEND_ARM
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        *(data_fp32 + i) = val;
    }
#endif // BACKEND_ARM

}

void Tensor::set_data_int32(int* new_data)
{
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        *(data_int32 + i) = new_data[i];
    }
}

void Tensor::set_data_int32(int val)
{
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        *(data_int32 + i) = val;
    }
}

void Tensor::normal_init(float mean, float std, int seed)
{
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(mean, std);
    for (int i = 0; i < numel; i++)
    {
        float val = distribution(generator);
        *(data_fp32 + i) = val;
    }
}

void Tensor::save_as_bin(char* name)
{
    FILE* bin_fp = fopen(name, "wb");
    // float值转char值
    unsigned char* param_bytes = new unsigned char[sizeof(float) * numel];
    memcpy(&param_bytes[0], data_fp32, sizeof(float) * numel);  // 注意，要传入 &param_bytes[0] 而不是 &param_bytes

    // 从第start_bytes_i个char值开始，读ele_bytes * numel个char值
    fwrite(param_bytes, sizeof(float) * numel, 1, bin_fp);

    fclose(bin_fp);
}

float* Tensor::get_data_fp32()
{
/*
返回的是副本！！！记得释放返回的副本
remenber to do these:

free(temp);
temp = nullptr;
*/
    const int bytes = sizeof(float) * numel;
    float* temp = (float*) malloc(bytes);
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        temp[i] = *(data_fp32 + i);
    }
    return temp;
}

int* Tensor::get_data_int32()
{
/*
返回的是副本！！！记得释放返回的副本
remenber to do these:

free(temp);
temp = nullptr;
*/
    const int bytes = sizeof(int) * numel;
    int* temp = (int*) malloc(bytes);
    #pragma omp parallel for num_threads(Config::getInstance()->num_threads)
    for (int i = 0; i < numel; i++)
    {
        temp[i] = *(data_int32 + i);
    }
    return temp;
}

void Tensor::print_data(int max_num)
{
    int numel_ = numel;
    if (max_num > -1)
    {
        numel_ = std::min(numel, max_num);
    }
    printf("data=");
    if (dtype == FP32)
    {
        for (int i = 0; i < numel_; i++) {
//            printf("%e, ", temp[i]);
            printf("%f, ", data_fp32[i]);
        }
    }
    else if (dtype == INT32)
    {
        for (int i = 0; i < numel_; i++) {
            printf("%d, ", data_int32[i]);
        }
    }
    printf("\n");
}

void Tensor::reshape(std::vector<int>* shape)
{
    int numel2 = 1;
    int neg1_pos = -1;
    int neg1_num = 0;
    for (int i = 0; i < shape->size(); i++)
    {
        if (shape->at(i) == 0 || shape->at(i) < -1)
        {
            printf("Error From Tensor.reshape(), shape invalid.\n");
            exit(1);
        }
        if (shape->at(i) == -1)
        {
            neg1_num++;
            neg1_pos = i;
        }
        else
        {
            numel2 *= shape->at(i);
        }
    }
    if (neg1_num == 0)
    {
        if (numel2 != numel)
        {
            printf("Error From Tensor.reshape(), numel2 != numel, shape invalid.\n");
            exit(1);
        }
        delete this->shape;
        this->shape = shape;
        dims = this->shape->size();
    }
    else if (neg1_num == 1)
    {
        int a = numel / numel2;
        shape->at(neg1_pos) = a;
        delete this->shape;
        this->shape = shape;
        dims = this->shape->size();
    }
    else
    {
        printf("multi -1, shape invalid.");
        exit(1);
    }
}

void Tensor::restore_shape()
{
    this->reshape(MMSHAPE4D(ori_D0, ori_D1, ori_D2, ori_D3));
//    if (dims == 4)
//    {
//        for (int i = shape->size() - 1; i >= 0; i--)
//        {
//            shape->erase(shape->begin() + i);
//        }
//        shape->push_back(ori_D0);
//        shape->push_back(ori_D1);
//        shape->push_back(ori_D2);
//        shape->push_back(ori_D3);
//        dims = shape->size();
//    }
}

void Tensor::squeeze(int dim)
{
    if (shape->at(dim) != 1)
    {
        printf("squeeze failed, shape[dim] != 1.");
        exit(1);
    }
    else
    {
        shape->erase(shape->begin() + dim);   // delete the dim-th element.
        dims = shape->size();
    }
}

void Tensor::unsqueeze(int dim)
{
    shape->insert(shape->begin() + dim, 1);
    dims = shape->size();
}

void Tensor::print_msg(char* name)
{
    if (!shape)
    {
        printf("Tensor \'%s\' msg: ", name);
        printf("id=%d, ", id);
        printf("dims=%d, ", dims);
        printf("numel=%d, ", numel);
        printf("shape is nullptr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        printf("\n");
        exit(1);
    }
    printf("Tensor \'%s\' msg: ", name);
    printf("id=%d, ", id);
    printf("dims=%d, ", dims);
    printf("numel=%d, ", numel);
    printf("shape=(");
    for (int ILoveC = 0; ILoveC < shape->size(); ILoveC++) {
        printf("%d, ", shape->at(ILoveC));
    }
    printf("), ");
    printf("is_param=%d, ", is_param);
    printf("times_as_input=%d, ", times_as_input);
    printf("referenceCount=%d, ", referenceCount);
    printf("\n");
}

std::vector<int>* Tensor::clone_shape()
{
    std::vector<int>* shape2 = new std::vector<int>;
    for (int i = 0; i < this->shape->size(); i++)
    {
        shape2->push_back(this->shape->at(i));
    }
    return shape2;
}


NS_MM_END
