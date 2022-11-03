#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer.h"

NS_MM_BEGIN

Layer::Layer()
{
    this->inplace = false;
    this->training = true;
    this->input_tensors = new std::vector<Tensor*>;
    this->temp_tensors = new std::vector<Tensor*>;
    this->temp2_tensors = new std::vector<Tensor*>;
    this->output_tensors = new std::vector<Tensor*>;
}

Layer::~Layer()
{
    // 倒序遍历，可以边遍历边删除
    for (int i = input_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = input_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        input_tensors->erase(input_tensors->begin() + i);
    }
    delete input_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = temp_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = temp_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        temp_tensors->erase(temp_tensors->begin() + i);
    }
    delete temp_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = temp2_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = temp2_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        temp2_tensors->erase(temp2_tensors->begin() + i);
    }
    delete temp2_tensors;
    // 倒序遍历，可以边遍历边删除
    for (int i = output_tensors->size() - 1; i >= 0; i--)
    {
        Tensor* tensor = output_tensors->at(i);
        if (tensor != nullptr)
        {
            tensor->referenceCount--;
            if (tensor->referenceCount <= 0){
                delete tensor;
            }
        }
        output_tensors->erase(output_tensors->begin() + i);
    }
    delete output_tensors;
}

void Layer::register_sublayer(char* name, Layer* layer)
{
    this->sublayer_names.push_back(name);
    this->sublayers.push_back(layer);
}

void Layer::train()
{
    training = true;
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        this->sublayers[i]->train();
    }
}

void Layer::eval()
{
    training = false;
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        this->sublayers[i]->eval();
    }
}

Tensor* Layer::register_buffer(char* name, std::vector<int>* shape, int dtype, bool init, float init_value)
{
    Tensor* buffer = new SNT Tensor(shape, dtype, true, init, init_value);
    buffer->is_buffer = true;       // layer->requires_grad_(bool)时，对buffer张量无效。
    this->params.push_back(buffer);
    this->param_names.push_back(name);
    return buffer;
}

Tensor* Layer::create_parameter(char* name, std::vector<int>* shape, int dtype, bool init, float init_value)
{
    Tensor* param = new SNT Tensor(shape, dtype, true, init, init_value);
    this->params.push_back(param);
    this->param_names.push_back(name);
    return param;
}

void Layer::parameters(std::vector<Tensor*>* params)
{
    for (int i = 0; i < this->params.size(); i++)
    {
        params->push_back(this->params[i]);
    }
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        this->sublayers[i]->parameters(params);
    }
}

void Layer::named_parameters(std::vector<char*>* param_names, std::vector<Tensor*>* params, char* prefix)
{
    for (int i = 0; i < this->params.size(); i++)
    {
        if (prefix)
        {
            char* param_name = new char[256];
            strcpy(param_name, prefix);
            strcat(param_name, ".");
            strcat(param_name, this->param_names[i]);
            param_names->push_back(param_name);
        }
        else
        {
            param_names->push_back(this->param_names[i]);
        }
        params->push_back(this->params[i]);
    }
    for (int i = 0; i < this->sublayers.size(); i++)
    {
        char* sub_prefix = new char[256];
        if (prefix)
        {
            strcpy(sub_prefix, prefix);
            strcat(sub_prefix, ".");
            strcat(sub_prefix, this->sublayer_names[i]);
            this->sublayers[i]->named_parameters(param_names, params, sub_prefix);
        }
        else
        {
            strcpy(sub_prefix, this->sublayer_names[i]);
            this->sublayers[i]->named_parameters(param_names, params, sub_prefix);
        }
        delete sub_prefix;
    }
}

void Layer::load_state_dict(char* name, char* param_prefix_name)
{
    char model_file[256];   // 离开变量的作用域时释放，也就是这个函数。
    sprintf(model_file, "%s.mie", name);
    FILE* fp = fopen(model_file, "r");
    if (!fp)
    {
        printf("file %s not exist.\n", model_file);
        exit(1);
    }

    std::vector<char*>* my_param_names = new std::vector<char*>;
    std::vector<Tensor*>* my_params = new std::vector<Tensor*>;
    this->named_parameters(my_param_names, my_params);

    const int N = 512;
    char buf[N];
    fgets(buf, N, fp);
    int param_num = atoi(buf);
    char param_name[256];
    char start_bytes_i_str[256];
    char ele_bytes_str[256];
    char numel_str[256];
    int start_bytes_i;
    int ele_bytes;
    int numel;
    for (int i = 0; i < param_num; i++)
    {
        fgets(buf, N, fp);
//        printf("%s", buf);
//        printf("%d\n", strlen(buf));
        int comma_num = 0;   // 当前行读到的逗号的数量
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        if (param_prefix_name)
        {
            for (int j = 0; j < strlen(param_prefix_name); j++)
            {
                param_name[i0++] = param_prefix_name[j];
            }
        }
        for (int j = 0; j < strlen(buf) - 1; j++)  // 最后1个是换行符，不要读
        {
            if (buf[j] == ',')
            {
                comma_num++;
                continue;
            }
            if (comma_num == 0)
            {
                param_name[i0++] = buf[j];
            }
            else if (comma_num == 1)
            {
                start_bytes_i_str[i1++] = buf[j];
            }
            else if (comma_num == 2)
            {
                ele_bytes_str[i2++] = buf[j];
            }
            else if (comma_num == 3)
            {
                numel_str[i3++] = buf[j];
            }
        }
        param_name[i0] = '\0';
        start_bytes_i_str[i1] = '\0';
        ele_bytes_str[i2] = '\0';
        numel_str[i3] = '\0';
        start_bytes_i = atoi(start_bytes_i_str);
        ele_bytes = atoi(ele_bytes_str);
        numel = atoi(numel_str);
//        printf("%s\n", param_name);
//        printf("%d\n", start_bytes_i);
//        printf("%d\n", ele_bytes);
//        printf("%d\n", numel);
//        printf("\n\n");

        // 读权重
        // float和4字节char互转，都是用memcpy()实现。
        sprintf(model_file, "%s.bin", name);
        FILE* bin_fp = fopen(model_file, "rb");
        if (!bin_fp)
        {
            printf("file %s not exist.\n", model_file);
            exit(1);
        }
        // 现在只支持fp32, 1个float占4字节
        float* param = new float[numel];

        // 把前start_bytes_i个char值读掉，不处理。(把前 start_bytes_i / 4 个float值读掉，不处理。)
        if (start_bytes_i > 0)
        {
            unsigned char* param_bytes = new unsigned char[start_bytes_i];
            fread(param_bytes, start_bytes_i, 1, bin_fp);
            delete param_bytes;
        }

        // 从第start_bytes_i个char值开始，读ele_bytes * numel个char值
        unsigned char* param_bytes = new unsigned char[ele_bytes * numel];
        fread(param_bytes, ele_bytes * numel, 1, bin_fp);
        // char值转float值
        float* values = new float[numel];
        memcpy(&values[0], param_bytes, ele_bytes * numel);  // 注意，要传入 &values[0] 而不是 &values
//        for (int j = 0; j < numel; j++)
//        {
//            printf("%f, ", values[j]);
//        }
//        printf("\n");

        int target_j = -1;
        for (int j = 0; j < my_param_names->size(); j++)
        {
            if (strcmp(param_name, my_param_names->at(j)) == 0)
            {
                target_j = j;
                break;
            }
        }
        if (target_j < 0)
        {
            printf("Warning! Checkpont Param name \'%s\' not found! miemienet would not load this param.\n", param_name);
        }
        else
        {
            Tensor* mpr = my_params->at(target_j);
            if (mpr->numel != numel)
            {
                printf("Warning! Checkpont Param name \'%s\' \'s numel(%d) not equal the numel(%d) in checkpoint! miemienet would not load this param.\n", param_name, mpr->numel, numel);
            }
            else
            {
                mpr->set_data_fp32(values);
            }
        }

        delete param_bytes;
        delete values;
        fclose(bin_fp);
    }
    fclose(fp);

    // 倒序遍历，可以边遍历边删除
    for (int i = my_param_names->size() - 1; i >= 0; i--)
    {
        char* param_name = my_param_names->at(i);
        delete param_name;
        my_param_names->erase(my_param_names->begin() + i);
    }
    delete my_param_names;
    delete my_params;
}

NS_MM_END
