#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include "../miemienet/miemienet.h"
#include "../miemiedet/miemiedet.h"

using namespace miemienet;
using namespace miemiedet;

float calc_diff(float* x, float* y, int numel)
{
    float diff = 0.f;
    float M = 1.f;
//    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}


class Model : public Layer
{
public:
    Model(int in_features, int out_features, int kernel_size, int stride, int padding, bool use_bias=true, int groups=1)
    {
        conv = new SNT Conv2d(in_features, out_features, kernel_size, stride, padding, 1, groups, use_bias);
        register_sublayer("conv", conv);
//        act = new SNT Activation("leakyrelu", 0.33, true);
//        register_sublayer("act", act);
    }
    ~Model()
    {
        delete conv;
//        delete act;
    }

    Conv2d* conv;
//    Activation* act;

    virtual Tensor* create_tensors(Tensor* x)
    {
        Tensor* y = conv->create_tensors(x);
//        y = act->create_tensors(y);
        return y;
    }

    virtual Tensor* feed_forward(Tensor* x)
    {
        Tensor* y = conv->feed_forward(x);
//        y = act->feed_forward(y);
        return y;
    }
private:
};




int main(int argc, char** argv)
{
/*
python build.py --platform LINUX --cxx g++ --backend BACKEND_X86 --exec_file test2_00004_conv2d

./test2_00004_conv2d.out


*/
#if defined(WINDOWS)
    printf("%s\n", "WINDOWS");
#endif
#if defined(LINUX)
    printf("%s\n", "LINUX");
#endif
    printf("%s\n", miemienet::miemienetVersion());

    // 修改Config全局配置时，要在网络创建之前修改。
    Config* cfg = Config::getInstance();
    printf("num_threads=%d\n", cfg->num_threads);
    // 图片张量设置为以NHWC顺序排列。这样对cpu计算更友好。
    // image_data_format会影响卷积层，全连接层的权重的排列方式。
//    cfg->image_data_format = NCHW;
    cfg->image_data_format = NHWC;
    cfg->use_cpp_compute = true;
    cfg->fuse_conv_bn = true;

    const int num_threads_ = cfg->num_threads;

    char* test_name = "00004";

//    int batch_size = 8;
//    int input_size = 256;
//    int in_features = 32;
//    int out_features = 32;
//    int kernel_size = 3;
//    int groups = 1;

    int batch_size = 8;
    int input_size = 256;
    int in_features = 32;
    int out_features = 32;
    int kernel_size = 3;
    int groups = 32;

    int stride = 1;
    int padding = (kernel_size - 1) / 2;
//    bool bias = false;
    bool bias = true;

    char file_name[256];

    Model* model = new SNT Model(in_features, out_features, kernel_size, stride, padding, bias, groups);

    std::vector<char*>* param_names = new std::vector<char*>;
    std::vector<Tensor*>* params = new std::vector<Tensor*>;
    model->named_parameters(param_names, params);
    for (int i = 0; i < param_names->size(); i++)
    {
        printf("param_names[%d]=%s\n", i, param_names->at(i));
    }

    model->eval();

    Tensor* x;
    Tensor* y;
    if (Config::getInstance()->image_data_format == NCHW)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, in_features, input_size, input_size), FP32, false, false);
    }
    else if (Config::getInstance()->image_data_format == NHWC)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, input_size, input_size, in_features), FP32, false, false);
    }

    printf("======================== init ========================\n");
    x->normal_init(0.f, 1.f, 0);
    model->conv->weight->normal_init(0.f, 1.f, 1);
    model->conv->bias->normal_init(0.f, 1.f, 2);

    // 建立计算图，初始化所有中间张量，给所有中间张量分配内存。
    y = model->create_tensors(x);

    model->conv->weight->print_msg("model->conv->weight");
    model->conv->weight->print_data(30);
    model->conv->bias->print_msg("model->conv->bias");
    model->conv->bias->print_data(30);

    sprintf(file_name, "test/save_data/%s-x.bin", test_name);
    x->save_as_bin(file_name);
    sprintf(file_name, "test/save_data/%s-w.bin", test_name);
    model->conv->weight->save_as_bin(file_name);
    sprintf(file_name, "test/save_data/%s-b.bin", test_name);
    model->conv->bias->save_as_bin(file_name);


    printf("======================== eval ========================\n");
    for (int batch_idx = 0; batch_idx < 20; batch_idx++)
    {
        printf("======================== batch_%.3d ========================\n", batch_idx);
        x->print_msg("x");
        x->print_data(30);


        auto startTime = std::chrono::system_clock::now();

        y = model->feed_forward(x);


        auto endTime = std::chrono::system_clock::now();
        // 1秒=1000毫秒=1000,000微秒
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("eval forward cost_time = %f ms\n", cost_ms);

        y->print_msg("y");
        y->print_data(30);


        if (batch_idx == 0)
        {
            sprintf(file_name, "test/save_data/%s-y.bin", test_name);
            y->save_as_bin(file_name);
        }
    }
    delete model;

    // 倒序遍历，可以边遍历边删除
    for (int i = param_names->size() - 1; i >= 0; i--)
    {
        char* param_name = param_names->at(i);
        delete param_name;
        param_names->erase(param_names->begin() + i);
    }
    delete param_names;

    Config::getInstance()->destroyInstance();
    MemoryAllocator::getInstance()->destroyInstance();
    TensorIdManager::getInstance()->destroyInstance();

    return 0;
}