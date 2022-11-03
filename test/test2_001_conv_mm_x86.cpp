#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../miemienet/miemienet.h"
//#include "miemiedet.h"

using namespace miemienet;
//using namespace miemiedet;

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

class Res : public Layer
{
public:
    Res(int in_features, int out_features, int kernel_size, int stride, int padding, bool use_bias=true, int groups=1)
    {
        fc = new SNT Conv2d(in_features, out_features, kernel_size, stride, padding, 1, groups, use_bias);
        register_sublayer("fc", fc);
//        bn = new SNT BatchNorm2d(out_features, 1e-5, 0.1f, true, true);
//        register_sublayer("bn", bn);
        act = new SNT Activation("leakyrelu", 0.33, true);
        register_sublayer("act", act);
    }
    ~Res()
    {
        delete fc;
//        delete bn;
        delete act;
    }

    Conv2d* fc;
//    BatchNorm2d* bn;
    Activation* act;

    virtual Tensor* create_tensors(Tensor* x)
    {
        Tensor* y = fc->create_tensors(x);
//        y = bn->create_tensors(y);
        y = act->create_tensors(y);
        return y;
    }

    virtual Tensor* feed_forward(Tensor* x)
    {
        Tensor* y = fc->feed_forward(x);
//        y = bn->feed_forward(y);
        y = act->feed_forward(y);
        miemienet::functional::activation(y, y, "sigmoid", 0.f);
        miemienet::functional::elementwise(y, x, y, ELE_ADD);
        return y;
    }
private:
};

class Model : public Layer
{
public:
    Model(int in_features, int out_features, int num_classes, bool use_bias=true)
    {
        res0 = new SNT Res(in_features, out_features, 3, 1, 1, use_bias, out_features);
        register_sublayer("res0", res0);
//        avgpool = new SNT AvgPool2d(2, 2, 0, true);
//        register_sublayer("avgpool", avgpool);
        maxpool = new SNT MaxPool2d(3, 2, 1, false);
        register_sublayer("maxpool", maxpool);
        res1 = new SNT Res(out_features, out_features, 1, 1, 0, use_bias, 1);
        register_sublayer("res1", res1);
        softmax = new SNT Softmax(3, true);
        register_sublayer("softmax", softmax);
        transpose = new SNT Transpose(TRANS4D_0213);
        register_sublayer("transpose", transpose);
        concat = new SNT Concat(-1);
        register_sublayer("concat", concat);
        interp = new SNT Interp(0, 0, 2.f, 2.f);
        register_sublayer("interp", interp);
        reduce = new SNT Reduce(MMSHAPE2D(1, 2), true, RED_MEAN);
        register_sublayer("reduce", reduce);
//        fc = new SNT Linear(out_features, num_classes, true);
//        register_sublayer("fc", fc);
    }
    ~Model()
    {
        delete res0;
//        delete avgpool;
        delete maxpool;
        delete res1;
        delete softmax;
        delete transpose;
        delete concat;
        delete interp;
        delete reduce;
//        delete fc;
    }

    Res* res0;
//    AvgPool2d* avgpool;
    MaxPool2d* maxpool;
    Res* res1;
    Softmax* softmax;
    Transpose* transpose;
    Concat* concat;
    Interp* interp;
    Reduce* reduce;
//    Linear* fc;

    virtual Tensor* create_tensors(Tensor* x)
    {
        Tensor* y = res0->create_tensors(x);
//        y = avgpool->create_tensors(y);
        y = maxpool->create_tensors(y);
//        y = FCI->maxpool2d(y, create_graph, 3, 2, 1, false);
        y = res1->create_tensors(y);
        y = softmax->create_tensors(y);
        Tensor* aaa = transpose->create_tensors(y);
        y = concat->create_tensors(aaa, y, aaa, y);
        y = interp->create_tensors(y);
        y = reduce->create_tensors(y);
//        y = fc->create_tensors(y);
        return y;
    }

    virtual Tensor* feed_forward(Tensor* x)
    {
        Tensor* y = res0->feed_forward(x);
//        y = avgpool->feed_forward(y);
        y = maxpool->feed_forward(y);
//        y = FCI->maxpool2d(y, create_graph, 3, 2, 1, false);
        y = res1->feed_forward(y);
//        y = softmax->feed_forward(y);
//        Tensor* aaa = transpose->feed_forward(y);
//        y = concat->feed_forward(aaa, y, aaa, y);
//        y = interp->feed_forward(y);
//        y = reduce->feed_forward(y);
//        y = y->mean(MMSHAPE2D(2, 3), false, create_graph);
//        y = fc->feed_forward(y);
        return y;
    }
private:
};



int main(int argc, char** argv)
{
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
    // 申请太多会报错: 段错误 (核心已转储)
    cfg->mem_fp32_size = 125999999 * 2;
    cfg->mem_int32_size = 65999999 * 4;

    const int num_threads_ = cfg->num_threads;

    char* test_name = "001";
    int batch_size = 8;
    int in_features = 32;
    int out_features = 32;
    int num_classes = 5;
//    bool bias = false;
    bool bias = true;

    char file_name[256];

    Model* model = new SNT Model(in_features, out_features, num_classes, bias);

    sprintf(file_name, "test/save_data/%s-modelfinal", test_name);
    model->load_state_dict(file_name);

    std::vector<char*>* param_names = new std::vector<char*>;
    std::vector<Tensor*>* params = new std::vector<Tensor*>;
    model->named_parameters(param_names, params);
    for (int i = 0; i < param_names->size(); i++)
    {
        printf("param_names[%d]=%s\n", i, param_names->at(i));
    }

//    int input_size = 32;
//    int input_size = 2;
    int input_size = 64;
//    int input_size = 640;

    model->eval();

    Tensor* x;
    Tensor* y_torch;
    Tensor* y;
    if (Config::getInstance()->image_data_format == NCHW)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, in_features, input_size, input_size), FP32, false, false);
        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, out_features, input_size, input_size), FP32, false, false);
    }
    else if (Config::getInstance()->image_data_format == NHWC)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, input_size, input_size, in_features), FP32, false, false);
//        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, input_size, input_size, out_features), FP32, false, false);
        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, input_size / 2, input_size / 2, out_features), FP32, false, false);
//        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, input_size / 2, input_size / 2, out_features*4), FP32, false, false);
//        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, input_size, input_size, out_features*4), FP32, false, false);
//        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, 1, 1, out_features*4), FP32, false, false);
//        y_torch = new SNT Tensor(MMSHAPE4D(batch_size, input_size / 4, input_size / 4, out_features), FP32, false, false);
    }

    // 建立计算图，初始化所有中间张量，给所有中间张量分配内存。
    y = model->create_tensors(x);

    printf("======================== eval ========================\n");
    for (int batch_idx = 0; batch_idx < 2; batch_idx++)
    {
        printf("======================== batch_%.3d ========================\n", batch_idx);
        sprintf(file_name, "test/save_data/%s-eval-x.txt", test_name);
        x->load_from_txt(file_name);
//        x->set_data_fp32(im_ptr);
        sprintf(file_name, "test/save_data/%s-eval-y.txt", test_name);
        y_torch->load_from_txt(file_name);
        x->print_msg("x");
        x->print_data(12);
        y_torch->print_msg("y_torch");
        y_torch->print_data(12);


        auto startTime = std::chrono::system_clock::now();

        y = model->feed_forward(x);


        auto endTime = std::chrono::system_clock::now();
        // 1秒=1000毫秒=1000,000微秒
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("eval forward cost_time = %f ms\n", cost_ms);

        y->print_msg("y");
        y->print_data(12);

        float* _y = y->get_data_fp32();
        float* _y_torch = y_torch->get_data_fp32();
        float diff = calc_diff(_y, _y_torch, y->numel);
        free(_y);
        free(_y_torch);
        printf("diff=%f (%s)\n", diff, "y");
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

    // y_torch没有参与前向计算图，不在神经网络层内，需要手动释放。
    delete y_torch;

//    free(im_ptr);
//    im_ptr = nullptr;

    Config::getInstance()->destroyInstance();
    MemoryAllocator::getInstance()->destroyInstance();
    TensorIdManager::getInstance()->destroyInstance();

    return 0;
}