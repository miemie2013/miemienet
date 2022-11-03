#include <stdio.h>
#include "config.h"

NS_MM_BEGIN

Config* Config::s_singleInstance = nullptr;

Config* Config::getInstance()
{
    if (s_singleInstance == nullptr)
    {
        s_singleInstance = new (std::nothrow) Config();
    }
    return s_singleInstance;
}

void Config::destroyInstance()
{
    delete s_singleInstance;
    s_singleInstance = nullptr;
}

Config::Config()
{
/*
修改Config全局配置时，要在网络创建之前修改。
当使用x86、naive作为后端时，图片张量设置为以 NHWC 顺序排列。这样对cpu计算更友好。
image_data_format会影响卷积层，全连接层的权重的排列方式。
当 image_data_format == NCHW, 卷积层weight形状是[out_C, in_C, kH, kW], 全连接层weight形状是[out_C, in_C]  (和pytorch一样);
当 image_data_format == NHWC, 卷积层weight形状是[kH, kW, in_C, out_C], 全连接层weight形状是[in_C, out_C]  (和tensorflow 1.x一样);

*/
    this->num_threads = 12;
    this->use_cpp_compute = true;
    this->fuse_conv_bn = false;   // 合并卷积层和bn层。在训练阶段，不要打开这个开关。推理部署时打开，此时不会创建bn层。
    this->image_data_format = NCHW;
}

Config::~Config()
{
}


NS_MM_END
