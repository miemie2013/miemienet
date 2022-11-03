#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <vector>
#include "../macros.h"

NS_MM_BEGIN

class Config
{
public:
    static Config* getInstance();
    static void destroyInstance();
    int num_threads;
    bool use_cpp_compute;
    bool fuse_conv_bn;
    int image_data_format;  // NCHW or NHWC
private:
    Config();
    ~Config();
    static Config* s_singleInstance;
};

NS_MM_END

#endif // __CONFIG_H__
