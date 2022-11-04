#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <immintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

float calc_diff(float* x, float* y, int numel)
{
    float diff = 0.f;
//    float M = 1.f;
    float M = 1.f / (float)numel;
    for (int i = 0; i < numel; i++)
    {
        diff += (x[i] - y[i]) * (x[i] - y[i]) * M;
    }
    return diff;
}


int main(int argc, char** argv)
{
/*
编译示例时为了避免
undefined reference to `cv::imread(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int)'
undefined reference to `cv::imread(std::string const&, int)'
这些恶灵缠身，
Ubuntu安装opencv-3.3.1版本
https://blog.csdn.net/CynalFly/article/details/126784079


sudo apt-get install build-essential -y
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev -y
sudo apt-get install libjasper-dev -y



cd ~/opencv-3.3.1
mkdir build
cd build
cmake -D BUILD_TIFF=ON -D BUILD_TESTS=OFF ..
cmake --build . --config Release -j 2
sudo cmake --build . --config Release --target install


error: ‘CODEC_FLAG_GLOBAL_HEADER’ was not declared in this scope
https://www.cnblogs.com/codeit/p/15748619.html

修改~/opencv-3.3.1/modules/videoio/src/cap_ffmpeg_impl.hpp，顶端添加：
#define AV_CODEC_FLAG_GLOBAL_HEADER (1 << 22)
#define CODEC_FLAG_GLOBAL_HEADER AV_CODEC_FLAG_GLOBAL_HEADER
#define AVFMT_RAWPICTURE 0x0020


jiancha:
pkg-config --modversion opencv

3.3.1



g++ test/test2_00000_opencv.cpp -fopenmp -march=native -o test2_00000_opencv.out -w `pkg-config --cflags --libs opencv`


./test2_00000_opencv.out test/000000000019.jpg



VS2022配置OpenCV操作步骤：
https://blog.csdn.net/Learning_Well/article/details/125232288

Path后接
D:\opencv\build\x64\vc15\bin
“通用属性-VC++目录-常规-包含目录”中添加
D:\opencv\build\include\opencv2
D:\opencv\build\include
“通用属性-VC++目录-常规-库目录”中添加
D:\opencv\build\x64\vc15\lib
“通用属性-链接器-输入-附加依赖项”中添加
D:\opencv\build\x64\vc15\lib\opencv_world455d.lib


如果要编译Release版本miemienet，Release的配置里和Debug的配置以下不同：
“通用属性-链接器-输入-附加依赖项”中添加
D:\opencv\build\x64\vc15\lib\opencv_world455.lib

opencv_world455.lib 才是给 Release版本 链接的，真的好坑啊。

*/
    const char* imagepath = argv[1];
    cv::Mat im_bgr = cv::imread(imagepath, 1);   // HWC顺序排列。通道默认是BGR排列。
    if (im_bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    cv::Mat im_rgb;
    cv::cvtColor(im_bgr, im_rgb, cv::COLOR_BGR2RGB);

    cv::Mat im_resize;
    cv::resize(im_rgb, im_resize, cv::Size(640, 640), 0, 0, cv::INTER_CUBIC);

    im_resize.convertTo(im_resize, 5);  // 转float32

    int HW = im_resize.total();
    int C = im_resize.channels();
    int HWC = HW*C;
    printf("HW=%d\n", HW);
    printf("C=%d\n", C);
    printf("HWC=%d\n", HWC);
    float* im_ptr = new float[HWC];
    memcpy(im_ptr, im_resize.ptr<float>(0), HWC * sizeof(float));
    for (int i = 0; i < 15; i++) {
        printf("%f, ", im_ptr[i]);
    }




    cv::imshow("aaaaa", im_resize);
    cv::waitKey(0);

    return 0;
}