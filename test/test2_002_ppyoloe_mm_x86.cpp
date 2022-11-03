#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

//struct Object
//{
//    float x0;
//    float y0;
//    float x1;
//    float y1;
//    int clsid;
//    float score;
//    int label;
//    float prob;
//};


static void generate_ppyoloe_proposals(Tensor* cls_score, Tensor* reg_dist, float scale_x, float scale_y, float prob_threshold, std::vector<Object>& objects)
{
    // python中cls_score的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    // python中reg_dist 的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
    const int N = cls_score->shape->at(0);
    const int num_grid = cls_score->shape->at(1);
    const int num_class = cls_score->shape->at(2);

    // 最大感受野输出的特征图一行（一列）的格子数stride32_grid。设为G，则
    // G*G + (2*G)*(2*G) + (4*G)*(4*G) = 21*G^2 = num_grid
    // 所以G = sqrt(num_grid/21)
    int stride32_grid = sqrt(num_grid / 21);
    int stride16_grid = stride32_grid * 2;
    int stride8_grid = stride32_grid * 4;

    const float* cls_score_ptr = cls_score->data_fp32;
    const float* reg_dist_ptr = reg_dist->data_fp32;

    // stride==32的格子结束的位置
    int stride32_end = stride32_grid * stride32_grid;
    // stride==16的格子结束的位置
    int stride16_end = stride32_grid * stride32_grid * 5;
    for (int anchor_idx = 0; anchor_idx < num_grid; anchor_idx++)
    {
        float stride = 32.0f;
        int row_i = 0;
        int col_i = 0;
        if (anchor_idx < stride32_end) {
            stride = 32.0f;
            row_i = anchor_idx / stride32_grid;
            col_i = anchor_idx % stride32_grid;
        }else if (anchor_idx < stride16_end) {
            stride = 16.0f;
            row_i = (anchor_idx - stride32_end) / stride16_grid;
            col_i = (anchor_idx - stride32_end) % stride16_grid;
        }else {  // stride == 8
            stride = 8.0f;
            row_i = (anchor_idx - stride16_end) / stride8_grid;
            col_i = (anchor_idx - stride16_end) % stride8_grid;
        }
        float x_center = 0.5f + (float)col_i;
        float y_center = 0.5f + (float)row_i;
        float x0 = x_center - reg_dist_ptr[0];
        float y0 = y_center - reg_dist_ptr[1];
        float x1 = x_center + reg_dist_ptr[2];
        float y1 = y_center + reg_dist_ptr[3];
        x0 = x0 * stride / scale_x;
        y0 = y0 * stride / scale_y;
        x1 = x1 * stride / scale_x;
        y1 = y1 * stride / scale_y;
        float h = y1 - y0;
        float w = x1 - x0;

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_prob = cls_score_ptr[class_idx];
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        cls_score_ptr += num_class;
        reg_dist_ptr += 4;
    }
}

static void generate_picodet_proposals(Tensor* cls_score, Tensor* reg_dist, int input_size, float scale_x, float scale_y, float prob_threshold, std::vector<Object>& objects)
{
    // python中cls_score的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    // python中reg_dist 的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
    const int N = cls_score->shape->at(0);
    const int num_grid = cls_score->shape->at(1);
    const int num_class = cls_score->shape->at(2);

    // 每个感受野输出的特征图一行（一列）的格子数
    int stride8_grid = input_size / 8;
    int stride16_grid = (int)((stride8_grid + 1) / 2);
    int stride32_grid = (int)((stride16_grid + 1) / 2);
    int stride64_grid = (int)((stride32_grid + 1) / 2);

    const float* cls_score_ptr = cls_score->data_fp32;
    const float* reg_dist_ptr = reg_dist->data_fp32;

    // stride==8的格子结束的位置
    int stride8_end = stride8_grid * stride8_grid;
    // stride==16的格子结束的位置
    int stride16_end = stride8_end + stride16_grid * stride16_grid;
    // stride==32的格子结束的位置
    int stride32_end = stride16_end + stride32_grid * stride32_grid;
    // stride==64的格子结束的位置
    int stride64_end = stride32_end + stride64_grid * stride64_grid;

    for (int anchor_idx = 0; anchor_idx < num_grid; anchor_idx++)
    {
        float stride = 32.0f;
        int row_i = 0;
        int col_i = 0;
        if (anchor_idx < stride8_end) {
            stride = 8.0f;
            row_i = anchor_idx / stride8_grid;
            col_i = anchor_idx % stride8_grid;
        }else if (anchor_idx < stride16_end) {
            stride = 16.0f;
            row_i = (anchor_idx - stride8_end) / stride16_grid;
            col_i = (anchor_idx - stride8_end) % stride16_grid;
        }else if (anchor_idx < stride32_end) {
            stride = 32.0f;
            row_i = (anchor_idx - stride16_end) / stride32_grid;
            col_i = (anchor_idx - stride16_end) % stride32_grid;
        }else {  // stride == 64
            stride = 64.0f;
            row_i = (anchor_idx - stride32_end) / stride64_grid;
            col_i = (anchor_idx - stride32_end) % stride64_grid;
        }
        float x_center = 0.5f + (float)col_i;
        float y_center = 0.5f + (float)row_i;
        float x0 = x_center - reg_dist_ptr[0];
        float y0 = y_center - reg_dist_ptr[1];
        float x1 = x_center + reg_dist_ptr[2];
        float y1 = y_center + reg_dist_ptr[3];
        x0 = x0 * stride / scale_x;
        y0 = y0 * stride / scale_y;
        x1 = x1 * stride / scale_x;
        y1 = y1 * stride / scale_y;
        float h = y1 - y0;
        float w = x1 - x0;

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_prob = cls_score_ptr[class_idx];
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        cls_score_ptr += num_class;
        reg_dist_ptr += 4;
    }
}


static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
/*
python build.py --platform LINUX --cxx g++ --backend BACKEND_X86 --exec_file test2_002_ppyoloe_mm_x86

./test2_002_ppyoloe_mm_x86.out test/000000014439.jpg ppyoloe test/save_data/ppyoloe_crn_s_300e_coco 640 0.5 0.6

./test2_002_ppyoloe_mm_x86.out test/000000014439.jpg picodet test/save_data/picodet_s_416_coco_lcnet 416 0.5 0.6


PaddleDetection下的模型预测同一张图片：

python tools/infer.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams --infer_img demo/000000014439.jpg --draw_threshold 0.5


python tools/infer.py -c configs/picodet/picodet_s_416_coco_lcnet.yml -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_416_coco_lcnet.pdparams --infer_img demo/000000014439.jpg --draw_threshold 0.5



*/
#if defined(WINDOWS)
    printf("%s\n", "WINDOWS");
#endif
#if defined(LINUX)
    printf("%s\n", "LINUX");
#endif
    printf("%s\n", miemienet::miemienetVersion());

    if (argc != 7)
    {
        fprintf(stderr, "Usage: %s [imagepath] [archi_name] [model_path] [input_size] [CONF_THRESH] [NMS_THRESH]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* archi_name = argv[2];
    const char* model_path = argv[3];
    int input_size = atoi(argv[4]);
    float CONF_THRESH = atof(argv[5]);
    float NMS_THRESH = atof(argv[6]);

    // 修改Config全局配置时，要在网络创建之前修改。
    Config* cfg = Config::getInstance();
    printf("num_threads=%d\n", cfg->num_threads);
    // 图片张量设置为以NHWC顺序排列。这样对cpu计算更友好。
    // image_data_format会影响卷积层，全连接层的权重的排列方式。
//    cfg->image_data_format = NCHW;
    cfg->image_data_format = NHWC;
    cfg->use_cpp_compute = true;
//    cfg->use_cpp_compute = false;
    cfg->fuse_conv_bn = true;

    const int num_threads_ = cfg->num_threads;

    int batch_size = 1;
    int in_features = 3;
    int num_classes = 80;

    char file_name[256];


    Layer* model = nullptr;
    if (strcmp(archi_name, "ppyoloe") == 0)
    {
        std::vector<int>* layers = MMSHAPE4D(3, 6, 6, 3);
        std::vector<int>* channels = MMSHAPE5D(64, 128, 256, 512, 1024);
        char* act_name = "swish";
        std::vector<int>* return_idx = MMSHAPE3D(1, 2, 3);
        bool depth_wise = false;
        bool use_large_stem = true;
        float width_mult = 0.5f;
        float depth_mult = 0.33f;
        int freeze_at = -1;

        std::vector<int>* in_channels = MMSHAPE3D(int(256 * width_mult), int(512 * width_mult), int(1024 * width_mult));
        std::vector<int>* out_channels = MMSHAPE3D(768, 384, 192);
        char* neck_act_name = "swish";
        int stage_num = 1;
        int block_num = 3;
        bool drop_block = false;
        int block_size=3;
        float keep_prob=0.9f;
        bool spp=true;

        std::vector<int>* head_in_channels = MMSHAPE3D(int(768 * width_mult), int(384 * width_mult), int(192 * width_mult));
        char* head_act_name = "swish";
        std::vector<float>* fpn_strides = new std::vector<float>({32.f, 16.f, 8.f});
        float grid_cell_scale=5.f;
        float grid_cell_offset=0.5f;
        int reg_max = 16;
        int static_assigner_epoch = 100;
        bool use_varifocal_loss = true;

        Layer* backbone = new SNT CSPResNet(layers, channels, act_name, return_idx, depth_wise, use_large_stem, width_mult, depth_mult, freeze_at);
        Layer* neck = new SNT CustomCSPPAN(in_channels, out_channels, neck_act_name, stage_num, block_num, drop_block, block_size, keep_prob, spp, width_mult, depth_mult);
        Layer* yolo_head = new SNT PPYOLOEHead(head_in_channels, num_classes, head_act_name, fpn_strides, grid_cell_scale, grid_cell_offset, reg_max, static_assigner_epoch, use_varifocal_loss);
        model = new SNT PPYOLOE(backbone, neck, yolo_head);

        sprintf(file_name, "%s", model_path);
    }
    else if (strcmp(archi_name, "picodet") == 0)
    {
        float scale = 0.75f;
        std::vector<int>* feature_maps = MMSHAPE3D(3, 4, 5);

        std::vector<int>* in_channels = new std::vector<int>({96, 192, 384});
        int out_channels = 96;
        int kernel_size = 5;
        int num_features = 4;
        bool use_depthwise = true;
        char* neck_act_name = "hardswish";
        std::vector<float>* spatial_scales = new std::vector<float>({0.125f, 0.0625f, 0.03125f});

        PicoFeat* conv_feat = new SNT PicoFeat(96, 96, 4, 2, "bn", true, "hardswish", true);
        std::vector<float>* fpn_stride = new std::vector<float>({8.f, 16.f, 32.f, 64.f});
        bool use_align_head = true;
        int reg_max = 7;
        int feat_in_chan = 96;
        float cell_offset=0.5f;
        char* head_act_name = "hardswish";
        float grid_cell_scale=5.f;

        Layer* backbone = new SNT LCNet(scale, feature_maps);
        Layer* neck = new SNT LCPAN(in_channels, out_channels, kernel_size, num_features, use_depthwise, neck_act_name, spatial_scales);
        Layer* head = new SNT PicoHeadV2(conv_feat, num_classes, fpn_stride, use_align_head, reg_max, feat_in_chan, cell_offset, head_act_name, grid_cell_scale);
        model = new SNT PICODET(backbone, neck, head);

        sprintf(file_name, "%s", model_path);
    }
    else
    {
        printf("archi_name \'%s\' not implemented!\n", archi_name);
        exit(1);
    }


    model->load_state_dict(file_name);

    std::vector<char*>* param_names = new std::vector<char*>;
    std::vector<Tensor*>* params = new std::vector<Tensor*>;
    model->named_parameters(param_names, params);
    for (int i = 0; i < param_names->size(); i++)
    {
        printf("param_names[%d]=%s\n", i, param_names->at(i));
    }


    int target_size = input_size;
    printf("%s\n", imagepath);
    cv::Mat im_bgr = cv::imread(imagepath, 1);   // HWC顺序排列。通道默认是BGR排列。
    if (im_bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    cv::Mat im_rgb;
    cv::cvtColor(im_bgr, im_rgb, cv::COLOR_BGR2RGB);

    cv::Mat im_resize;
    cv::resize(im_rgb, im_resize, cv::Size(target_size, target_size), 0, 0, cv::INTER_CUBIC);

    im_resize.convertTo(im_resize, 5);  // 转float32

    int img_w = im_bgr.cols;
    int img_h = im_bgr.rows;
    float scale_x = (float)target_size / img_w;
    float scale_y = (float)target_size / img_h;

    int HW = im_resize.total();
    int C = im_resize.channels();
    int HWC = HW*C;
    printf("HW=%d\n", HW);
    printf("C=%d\n", C);
    printf("HWC=%d\n", HWC);
    float* im_ptr = (float*) malloc(sizeof(float) * HWC);
    memcpy(im_ptr, im_resize.ptr<float>(0), HWC * sizeof(float));

    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f/58.395f, 1.0f/57.12f, 1.0f/57.375f};
    #pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < HW; i++) {
        for (int j = 0; j < C; j++) {
            im_ptr[i * C + j] = (im_ptr[i * C + j] - mean_vals[j]) * norm_vals[j];
        }
    }
    for (int i = 0; i < 30; i++) {
        printf("%f, ", im_ptr[i]);
    }
    printf("\n");
    std::vector<Object> objects;

    model->eval();

    Tensor* x;
    std::vector<Tensor*>* outs;
    if (Config::getInstance()->image_data_format == NCHW)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, in_features, input_size, input_size), FP32, false, false);
    }
    else if (Config::getInstance()->image_data_format == NHWC)
    {
        x = new SNT Tensor(MMSHAPE4D(batch_size, input_size, input_size, in_features), FP32, false, false);
    }

    // 建立计算图，初始化所有中间张量，给所有中间张量分配内存。
    outs = model->create_tensors(x, 'm');

    printf("======================== eval ========================\n");
    for (int batch_idx = 0; batch_idx < 30; batch_idx++)
    {
        printf("======================== batch_%.3d ========================\n", batch_idx);
        x->set_data_fp32(im_ptr);
//        x->print_msg("x");
//        x->print_data(12);


        auto startTime = std::chrono::system_clock::now();

        outs = model->feed_forward(x, 'm');

        auto endTime = std::chrono::system_clock::now();
        // 1秒=1000毫秒=1000,000微秒
        int cost_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        float cost_ms = (float)cost_microseconds / 1000.f;
        printf("eval forward cost_time = %f ms\n", cost_ms);



        std::vector<Object> proposals;
        if (strcmp(archi_name, "ppyoloe") == 0)
        {
            generate_ppyoloe_proposals(outs->at(0), outs->at(1), scale_x, scale_y, CONF_THRESH, proposals);
        }
        else if (strcmp(archi_name, "picodet") == 0)
        {
            generate_picodet_proposals(outs->at(0), outs->at(1), input_size, scale_x, scale_y, CONF_THRESH, proposals);
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);




        int count = picked.size();
        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            //
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }



        Tensor* y = outs->at(0);
        y->print_msg("y");
        y->print_data(30);
        y = outs->at(1);
        y->print_msg("y");
        y->print_data(30);
    }
    draw_objects(im_bgr, objects);
    delete model;

    // 倒序遍历，可以边遍历边删除
    for (int i = param_names->size() - 1; i >= 0; i--)
    {
        char* param_name = param_names->at(i);
        delete param_name;
        param_names->erase(param_names->begin() + i);
    }
    delete param_names;

    free(im_ptr);
    im_ptr = nullptr;

    Config::getInstance()->destroyInstance();
    MemoryAllocator::getInstance()->destroyInstance();
    TensorIdManager::getInstance()->destroyInstance();

    return 0;
}