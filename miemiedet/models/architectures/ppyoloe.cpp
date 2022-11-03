#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "ppyoloe.h"

namespace miemiedet {

PPYOLOE::PPYOLOE(Layer* backbone, Layer* neck, Layer* yolo_head)
{
    this->backbone = backbone;
    register_sublayer("backbone", backbone);
    this->neck = neck;
    register_sublayer("neck", neck);
    this->yolo_head = yolo_head;
    register_sublayer("yolo_head", yolo_head);
}

PPYOLOE::~PPYOLOE()
{
    delete backbone;
    delete neck;
    delete yolo_head;
}

std::vector<Tensor*>* PPYOLOE::create_tensors(Tensor* input, char miemie2013)
{
    std::vector<Tensor*>* body_feats = backbone->create_tensors(input, miemie2013);
    std::vector<Tensor*>* fpn_feats = neck->create_tensors(body_feats, miemie2013);
    std::vector<Tensor*>* outs = yolo_head->create_tensors(fpn_feats, miemie2013);
    return outs;
}

std::vector<Tensor*>* PPYOLOE::feed_forward(Tensor* input, char miemie2013)
{
    std::vector<Tensor*>* body_feats = backbone->feed_forward(input, miemie2013);
    std::vector<Tensor*>* fpn_feats = neck->feed_forward(body_feats, miemie2013);
    std::vector<Tensor*>* outs = yolo_head->feed_forward(fpn_feats, miemie2013);
    return outs;
}


}  // namespace miemiedet
