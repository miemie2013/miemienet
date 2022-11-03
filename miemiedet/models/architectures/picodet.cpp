#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include "picodet.h"

namespace miemiedet {

PICODET::PICODET(Layer* backbone, Layer* neck, Layer* head)
{
    this->backbone = backbone;
    register_sublayer("backbone", backbone);
    this->neck = neck;
    register_sublayer("neck", neck);
    this->head = head;
    register_sublayer("head", head);
}

PICODET::~PICODET()
{
    delete backbone;
    delete neck;
    delete head;
}

std::vector<Tensor*>* PICODET::create_tensors(Tensor* input, char miemie2013)
{
    std::vector<Tensor*>* body_feats = backbone->create_tensors(input, miemie2013);
    std::vector<Tensor*>* fpn_feats = neck->create_tensors(body_feats, miemie2013);
    std::vector<Tensor*>* outs = head->create_tensors(fpn_feats, miemie2013);
    return outs;
}

std::vector<Tensor*>* PICODET::feed_forward(Tensor* input, char miemie2013)
{
    std::vector<Tensor*>* body_feats = backbone->feed_forward(input, miemie2013);
    std::vector<Tensor*>* fpn_feats = neck->feed_forward(body_feats, miemie2013);
    std::vector<Tensor*>* outs = head->feed_forward(fpn_feats, miemie2013);
    return outs;
}


}  // namespace miemiedet
