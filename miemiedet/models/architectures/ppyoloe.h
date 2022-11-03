#ifndef __MIEMIEDET_PPYOLOE_H__
#define __MIEMIEDET_PPYOLOE_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class PPYOLOE : public Layer
{
public:
    PPYOLOE(Layer* backbone, Layer* neck, Layer* yolo_head=nullptr);
    ~PPYOLOE();

    Layer* backbone;
    Layer* neck;
    Layer* yolo_head;

    virtual std::vector<Tensor*>* create_tensors(Tensor* input, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(Tensor* input, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_PPYOLOE_H__
