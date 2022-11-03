#ifndef __MIEMIEDET_PICODET_H__
#define __MIEMIEDET_PICODET_H__

#include "../../miemiedet.h"
#include "../../../miemienet/macros.h"
#include "../../../miemienet/miemienet.h"

using namespace miemienet;

namespace miemiedet {

class PICODET : public Layer
{
public:
    PICODET(Layer* backbone, Layer* neck, Layer* head=nullptr);
    ~PICODET();

    Layer* backbone;
    Layer* neck;
    Layer* head;

    virtual std::vector<Tensor*>* create_tensors(Tensor* input, char miemie2013);
    virtual std::vector<Tensor*>* feed_forward(Tensor* input, char miemie2013);
private:
};

}  // namespace miemiedet

#endif // __MIEMIEDET_PICODET_H__
