#ifndef __TENSORIDMANAGER_H__
#define __TENSORIDMANAGER_H__

#include <vector>
#include "../macros.h"

NS_MM_BEGIN

class TensorIdManager
{
public:
    static TensorIdManager* getInstance();
    static void destroyInstance();
    int assign_tensor_id();
    int get_tensor_id();
    int assign_param_id();
    void reset();
private:
    TensorIdManager();
    ~TensorIdManager();
    static TensorIdManager* s_singleInstance;
    int tensor_id;
    int param_id;
};

NS_MM_END

#endif // __TENSORIDMANAGER_H__
