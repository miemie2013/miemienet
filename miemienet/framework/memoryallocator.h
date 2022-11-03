#ifndef __MEMORYALLOCATOR_H__
#define __MEMORYALLOCATOR_H__

#include <vector>
#include "../macros.h"
#include "layer.h"

NS_MM_BEGIN

class MemoryAllocator
{
public:
    static MemoryAllocator* getInstance();
    static void destroyInstance();
    void reset();
//    float* assign_fp32_memory(const int id, const int bytes);
//    int* assign_int32_memory(const int id, const int bytes);
    float* assign_fp32_memory(const int bytes);
    int* assign_int32_memory(const int bytes);
private:
    MemoryAllocator();
    ~MemoryAllocator();
    static MemoryAllocator* s_singleInstance;
    float* mem_fp32;
    int* mem_int32;
    int offset_fp32;
    int offset_int32;
//    std::vector<float*> datas_fp32;
//    std::vector<int*> datas_int32;
//    std::vector<int> datas_id_fp32;
//    std::vector<int> datas_id_int32;
};

NS_MM_END

#endif // __MEMORYALLOCATOR_H__
