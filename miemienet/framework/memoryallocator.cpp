#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "memoryallocator.h"

NS_MM_BEGIN

MemoryAllocator* MemoryAllocator::s_singleInstance = nullptr;

MemoryAllocator* MemoryAllocator::getInstance()
{
    if (s_singleInstance == nullptr)
    {
        s_singleInstance = new (std::nothrow) MemoryAllocator();
    }
    return s_singleInstance;
}

void MemoryAllocator::destroyInstance()
{
    delete s_singleInstance;
    s_singleInstance = nullptr;
}

MemoryAllocator::MemoryAllocator()
{
    offset_fp32 = 0;
    offset_int32 = 0;
//    const int bytes1 = sizeof(float) * Config::getInstance()->mem_fp32_size;
//    mem_fp32 = (float*) malloc(bytes1);
//    const int bytes2 = sizeof(int) * Config::getInstance()->mem_int32_size;
//    mem_int32 = (int*) malloc(bytes2);
}

MemoryAllocator::~MemoryAllocator()
{
//    free(mem_fp32);
//    mem_fp32 = nullptr;
//    free(mem_int32);
//    mem_int32 = nullptr;
}

void MemoryAllocator::reset()
{
    offset_fp32 = 0;
    offset_int32 = 0;
}

float* MemoryAllocator::assign_fp32_memory(const int bytes)
{
//    int start_i = offset_fp32;
//    offset_fp32 += bytes;
//    if (offset_fp32 > Config::getInstance()->mem_fp32_size)
//    {
//        printf("Out Of Memory! Please modify Config::getInstance()->mem_fp32_size!\n");
//        exit(1);
//    }
//    return mem_fp32 + start_i;
    float* aaaaaaaaaaa = (float*) malloc(bytes);
    return aaaaaaaaaaa;
}

int* MemoryAllocator::assign_int32_memory(const int bytes)
{
//    int start_i = offset_int32;
//    offset_int32 += bytes;
//    if (offset_int32 > Config::getInstance()->mem_int32_size)
//    {
//        printf("Out Of Memory! Please modify Config::getInstance()->mem_int32_size!\n");
//        exit(1);
//    }
//    return mem_int32 + start_i;
    int* aaaaaaaaaaa = (int*) malloc(bytes);
    return aaaaaaaaaaa;
}

NS_MM_END
