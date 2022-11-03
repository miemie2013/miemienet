#include "tensoridmanager.h"

NS_MM_BEGIN

TensorIdManager* TensorIdManager::s_singleInstance = nullptr;

TensorIdManager* TensorIdManager::getInstance()
{
    if (s_singleInstance == nullptr)
    {
        s_singleInstance = new (std::nothrow) TensorIdManager();
    }
    return s_singleInstance;
}

void TensorIdManager::destroyInstance()
{
    delete s_singleInstance;
    s_singleInstance = nullptr;
}

TensorIdManager::TensorIdManager()
{
    this->tensor_id = 0;
    this->param_id = 19950817;
}

TensorIdManager::~TensorIdManager()
{
}

int TensorIdManager::assign_tensor_id()
{
    return this->tensor_id++;
}

int TensorIdManager::get_tensor_id()
{
    return this->tensor_id;
}

int TensorIdManager::assign_param_id()
{
    return this->param_id++;
}

void TensorIdManager::reset()
{
    this->tensor_id = 0;
}

NS_MM_END
