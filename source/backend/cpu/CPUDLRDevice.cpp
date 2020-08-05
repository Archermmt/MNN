//
//  CPUDLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include "backend/cpu/CPUDLRDevice.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
using namespace std;

namespace MNN {
ErrorCode CPUDLRDevice::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::cout<<"calling CPUDLRDevice::onResize "<<std::endl;
    return NO_ERROR;
}

ErrorCode CPUDLRDevice::onExecute(const vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    std::cout<<"calling CPUDLRDevice::onExecute "<<std::endl;
    return NO_ERROR;
}

class CPUDLRDeviceCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        std::cout<<"calling CPUDLRDevice::onCreate "<<std::endl;
        return new CPUDLRDevice(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDLRDeviceCreator, OpType_DLRDevice);
} // namespace MNN
