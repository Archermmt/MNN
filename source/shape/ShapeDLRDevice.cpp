//
//  ShapeDLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include <iostream>

namespace MNN {
class DLRDeviceSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        std::cout<<"calling DLRDeviceSizeComputer "<<std::endl;
        return true;
    }
};

REGISTER_SHAPE(DLRDeviceSizeComputer, OpType_DLRDevice);
} // namespace MNN
