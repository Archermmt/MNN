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
    MNN_ASSERT(1 == outputs.size());
    auto& ob = outputs[0]->buffer();

    auto deviceParam = op->main_as_DLRDeviceParam();
    //get batchsize and check if all the inputs has same batchsize
    int batch_size=inputs[0]->buffer().dim[0].extent;
    for(auto i:inputs){
      MNN_ASSERT(i->buffer().dim[0].extent == batch_size);
    }
    ob.dimensions = 2;
    ob.dim[0].extent=batch_size;
    ob.dim[1].extent=deviceParam->out_stride();
    ob.type = inputs[0]->buffer().type;
    return true;
  }
};

REGISTER_SHAPE(DLRDeviceSizeComputer, OpType_DLRDevice);
} // namespace MNN
