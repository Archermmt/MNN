//
//  DLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DLRDeviceOnnx);

MNN::OpType DLRDeviceOnnx::opType() {
    std::cout<<"calliing DLRDeviceOnnx opType"<<std::endl;
    return MNN::OpType_DLRDevice;
}
MNN::OpParameter DLRDeviceOnnx::type() {
    std::cout<<"calliing DLRDeviceOnnx type"<<std::endl;
    return MNN::OpParameter_DLRDeviceParam;
}

void DLRDeviceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
    std::cout<<"calling DLRDeviceOnnx run !!"<<std::endl;
    auto para  = new MNN::DLRDeviceParamT;
    /*
    auto para  = new MNN::DLRDeviceParam;
    para->axis = 0;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attributeProto = onnxNode->attribute(i);
        const auto& attributeName  = attributeProto.name();
        if (attributeName == "axis") {
            para->axis = attributeProto.i();
        }
    }

    dstOp->main.value = para;
    */
}

REGISTER_CONVERTER(DLRDeviceOnnx, DLRDevice);
