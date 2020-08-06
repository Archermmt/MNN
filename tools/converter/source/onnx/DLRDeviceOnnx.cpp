//
//  DLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(DLRDeviceOnnx);

void parse_ints(const onnx::AttributeProto& proto,std::vector<int>& dims){
  DCHECK(proto.type() == ::onnx::AttributeProto_AttributeType_INTS) << "Node Attribute ERROR";
  const int size = proto.ints_size();
  for (int k = 0; k < size; ++k) {
    dims.push_back(proto.ints(k));
  }
}

MNN::OpType DLRDeviceOnnx::opType() {
  return MNN::OpType_DLRDevice;
}

MNN::OpParameter DLRDeviceOnnx::type() {
  return MNN::OpParameter_DLRDeviceParam;
}

void DLRDeviceOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
  auto para  = new MNN::DLRDeviceParamT;
  for (int i = 0; i < onnxNode->attribute_size(); ++i) {
    const auto& attributeProto = onnxNode->attribute(i);
    const auto& attributeName  = attributeProto.name();
    if (attributeName == "engine") {
      para->engine = attributeProto.s();
    }else if (attributeName == "target") {
      para->target = attributeProto.s();
    }else if (attributeName == "ref_path") {
      para->ref_path = attributeProto.s();
    }else if (attributeName == "max_batch") {
      para->max_batch = attributeProto.i();
    }else if (attributeName == "out_stride") {
      para->out_stride = attributeProto.i();
    }else if (attributeName == "in_dims") {
      parse_ints(attributeProto,para->in_dims);
    }else if (attributeName == "in_ndims") {
      parse_ints(attributeProto,para->in_ndims);
    }else if (attributeName == "out_dims") {
      parse_ints(attributeProto,para->out_dims);
    }else if (attributeName == "out_ndims") {
      parse_ints(attributeProto,para->out_ndims);
    }else if (attributeName == "out_names") {
      para->out_names = attributeProto.s();
    }
  }
  dstOp->main.value = para;
}

REGISTER_CONVERTER(DLRDeviceOnnx, DLRDevice);
