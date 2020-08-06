//
//  ShapeDLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include <iostream>

namespace MNN {
/*
void get_shapes_from_ndims(const flatbuffers::Vector<int>* ndims,const flatbuffers::Vector<int>* dims,
  std::vector<std::vector<int>>& shapes){
  int base=0;
  shapes.resize(ndims->size());
  for(size_t i=0;i<(ndims->size());i++){
    shapes[i].resize(ndims->data()[i]);
    for(int j=0;j<ndims->data()[i];j++){
      shapes[i][j]=dims->data()[base+j];
    }
    base+=ndims->data()[i];
  }
}
*/
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

    /*
    //get the shapes
    std::vector<std::vector<int>> in_shapes;
    std::vector<std::vector<int>> out_shapes;
    get_shapes_from_ndims(deviceParam->in_ndims(),deviceParam->in_dims(),in_shapes);
    get_shapes_from_ndims(deviceParam->out_ndims(),deviceParam->out_dims(),out_shapes);

    std::cout<<"get engine with "<<(deviceParam->engine()->c_str())<<std::endl;
    std::cout<<"get target with "<<(deviceParam->target()->c_str())<<std::endl;
    std::cout<<"get ref_path with "<<(deviceParam->ref_path()->c_str())<<std::endl;
    std::cout<<"get max_batch with "<<(deviceParam->max_batch())<<std::endl;
    MNN_ASSERT(inputs.size() == in_shapes.size());
    //use batch size for size compute
    int batch_size=inputs[0]->buffer().dim[0].extent;
    int whole_stride=0;
    for(auto o:out_shapes){
      int ele_size=1;
      for(int d=1;d<o.size();d++){
        ele_size*=o[d];
      }
      whole_stride+=ele_size;
    }
    ob.dim[0].extent=batch_size;

    for(size_t i=0;i<in_shapes.size();i++){
      std::cout<<"in_shape["<<i<<"]:";
      for(auto s:in_shapes[i]){
        std::cout<<s<<":";
      }
      std::cout<<std::endl;
    }
    for(size_t i=0;i<out_shapes.size();i++){
      std::cout<<"out_shape["<<i<<"]:";
      for(auto s:out_shapes[i]){
        std::cout<<s<<":";
      }
      std::cout<<std::endl;
    }
    */
    inputs[0]->printShape();
    outputs[0]->printShape();
    return true;
  }
};

REGISTER_SHAPE(DLRDeviceSizeComputer, OpType_DLRDevice);
} // namespace MNN
