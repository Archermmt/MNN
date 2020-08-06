//
//  ShapeDLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include "core/Macro.h"
#include "core/SizeComputer.hpp"
#include <iostream>

namespace MNN {

template<typename T>
void get_shapes_from_ndims(const std::vector<T>& ndims,const std::vector<T>& dims,
  std::vector<std::vector<T>>& shapes){
  int base=0;
  shapes.resize(ndims.size());
  for(size_t i=0;i<ndims.size();i++){
    shapes[i].resize(ndims[i]);
    for(int j=0;j<ndims[i];j++){
      shapes[i][j]=dims[base+j];
    }
    base+=ndims[i];
  }
}

template
void get_shapes_from_ndims<int>(const std::vector<int>& ndims,const std::vector<int>& dims,
  std::vector<std::vector<int>>& shapes);

class DLRDeviceSizeComputer : public SizeComputer {
  virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
    std::cout<<"calling DLRDeviceSizeComputer "<<std::endl;
    std::cout<<"input size "<<inputs.size()<<std::endl;
    std::cout<<"output size "<<outputs.size()<<std::endl;
    auto deviceParam = op->main_as_DLRDeviceParam();
    
    std::cout<<"get engine with "<<(deviceParam->engine())<<std::endl;
    std::cout<<"get target with "<<(deviceParam->target())<<std::endl;
    std::cout<<"get ref_path with "<<(deviceParam->ref_path())<<std::endl;
    std::cout<<"get max_batch with "<<(deviceParam->max_batch())<<std::endl;
    /*
    std::vector<std::vector<int>> in_shapes;
    get_shapes_from_ndims(deviceParam->in_ndims(),deviceParam->in_dims(),in_shapes);
    std::cout<<"get in_dims with ";
    for(auto s:deviceParam->in_dims){
        std::cout<<s<<":";
    }
    std::cout<<std::endl;
    std::cout<<"get in_ndims with ";
    for(auto s:deviceParam->in_ndims){
        std::cout<<s<<":";
    }
    std::cout<<std::endl;
    std::cout<<"get out_dims with ";
    for(auto s:deviceParam->out_dims){
        std::cout<<s<<":";
    }
    std::cout<<std::endl;
    std::cout<<"get out_ndims with ";
    for(auto s:deviceParam->out_ndims){
        std::cout<<s<<":";
    }
    std::cout<<std::endl;
    std::cout<<"get out_names with ";
    for(auto s:deviceParam->out_names){
        std::cout<<s<<":";
    }
    std::cout<<std::endl;
    */
    return true;
  }
};

REGISTER_SHAPE(DLRDeviceSizeComputer, OpType_DLRDevice);
} // namespace MNN
