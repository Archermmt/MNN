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

template<typename T>
static bool read_file_to_buffer(const std::string& file,T* buffer,int size){
  std::ifstream in_file(file,std::ifstream::binary);
  if(!in_file.is_open()){
    return false;
  }
  in_file.read((char*)(&buffer[0]), sizeof(T)*size);
  in_file.close();
  return true;
}

void CPUDLRDevice::device_setup(){
  std::cout<<"setup the device with engine "<<mEngine<<" and target"<<mTarget<<std::endl;
}

bool CPUDLRDevice::device_run(std::vector<void*> input_datas,std::vector<void*> tmp_datas,void* out_data){
  std::cout<<"run the device "<<std::endl;
  //define the running function here
  return false
}

void CPUDLRDevice::device_close(){
  std::cout<<"closing the device "<<std::endl;
}

CPUDLRDevice::CPUDLRDevice(Backend *b,
  const std::string& engine,
  const std::string& target,
  const int max_batch,const int out_stride,
  const flatbuffers::Vector<int>* in_dims,
  const flatbuffers::Vector<int>* in_ndims,
  const flatbuffers::Vector<int>* out_dims,
  const flatbuffers::Vector<int>* out_ndims,
  const std::string& ref_path,
  const flatbuffers::Vector<std::string>* out_names) 
  : Execution(b)
  , mEngine(engine)
  , mTarget(target)
  , mMaxBatch(max_batch)
  , mOutStride(out_stride)
  , mRefPath(ref_path)
{
  get_shapes_from_ndims(in_ndims,in_dims,mInShapes);
  get_shapes_from_ndims(out_ndims,out_dims,mOutShapes);
  for(auto o:mOutShapes){
  	int ele_size=1;
  	for(int d=0;d<o.size();d++){
  	  ele_size*=d;
  	}
  	mOutSizes.emplace_back(ele_size);
  }
  for(size_t i=0;i<(out_names->size());i++){
  	mOutNames.emplace_back(out_names->data()[i]);
  }
  for(size_t i=0;i<mInShapes.size();i++){
    std::cout<<"in_shape["<<i<<"]:";
    for(auto s:mInShapes[i]){
      std::cout<<s<<":";
    }
    std::cout<<std::endl;
  }
  for(size_t i=0;i<mOutShapes.size();i++){
    std::cout<<"out_shape["<<i<<"]:";
    for(auto s:mOutShapes[i]){
      std::cout<<s<<":";
    }
    std::cout<<" has size "<<mOutSizes[i]<<std::endl;
  }
  std::cout<<"has names: ";
  for(auto n:mOutNames){
  	std::cout<<n<<" ,";
  }
  std::cout<<std::endl;
  MNN_ASSERT(mOutNames.size() == mOutShapes.size());
  device_setup();
}

ErrorCode CPUDLRDevice::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
  std::cout<<"calling CPUDLRDevice::onResize "<<std::endl;
  for(size_t i=0;i<mOutShapes.size();i++){
    std::cout<<"out_shape["<<i<<"]:";
    for(auto s:mOutShapes[i]){
      std::cout<<s<<":";
    }
    std::cout<<std::endl;
  }
  int batch_size=inputs[0]->batch();
  std::cout<<"batchsize: "<<batch_size<<std::endl;
  mTempOutputs.resize(mOutShapes.size());
  for(size_t i=0;i<mOutShapes.size();i++){
  	mTempOutputs[i].reset();
  	mOutShapes[i][0]=batch_size;
  	mTempOutputs[i].reset(Tensor::createDevice<float>(mOutShapes[i]));
    bool success = backend()->onAcquireBuffer(mTempOutputs[i].get(),Backend::DYNAMIC);
    if (false == success) {
      return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempOutputs[i].get(),Backend::DYNAMIC);
    std::cout<<i<<" th tensor :";
    mTempOutputs[i]->printShape();
  }
  return NO_ERROR;
}

ErrorCode CPUDLRDevice::onExecute(const vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
  std::cout<<"calling CPUDLRDevice::onExecute "<<std::endl;
  MNN_ASSERT(outputs.size() == 1);
  int batch_size=inputs[0]->batch();
  std::vector<void*> input_datas;
  std::vector<void*> tmp_datas;
  void* out_data=outputs[0]->host<void>();
  for(auto i:inputs){
    input_datas.emplace_back(i->host<void>());
  }
  for(auto t:mTempOutputs){
  	tmp_datas.emplace_back(t->host<void>());
  }
  bool success=device_run(input_datas,tmp_datas,out_data);
  if(!success){
  	std::cout<<"[WARN] Failed to run the device, use recorded data as output"<<std::endl;
    //read datas from file
    #pragma omp parallel for
    for(int i=0;i<mOutNames.size();i++){
      std::string file_path=mRefPath+"/"+mOutNames[i]+".bin";
      std::cout<<"reading data from "<<file_path;
      mTempOutputs[i]->printShape();
      success=read_file_to_buffer(file_path,(float*)tmp_datas[i],batch_size*mOutSizes[i]);
    }
    std::vector<int> offsets;
    offsets.resize(mOutNames.size());
    offsets[0]=0;
    for(int i=1;i<mOutNames.size();i++){
      offsets[i]=offsets[i-1]+mOutSizes[i-1];
    }
    //concat together
    #pragma omp parallel for
    for(int b=0;i<batch_size;i++){
      #pragma omp parallel for
      for(int i=0;i<mOutSizes.size();i++){
        memcpy(out_data+(b*mOutStride+offsets[i])*sizeof(float),tmp_datas[i]+b*mOutSizes[i]*sizeof(float),mOutSizes[i]*sizeof(float));
      }
    }
  }
  return NO_ERROR;
}

class CPUDLRDeviceCreator : public CPUBackend::Creator {
public:
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                              const MNN::Op* op, Backend* backend) const {
    std::cout<<"calling CPUDLRDevice::onCreate "<<std::endl;
    
    auto deviceParam = op->main_as_DLRDeviceParam();
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

    return new CPUDLRDevice(backend);
  }
};

REGISTER_CPU_OP_CREATOR(CPUDLRDeviceCreator, OpType_DLRDevice);
} // namespace MNN
