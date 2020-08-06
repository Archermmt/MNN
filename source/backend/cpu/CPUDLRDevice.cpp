//
//  CPUDLRDevice.cpp
//  Created by chengjin on 2020-08-04.

#include <fstream>
#include "backend/cpu/CPUDLRDevice.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

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

void split_str(const std::string& s,const std::string& c,std::vector<std::string>& v){
  std::string::size_type pos1, pos2;
  size_t len = s.length();
  pos2 = s.find(c);
  pos1 = 0;
  while(std::string::npos != pos2){
    v.emplace_back(s.substr(pos1, pos2-pos1));
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != len)
    v.emplace_back(s.substr(pos1));
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
  std::cout<<"[DLRDevice] Setup the device with engine : "<<mEngine<<", target : "<<mTarget<<", max_batch : "<<mMaxBatch<<std::endl;
}

bool CPUDLRDevice::device_run(std::vector<void*> input_datas,std::vector<void*> tmp_datas,void* out_data){
  std::cout<<"[DLRDevice] Run the device with "<<input_datas.size()<<" inputs and "<<tmp_datas.size()<<" outputs"<<std::endl;
  //define the running function here
  return false;
}

void CPUDLRDevice::device_close(){
  std::cout<<"[DLRDevice] Closing the device "<<std::endl;
}

CPUDLRDevice::CPUDLRDevice(Backend *b,
  const std::string engine,
  const std::string target,
  const int max_batch,const int out_stride,
  const flatbuffers::Vector<int>* in_dims,
  const flatbuffers::Vector<int>* in_ndims,
  const flatbuffers::Vector<int>* out_dims,
  const flatbuffers::Vector<int>* out_ndims,
  const std::string ref_path,
  const std::string& out_names) 
  : Execution(b)
  , mEngine(engine)
  , mTarget(target)
  , mMaxBatch(max_batch)
  , mOutStride(out_stride)
  , mRefPath(ref_path)
{
  split_str(out_names,",",mOutNames);
  get_shapes_from_ndims(in_ndims,in_dims,mInShapes);
  get_shapes_from_ndims(out_ndims,out_dims,mOutShapes);
  for(auto o:mOutShapes){
    int ele_size=1;
    for(int d=1;d<o.size();d++)
      ele_size*=o[d];
    mOutSizes.emplace_back(ele_size);
  }
  MNN_ASSERT(mOutNames.size() == mOutShapes.size());
  device_setup();
}

ErrorCode CPUDLRDevice::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
  int batch_size=inputs[0]->batch();
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
      success=read_file_to_buffer(file_path,(float*)tmp_datas[i],batch_size*mOutSizes[i]);
      std::cout<<(success?"success":"fail")<<", reading data from "<<file_path;
      mTempOutputs[i]->printShape();
    }
    std::vector<int> offsets;
    offsets.resize(mOutNames.size());
    offsets[0]=0;
    for(int i=1;i<mOutNames.size();i++){
      offsets[i]=offsets[i-1]+mOutSizes[i-1];
    }
    //concat together
    #pragma omp parallel for
    for(int b=0;b<batch_size;b++){
      #pragma omp parallel for
      for(int i=0;i<mOutSizes.size();i++){
        memcpy((float*)out_data+(b*mOutStride+offsets[i]),(float*)tmp_datas[i]+b*mOutSizes[i],mOutSizes[i]*sizeof(float));
      }
    }
  }
  return NO_ERROR;
}

class CPUDLRDeviceCreator : public CPUBackend::Creator {
public:
  virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                              const MNN::Op* op, Backend* backend) const {
    auto deviceParam = op->main_as_DLRDeviceParam();
    return new CPUDLRDevice(backend,
      deviceParam->engine()->c_str(),
      deviceParam->target()->c_str(),
      deviceParam->max_batch(),
      deviceParam->out_stride(),
      deviceParam->in_dims(),
      deviceParam->in_ndims(),
      deviceParam->out_dims(),
      deviceParam->out_ndims(),
      deviceParam->ref_path()->c_str(),
      deviceParam->out_names()->c_str());
  }
};

REGISTER_CPU_OP_CREATOR(CPUDLRDeviceCreator, OpType_DLRDevice);
} // namespace MNN
