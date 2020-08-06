//
//  CPUDLRDevice.hpp
//  Created by chengjin on 2020-08-04.

#ifndef CPUDLRDevice_hpp
#define CPUDLRDevice_hpp

#include "core/Execution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include <iostream>

namespace MNN {
class CPUDLRDevice : public Execution {
public:
  CPUDLRDevice(Backend *b,
    const std::string engine,
    const std::string target,
    const int max_batch,const int out_stride,
    const flatbuffers::Vector<int>* in_dims,
    const flatbuffers::Vector<int>* in_ndims,
    const flatbuffers::Vector<int>* out_dims,
    const flatbuffers::Vector<int>* out_ndims,
    const std::string ref_path,
    const std::string& out_names);

  virtual ~CPUDLRDevice(){
    device_close();
  }
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
  void device_setup();
  bool device_run(std::vector<void*> input_datas,std::vector<void*> tmp_datas,void* out_data);
  void device_close();

  void* mDevicePtr{nullptr};
  std::vector<std::shared_ptr<Tensor>> mTempOutputs;

  const std::string mEngine;
  const std::string mTarget;
  int mMaxBatch;
  int mOutStride;
  const std::string mRefPath;
  std::vector<std::vector<int>> mInShapes;
  std::vector<std::vector<int>> mOutShapes;
  std::vector<int> mOutSizes;
  std::vector<std::string> mOutNames;
};

} // namespace MNN

#endif /* CPUDLRDevice_hpp */
