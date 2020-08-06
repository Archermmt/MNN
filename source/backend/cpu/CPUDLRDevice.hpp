//
//  CPUDLRDevice.hpp
//  Created by chengjin on 2020-08-04.

#ifndef CPUDLRDevice_hpp
#define CPUDLRDevice_hpp

#include "core/Execution.hpp"
#include <iostream>

namespace MNN {
class CPUDLRDevice : public Execution {
public:
  CPUDLRDevice(Backend *b);
  virtual ~CPUDLRDevice(){
    close_device();
  }
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
  void device_setup();
  bool device_run();
  void device_close();

  void* mDevicePtr{nullptr};
  std::vector<std::shared_ptr<Tensor>> mTempOutputs;

  std::string mEngine;
  std::string mTarget;
  std::string mMaxBatch;
  std::string mOutStride;
  std::string mRefPath;
  std::vector<std::vector<int>> mInShapes;
  std::vector<std::vector<int>> mOutShapes;
  std::vector<int> mOutSizes;
  std::vector<std::string> mOutNames;
};

} // namespace MNN

#endif /* CPUDLRDevice_hpp */
