//
//  CPUDLRDevice.hpp
//  Created by chengjin on 2020-08-04.

#ifndef CPUDLRDevice_hpp
#define CPUDLRDevice_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUDLRDevice : public Execution {
public:
    CPUDLRDevice(Backend *b) : Execution(b) {
        // Do nothing
    }
    virtual ~CPUDLRDevice() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempOutput;
};

} // namespace MNN

#endif /* CPUDLRDevice_hpp */
