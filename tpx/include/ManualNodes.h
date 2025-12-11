#pragma once
#include "Node.h"
#include "TPXTensor.h"

namespace tensorplay {
namespace tpx {

struct SelectBackward : public Node {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t index_;
    DType dtype_;
    Device device_;

    SelectBackward(Size shape, int64_t dim, int64_t index, DType dtype, Device device)
        : input_shape_(static_cast<std::vector<int64_t>>(shape)), dim_(dim), index_(index), dtype_(dtype), device_(device) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        
        Tensor grad_input = Tensor::zeros(input_shape_, dtype_, device_);
        grad_input.core().select(dim_, index_).copy_(grad.core());
        
        return {grad_input};
    }
};

struct SliceBackward : public Node {
    std::vector<int64_t> input_shape_;
    int64_t dim_;
    int64_t start_;
    int64_t end_;
    int64_t step_;
    DType dtype_;
    Device device_;

    SliceBackward(Size shape, int64_t dim, int64_t start, int64_t end, int64_t step, DType dtype, Device device)
        : input_shape_(static_cast<std::vector<int64_t>>(shape)), dim_(dim), start_(start), end_(end), step_(step), dtype_(dtype), device_(device) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].defined()) return {Tensor()};
        Tensor grad = inputs[0];
        
        Tensor grad_input = Tensor::zeros(input_shape_, dtype_, device_);
        grad_input.core().slice(dim_, start_, end_, step_).copy_(grad.core());
        
        return {grad_input};
    }
};

}
}
