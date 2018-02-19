#pragma once
#include "ATen/ATen.h"

template<typename T>
inline T tensor_as(at::Tensor&& t) = delete;

template<>
inline int64_t tensor_as(at::Tensor&& t) {
  // workaround for 1-dim 1-element pytorch tensors until zero-dim
  // tensors are fully supported
  if(t.ndimension() == 1 && t.size(0) == 1) {
    t = t[0];
  }
  return at::Scalar(t).to<int64_t>();
}

template<>
inline bool tensor_as(at::Tensor&& t) {
  // workaround for 1-dim 1-element pytorch tensors until zero-dim
  // tensors are fully supported
  if(t.ndimension() == 1 && t.size(0) == 1) {
    t = t[0];
  }
  return at::Scalar(t).to<bool>();
}

template<>
inline double tensor_as(at::Tensor&& t) {
  // workaround for 1-dim 1-element pytorch tensors until zero-dim
  // tensors are fully supported
  if(t.ndimension() == 1 && t.size(0) == 1) {
    t = t[0];
  }
  return at::Scalar(t).to<double>();
}

template<>
inline at::IntList tensor_as(at::Tensor&& t) {
  if (t.type().scalarType() != at::ScalarType::Long)
    throw std::runtime_error("Expected a LongTensor");
  if (t.dim() != 1)
    throw std::runtime_error("Expected a 1D LongTensor");
  if (!t.is_contiguous())
    throw std::runtime_error("Expected a contiguous LongTensor");
  return at::IntList{t.data<int64_t>(), static_cast<size_t>(t.numel())};
}

template<>
inline at::Scalar tensor_as(at::Tensor&& t) {
  throw at::Scalar(t.view({}));
}

template<size_t N>
inline std::array<bool, N> tensor_as(at::Tensor&& t) {
  throw std::runtime_error("tensor_as<std::array<bool, N>>: NYI");
}
