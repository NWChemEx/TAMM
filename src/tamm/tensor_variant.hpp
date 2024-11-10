#pragma once

#include "tamm/scalar.hpp"
#include "tamm/scheduler.hpp"
#include "tamm/tensor.hpp"

namespace tamm {
class TensorVariant {
public:
  using TensorType =
      std::variant</* Tensor<int>, Tensor<int64_t>,  Tensor<float>, */Tensor<double>/* ,
                   Tensor<std::complex<float>>, Tensor<std::complex<double>> */>;

  TensorVariant() = default;
  template<typename T>
  TensorVariant(Tensor<T> value): value_{value} {}

  TensorVariant(TensorType value): value_{value} {}

  TensorVariant(ElType eltype, const IndexLabelVec& ilv) { init_tensor(eltype, ilv); }

  void init_tensor(ElType eltype, const IndexLabelVec& ilv) {
    switch(eltype) {
      case ElType::inv: value_ = Tensor<double>{ilv}; break;
      // case ElType::i32: value_ = Tensor<int>{ilv}; break;
      // case ElType::i64: value_ = Tensor<int64_t>{ilv}; break;
      // case ElType::fp32: value_ = Tensor<float>{ilv}; break;
      case ElType::fp64:
        value_ = Tensor<double>{ilv};
        break;
        // case ElType::cfp32:
        //     value_ = Tensor<std::complex<float>>{ilv};
        //     break;
        // case ElType::cfp64:
        //     value_ = Tensor<std::complex<double>>{ilv};
        //     break;
      default: UNREACHABLE();
    }
  }

  TensorVariant(const TensorVariant&)  = default;
  TensorVariant(TensorVariant&& other) = default;

  TensorVariant& operator=(TensorVariant other) noexcept {
    using std::swap;
    swap(*this, other);
    return *this;
  }

  friend void swap(TensorVariant& first, TensorVariant& second) noexcept {
    using std::swap;
    swap(first.value_, second.value_);
  }

  TensorType value() const { return value_; }

  void allocate(Scheduler& sch) {
    std::visit(overloaded{[&](auto& e1) { sch.allocate(e1); }}, value_);
  }

  void deallocate(Scheduler& sch) {
    std::visit(overloaded{[&](auto& e1) { sch.deallocate(e1); }}, value_);
  }

  TensorBase* base_ptr() const {
    return std::visit(overloaded{[&](const auto& tensor) { return tensor.base_ptr(); }}, value_);
  }

  std::string to_string() const {
    return std::visit(
      overloaded{
        //   [&](const Tensor<int>& tensor) { return "int32-tensor"; },
        //   [&](const Tensor<int64_t>& tensor) { return "int64-tensor"; },
        //   [&](const Tensor<float>& tensor) { return "float-tensor"; },
        [&](const Tensor<double>& tensor) { return "double-tensor"; },
        //   [&](const Tensor<std::complex<float>>& tensor) { return "complex-float-tensor"; },
        //   [&](const Tensor<std::complex<double>>& tensor) { return "complex-double-tensor"; }
      },
      value_);
  }

  ElType to_eltype() const {
    return std::visit(
      overloaded{
        //   [&](const Tensor<int>& tensor) { return ElType::i32; },
        //   [&](const Tensor<int64_t>& tensor) { return ElType::i64; },
        //   [&](const Tensor<float>& tensor) { return ElType::fp32; },
        [&](const Tensor<double>& tensor) { return ElType::fp64; },
        //   [&](const Tensor<std::complex<float>>& tensor) { return
        //   ElType::cf32; },
        //   [&](const Tensor<std::complex<double>>& tensor) { return
        //   ElType::cf64; }
      },
      value_);
  }

  size_t el_size() const {
    return std::visit(
      overloaded{
        //   [&](const Tensor<int>& tensor) { return sizeof(int); },
        //   [&](const Tensor<int64_t>& tensor) { return sizeof(int64_t); },
        //   [&](const Tensor<float>& tensor) { return sizeof(float); },
        [&](const Tensor<double>& tensor) { return sizeof(double); },
        //   [&](const Tensor<std::complex<float>>& tensor) { return
        //   sizeof(std::complex<float>); },
        //   [&](const Tensor<std::complex<double>>& tensor) { return
        //   sizeofstd::complex<double>(); }
      },
      value_);
  }

  size_t mem_size() const {
    size_t result   = el_size();
    auto   tis_list = std::visit(
        overloaded{[&](const auto& tensor) { return tensor.tiled_index_spaces(); }}, value_);
    for(const auto& tis: tis_list) { result *= tis.max_num_indices(); }
    return result;
  }

  bool is_allocated() const {
    return std::visit(overloaded{[&](const auto& tensor) { return tensor.is_allocated(); }},
                      value_);
  }

  bool has_ops() const {
    return std::visit(overloaded{[&](const auto& tensor) {
                        return (tensor.get_updates().size() - tensor.version() != 0);
                      }},
                      value_);
  }

  bool has_spin(const IndexLabelVec& ilv) const {
    for(const auto& label: ilv) {
      if(label.tiled_index_space().has_spin()) { return true; }
    }
    return false;
  }

  void* get_symbol_ptr() const {
    return std::visit(overloaded{[&](const auto& tensor) { return tensor.get_symbol_ptr(); }},
                      value_);
  }

private:
  TensorType value_;
};

} // namespace tamm
