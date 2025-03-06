#ifndef CONTRACT_HPP
#define CONTRACT_HPP
#include "coordinate.hpp"
#include "timer.hpp"
#include "index_tensors.hpp"
#include <algorithm>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <forward_list>
#include <iostream>
#include <ranges>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

template <class DT> class FastccTensor {
private:
  std::vector<NNZ<DT>> nonzeros;
  int *shape;
  int dimensionality = 42;

public:
  using iterator = typename std::vector<NNZ<DT>>::iterator;
  using value_type = typename std::vector<NNZ<DT>>::value_type;
  iterator begin() { return nonzeros.begin(); }
  iterator end() { return nonzeros.end(); }
  FastccTensor(std::string fname, bool);
  double reduce() {
    double sum = 0;
    for (auto &nnz : nonzeros) {
      sum += nnz.get_data();
    }
    return sum;
  }
  void write(std::string fname);
  int get_dimensionality() const {
    if (dimensionality == 42) {
        std::cerr << "Error, trying to access dimensionality of an uninitialized tensor" << std::endl;
        exit(1);
    }
    return dimensionality;
  }
  // Constructor for a tensor of given shape and number of non-zeros, fills
  // with random values and indices
  FastccTensor(int size, int dimensionality, int *shape) {
    this->shape = shape;
    this->dimensionality = dimensionality;
    nonzeros.reserve(size);
    for (int i = 0; i < size; i++) {
      nonzeros.emplace_back(dimensionality, shape);
    }
  }
  std::string to_string() {
    std::string str = "";
    for (auto &nnz : nonzeros) {
      str += nnz.to_string() + "\n";
    }
    return str;
  }
  // Make a tensor with just ones at given positions
  template <class It> FastccTensor(It begin, It end) {
    for (auto it = begin; it != end; it++) {
      if constexpr (std::is_class<DT>::value) {
        nonzeros.emplace_back(DT(), *it);
      } else {
        nonzeros.emplace_back(1.0, *it);
      }
    }
    this->_infer_dimensionality();
    this->_infer_shape();
  }
  FastccTensor(int size = 0) { nonzeros.reserve(size); }
  std::vector<NNZ<DT>> &get_nonzeros() { return nonzeros; }
  const NNZ<DT> &nnz_at(int index) const { return nonzeros[index]; }
  int get_size() const { return nonzeros.size(); }
  void _infer_dimensionality() {
    if (nonzeros.size() > 0) {
      dimensionality = nonzeros[0].get_coords().get_dimensionality();
    }
  }
  std::string get_shape_string() {
    if (dimensionality == 42) {
      this->_infer_dimensionality();
      this->_infer_shape();
    }
    std::string str = "";
    for (int i = 0; i < dimensionality; i++) {
      str += std::to_string(shape[i]) + " ";
    }
    return str;
  }
  int *get_shape_ref() {
    if (dimensionality == 42) {
      this->_infer_dimensionality();
      this->_infer_shape();
    }
    return shape;
  }
  void _infer_shape() {
    // TODO this is a mem-leak. Add a guard before allocation
    if (nonzeros.size() > 0) {
      shape = new int[dimensionality];
      for (int i = 0; i < dimensionality; i++) {
        shape[i] = 0;
      }
      for (auto &nnz : nonzeros) {
        auto coords = nnz.get_coords();
        for (int i = 0; i < dimensionality; i++) {
          if (coords.get_index(i) > shape[i]) {
            shape[i] = coords.get_index(i);
          }
        }
      }
      std::vector<int> shape_vec(shape, shape + dimensionality);
      for (auto &nnz : nonzeros) {
        nnz.get_coords().set_shape(shape_vec);
      }
    }
  }

  // ASSUMEs batch indices have same shape in both tensors.
  template <class RES, class RIGHT>
  ListTensor<RES> multiply_3d(FastccTensor<RIGHT> &other, CoOrdinate left_batch,
                              CoOrdinate left_contr, CoOrdinate left_ex,
                              CoOrdinate right_batch, CoOrdinate right_contr,
                              CoOrdinate right_ex) {
      std::cout<<"Kernle called"<<std::endl;
    // always run 4D loop structure, if the coordinates are empty then the
    // iteration is a singleton.
    // for b
    //    for l
    //       for c
    //           for r
    BoundedCoordinate sample_batch =
        this->nonzeros[0].get_coords().gather(left_batch).get_bounded();
    BoundedCoordinate sample_rightex =
        other.nonzeros[0].get_coords().gather(right_ex).get_bounded();
    BoundedCoordinate sample_leftex =
        this->nonzeros[0].get_coords().gather(left_ex).get_bounded();
    uint64_t batch_max = sample_batch.get_linear_bound();
    if (batch_max == 1) {
      assert(left_batch.get_dimensionality() == 0);
      assert(right_batch.get_dimensionality() == 0);
    } else {
      assert(left_batch.get_dimensionality() > 0);
      assert(right_batch.get_dimensionality() ==
             left_batch.get_dimensionality());
    }
    std::cout<<"Batch assertions done"<<std::endl;
    if (left_contr.get_dimensionality() != 0) {
      assert(this->get_nonzeros()[0]
                 .get_coords()
                 .gather(left_contr)
                 .get_linearized_max() == other.get_nonzeros()[0]
                                              .get_coords()
                                              .gather(right_contr)
                                              .get_linearized_max());
    }
    std::cout<<"Assertions done"<<std::endl;
    InputTensorMap3D<DT> left_indexed =
        InputTensorMap3D<DT>(*this, left_batch, left_ex, left_contr, batch_max);
    InputTensorMap3D<RIGHT> right_indexed = InputTensorMap3D<RIGHT>(
        other, right_batch, right_contr, right_ex, batch_max);
    std::cout<<"indexed both tensors"<<std::endl;
    init_heaps(1);

    RES *workspace =
        (RES *)calloc(sample_rightex.get_linear_bound(), sizeof(RES));
    ListTensor<RES> result_tensor(sample_batch.get_dimensionality() +
                                      sample_leftex.get_dimensionality() +
                                      sample_rightex.get_dimensionality(),
                                  0);

    for (uint64_t batch_iter = 0; batch_iter < batch_max; batch_iter++) {
      for (auto &left_slice_l : left_indexed.indexed_tensor[batch_iter]) { //left_slice_l is type of key value pair of hashmap
        for (auto &left_slice_c : left_slice_l.second) { //left_slice_c is type of pair of c and data nnz (iterator of vector)
          auto right_iter_r =
              right_indexed.indexed_tensor[batch_iter].find(left_slice_c.first);
          if (right_iter_r == right_indexed.indexed_tensor[batch_iter].end()) {
            continue;
          }
          for (auto &right_nnz : right_iter_r->second) {
            auto ws_index =
                right_nnz.first == uint64_t(-1) ? 0 : right_nnz.first;
            workspace[ws_index] += left_slice_c.second * right_nnz.second;
          }
        }
        // delinearize b, l, r and put data.
        for (uint64_t ws_iter = 0; ws_iter < sample_rightex.get_linear_bound();
             ws_iter++) {
          if (workspace[ws_iter] != RES()) {
            // put data
            CompactCordinate res_cord =
                CompactCordinate(batch_iter, sample_batch, left_slice_l.first,
                                 sample_leftex, ws_iter, sample_rightex, 0);
            result_tensor.push_nnz(workspace[ws_iter], res_cord);
          }
        }
        memset(workspace, 0, sample_rightex.get_linear_bound() * sizeof(RES));
      }
    }
    return result_tensor;
  }

//template <class RES, class RIGHT>
//  ListTensor<RES> multiply_3d_parallel(FastccTensor<RIGHT> &other, CoOrdinate left_batch,
//                              CoOrdinate left_contr, CoOrdinate left_ex,
//                              CoOrdinate right_batch, CoOrdinate right_contr,
//                              CoOrdinate right_ex) {
//    // always run 4D loop structure, if the coordinates are empty then the
//    // iteration is a singleton.
//    // for b
//    //    for l
//    //       for c
//    //           for r
//    BoundedCoordinate sample_batch =
//        this->nonzeros[0].get_coords().gather(left_batch).get_bounded();
//    BoundedCoordinate sample_rightex =
//        other.nonzeros[0].get_coords().gather(right_ex).get_bounded();
//    BoundedCoordinate sample_leftex =
//        this->nonzeros[0].get_coords().gather(left_ex).get_bounded();
//    uint64_t batch_max = sample_batch.get_linear_bound();
//    if (batch_max == 1) {
//      assert(left_batch.get_dimensionality() == 0);
//      assert(right_batch.get_dimensionality() == 0);
//    } else {
//      assert(left_batch.get_dimensionality() > 0);
//      assert(right_batch.get_dimensionality() ==
//             left_batch.get_dimensionality());
//    }
//    if (left_contr.get_dimensionality() != 0) {
//      assert(this->get_nonzeros()[0]
//                 .get_coords()
//                 .gather(left_contr)
//                 .get_linearized_max() == other.get_nonzeros()[0]
//                                              .get_coords()
//                                              .gather(right_contr)
//                                              .get_linearized_max());
//    }
//    InputTensorMap3D<DT> left_indexed =
//        InputTensorMap3D<DT>(*this, left_batch, left_ex, left_contr, batch_max);
//    InputTensorMap3D<RIGHT> right_indexed = InputTensorMap3D<RIGHT>(
//        other, right_batch, right_contr, right_ex, batch_max);
//    int num_workers = std::thread::hardware_concurrency() / 2;
//    init_heaps(num_workers);
//
//    RES** workspaces = (RES**)malloc(num_workers * sizeof(RES*));
//    ListTensor<RES>* result_tensors = (ListTensor<RES>*)malloc(num_workers * sizeof(ListTensor<RES>));
//    for (int _titer = 0; _titer < num_workers; _titer++) {
//      workspaces[_titer] =
//          (RES *)calloc(sample_rightex.get_linear_bound(), sizeof(RES));
//      result_tensors[_titer] =
//          ListTensor<RES>(sample_batch.get_dimensionality() +
//                              sample_leftex.get_dimensionality() +
//                              sample_rightex.get_dimensionality(),
//                          _titer);
//    }
//    tf::Taskflow taskflow;
//    tf::Executor executor(num_workers);
//
//    for (uint64_t batch_iter = 0; batch_iter < batch_max; batch_iter++) {
//      for (auto &left_slice_l :
//           left_indexed
//               .indexed_tensor[batch_iter]) { // left_slice_l is type of key
//                                              // value pair of hashmap
//        taskflow.emplace([&, batch_iter, left_slice_l]() mutable {
//          int my_id = executor.this_worker_id();
//          for (auto &left_slice_c :
//               left_slice_l.second) { // left_slice_c is type of pair of c and
//                                      // data nnz (iterator of vector)
//            auto right_iter_r = right_indexed.indexed_tensor[batch_iter].find(
//                left_slice_c.first);
//            if (right_iter_r ==
//                right_indexed.indexed_tensor[batch_iter].end()) {
//              continue;
//            }
//            for (auto &right_nnz : right_iter_r->second) {
//              auto ws_index =
//                  right_nnz.first == uint64_t(-1) ? 0 : right_nnz.first;
//              workspaces[my_id][ws_index] +=
//                  left_slice_c.second * right_nnz.second;
//            }
//          }
//          // delinearize b, l, r and put data.
//          for (uint64_t ws_iter = 0;
//               ws_iter < sample_rightex.get_linear_bound(); ws_iter++) {
//            if (workspaces[my_id][ws_iter] != RES()) {
//              // put data
//              CompactCordinate res_cord =
//                  CompactCordinate(batch_iter, sample_batch, left_slice_l.first,
//                                   sample_leftex, ws_iter, sample_rightex, my_id);
//              result_tensors[my_id].push_nnz(workspaces[my_id][ws_iter],
//                                             res_cord);
//            }
//          }
//          memset(workspaces[my_id], 0,
//                 sample_rightex.get_linear_bound() * sizeof(RES));
//        });
//      }
//    }
//    executor.run(taskflow).wait();
//    for (int iter = 1; iter < num_workers; iter++) {
//      result_tensors[0].concatenate(result_tensors[iter]);
//    }
//
//    return result_tensors[0];
//  }


  // TODO redo this with indexed tensor maybe.
  // In-place eltwise operations
  // For sparse tensors, the += can in-fact increase non-zeros.
#define OVERLOAD_OP(OP)                                                        \
  void operator OP(Tensor<DT> *other) {                                        \
    hashmap_vals indexed_tensor;                                               \
    for (auto &nnz : nonzeros) {                                               \
      indexed_tensor[nnz.get_coords()] = nnz.get_data();                       \
    }                                                                          \
    nonzeros.clear();                                                          \
    for (auto &nnz : (*other)) {                                               \
      auto ref = indexed_tensor.find(nnz.get_coords());                        \
      if (ref != indexed_tensor.end()) {                                       \
        ref.value() OP nnz.get_data();                                         \
      } else {                                                                 \
        nonzeros.push_back(nnz);                                               \
      }                                                                        \
    }                                                                          \
    for (auto &entry : indexed_tensor) {                                       \
      nonzeros.push_back(NNZ<DT>(entry.second, entry.first));                  \
    }                                                                          \
  }
  DT operator[](CoOrdinate cord) {
    for (auto &nnz : nonzeros) {
      if (nnz.get_coords() == cord) {
        return nnz.get_data();
      }
    }
    return DT();
  }
  void sort_nnz(){
    std::sort(nonzeros.begin(), nonzeros.end(), [](NNZ<DT> &a, NNZ<DT> &b) {
      return a.get_coords() < b.get_coords();
    });
  }
  bool operator==(const FastccTensor<DT> &other) const{
      //EXPECTS SORTED TENSORS
    if (this->get_dimensionality() != other.get_dimensionality()) {
      return false;
    }
    if (this->get_size() != other.get_size()) {
      return false;
    }
    for (int iter = 0; iter < this->get_size(); iter++) {
      if (!(nnz_at(iter) == other.nnz_at(iter))) {
        return false;
      }
    }
    return true;
  }

};

template <class LEFT>
template <class RES, class RIGHT>
ListTensor<RES>
ListTensor<LEFT>::multiply_3d(ListTensor<RIGHT> &other, CoOrdinate left_batch,
                              CoOrdinate left_contr, CoOrdinate left_ex,
                              CoOrdinate right_batch, CoOrdinate right_contr,
                              CoOrdinate right_ex) {
  // always run 4D loop structure, if the coordinates are empty then the
  // iteration is a singleton.
  // for b
  //    for l
  //       for c
  //           for r
  BoundedCoordinate sample_batch =
      this->nonzeros[0].get_coords().gather(left_batch).get_bounded();
  BoundedCoordinate sample_rightex =
      other.nonzeros[0].get_coords().gather(right_ex).get_bounded();
  BoundedCoordinate sample_leftex =
      this->nonzeros[0].get_coords().gather(left_ex).get_bounded();
  uint64_t batch_max = sample_batch.get_linear_bound();
  if (batch_max == 1) {
    assert(left_batch.get_dimensionality() == 0);
    assert(right_batch.get_dimensionality() == 0);
  } else {
    assert(left_batch.get_dimensionality() > 0);
    assert(right_batch.get_dimensionality() == left_batch.get_dimensionality());
  }
  if (left_contr.get_dimensionality() != 0) {
    assert(this->get_nonzeros()[0]
               .get_coords()
               .gather(left_contr)
               .get_linearized_max() == other.get_nonzeros()[0]
                                            .get_coords()
                                            .gather(right_contr)
                                            .get_linearized_max());
  }
  InputTensorMap3D<LEFT> left_indexed =
      InputTensorMap3D<LEFT>(*this, left_batch, left_ex, left_contr, batch_max);
  InputTensorMap3D<RIGHT> right_indexed = InputTensorMap3D<RIGHT>(
      other, right_batch, right_contr, right_ex, batch_max);
  init_heaps(1);

  RES *workspace =
      (RES *)calloc(sample_rightex.get_linear_bound(), sizeof(RES));
  ListTensor<RES> result_tensor(sample_batch.get_dimensionality() +
                                    sample_leftex.get_dimensionality() +
                                    sample_rightex.get_dimensionality(),
                                0);

  for (uint64_t batch_iter = 0; batch_iter < batch_max; batch_iter++) {
    for (auto &left_slice_l :
         left_indexed.indexed_tensor[batch_iter]) { // left_slice_l is type of
                                                    // key value pair of hashmap
      for (auto &left_slice_c :
           left_slice_l.second) { // left_slice_c is type of pair of c and data
                                  // nnz (iterator of vector)
        auto right_iter_r =
            right_indexed.indexed_tensor[batch_iter].find(left_slice_c.first);
        if (right_iter_r == right_indexed.indexed_tensor[batch_iter].end()) {
          continue;
        }
        for (auto &right_nnz : right_iter_r->second) {
          auto ws_index = right_nnz.first == uint64_t(-1) ? 0 : right_nnz.first;
          workspace[ws_index] += left_slice_c.second * right_nnz.second;
        }
      }
      // delinearize b, l, r and put data.
      for (uint64_t ws_iter = 0; ws_iter < sample_rightex.get_linear_bound();
           ws_iter++) {
        if (workspace[ws_iter] != RES()) {
          // put data
          CompactCordinate res_cord =
              CompactCordinate(batch_iter, sample_batch, left_slice_l.first,
                               sample_leftex, ws_iter, sample_rightex);
          result_tensor.push_nnz(workspace[ws_iter], res_cord);
        }
      }
      memset(workspace, 0, sample_rightex.get_linear_bound() * sizeof(RES));
    }
  }
  return result_tensor;
}

#endif
