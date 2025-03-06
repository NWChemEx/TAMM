#include "coordinate.hpp"
#include "timer.hpp"
#include <omp.h>
#include <forward_list>
#include <list>
#include <random>
#include <cmath>
template<class DT> class NNZ;

template <class DT> class CompactNNZ {
  CompactCordinate cord;
  DT data;

public:
  CompactNNZ(DT data, CompactCordinate cord) : data(data), cord(cord) {}
  DT get_data() const { return data; }
  CompactCordinate get_cord() const { return cord; }
  std::string to_string() {
    return cord.to_string() + ": " + std::to_string(data);
  }
};

template <class DT> class FastccTensor;
template <class DT> class IndexedTensor;

template <class DT> class NNZNode {
private:
  CompactNNZ<DT> nnz;
  NNZNode<DT> *next = nullptr;

public:
  NNZNode(DT data, CompactCordinate cord) : nnz(data, cord), next(nullptr) {}
  NNZNode<DT> *get_next() { return next; }
  void set_next(NNZNode<DT> *next) { this->next = next; }
  CompactNNZ<DT>const & get_nnz() { return nnz; }
  int get_dimensionality() { return nnz.get_cord().get_dimensionality(); }
  std::string to_string() { return nnz.to_string(); }
};

template <class DT> class ListTensor {
  NNZNode<DT>* head = nullptr;
  NNZNode<DT>* tail = nullptr;
  int dimensionality = 0;
  int thread_id = 0;
  uint64_t count = 0;

public:
  ListTensor(int dimensionality = 0, int thread_id=0):dimensionality(dimensionality), thread_id(thread_id) {}

  void push_nnz(DT data, CompactCordinate cord) {
    NNZNode<DT>* new_node = (NNZNode<DT>*)my_malloc(sizeof(NNZNode<DT>), thread_id);
    *new_node = NNZNode<DT>(data, cord);
    count++;
    if(head == tail && head == nullptr){
        head = new_node;
        tail = new_node;
    } else if(head == tail){
        head->set_next(new_node);
        tail = new_node;
    } else {
        tail->set_next(new_node);
        tail = new_node;
    }
  }
  int compute_nnz_count(){
      return count;
  }
  int run_through_nnz(){
      int count = 0;
      for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()){
          count++;
      }
      return count;
  }
  int get_dimensionality(){
      if(this->dimensionality == 0){
          this->dimensionality = this->head->get_dimensionality();
      }
      return dimensionality;
  }
  void concatenate(ListTensor& other){
      if(this->tail == nullptr){
          assert(this->head == nullptr && this->count == 0);
          this->head = other.head;
          this->tail = other.tail;
          this->count = other.count;
          return;
      }
      this->tail->set_next(other.head);
      if(other.tail != nullptr){
          this->tail = other.tail;
      }
      count += other.count;
  }
  std::string to_string(){
      std::string str = "";
      for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()){
          str += current->to_string() + "\n";
      }
      return str;
  }
  template<class RES, class OTHER>
  ListTensor<RES> multiply_3d(ListTensor<OTHER>& other, CoOrdinate left_batch,
                              CoOrdinate left_contr, CoOrdinate left_ex,
                              CoOrdinate right_batch, CoOrdinate right_contr,
                              CoOrdinate right_ex);
  FastccTensor<DT> to_tensor() {
      FastccTensor<DT> result;
      for (NNZNode<DT> *current = head; current != nullptr;
           current = current->get_next()) {
          NNZ<DT> nnz = NNZ<DT>(current->get_nnz().get_data(),
                                current->get_nnz().get_cord().as_coordinate());
          result.get_nonzeros().push_back(nnz);
      }
      assert(this->compute_nnz_count() == result.get_size());
      assert(this->run_through_nnz() == result.get_size());
      
      return result;
  }
};


template <class DT> class NNZ {
  DT data;
  CoOrdinate coords = CoOrdinate(0, nullptr);

public:
  // Constructor for a random value and coordinates
  NNZ(int dimensionality, int *shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disf(0.4, 1);
    std::uniform_int_distribution<> dis_cords[dimensionality];
    for (int i = 0; i < dimensionality; i++) {
      dis_cords[i] = std::uniform_int_distribution<>(0, shape[i]);
    }
    std::vector<int> temp_coords;
    this->data = disf(gen);
    for (int i = 0; i < dimensionality; i++) {
      temp_coords.push_back(dis_cords[i](gen));
    }
    this->coords = CoOrdinate(temp_coords);
  }
  std::string to_string() const {
    std::string str = "";
    for (int i = 0; i < this->coords.get_dimensionality(); i++) {
      str += std::to_string(this->coords.get_index(i)) + " ";
    }
    str += std::to_string(data);
    return str;
  }
  int get_index(int dim) { return coords.get_index(dim); }
  DT get_data() { return data; }
  void set_zero(){
      if constexpr(std::is_floating_point<DT>::value){
          data = 0.0;
      }


  }
  void operator+=(DT other) {
    data += other;
    if constexpr (std::is_class<DT>::value) {
      other.free();
    }
  }

  CoOrdinate &get_coords() { return coords; }

  // Constructor for a given value and coordinates
  NNZ(DT data, int dimensionality, int *coords)
      : data(data), coords(dimensionality, coords) {}
  NNZ(DT data, CoOrdinate coords) : data(data), coords(coords) {}
  NNZ(DT data, BoundedCoordinate bc) : data(data) {
    std::vector<int> vecords;
    for (int i = 0; i < bc.get_dimensionality(); i++) {
      vecords.push_back(bc.get_coordinate(i));
    }
    coords = CoOrdinate(vecords);
  }
  bool operator==(const NNZ &other) const {
    return data == other.data && coords == other.coords;
  }
};

template <class DT> class InputTensorMap3D {
#define DNE uint64_t(-1)
public:
  using lowest_type = std::vector<std::pair<uint64_t, DT>>;
  using middle_type = std::unordered_map<uint64_t, lowest_type>;
  using outermost_type = middle_type*;
  outermost_type indexed_tensor;
  // Pass in max outer value as 1 if it's a degenerate dimension.
  InputTensorMap3D(FastccTensor<DT>& base, CoOrdinate outermost, CoOrdinate middle, CoOrdinate lowest, uint64_t max_outermost_val){
    indexed_tensor = (outermost_type)calloc(max_outermost_val, sizeof(middle_type));
    if(outermost.get_dimensionality() == 0){
      assert(max_outermost_val == 1);
    }
    for(int _i = 0; _i < max_outermost_val; _i++){
      indexed_tensor[_i] = middle_type();
    }
    for(auto &nnz : base){
      uint64_t outer_index = 0;
      if(outermost.get_dimensionality() > 0){
        outer_index = nnz.get_coords().gather_linearize(outermost);
      }
      uint64_t middle_index = DNE;
      if(middle.get_dimensionality() > 0){
        middle_index = nnz.get_coords().gather_linearize(middle);
      }
      uint64_t lowest_index = DNE;
      if(lowest.get_dimensionality() > 0){
        lowest_index = nnz.get_coords().gather_linearize(lowest);
      }
      middle_type &middle_slice = indexed_tensor[outer_index];
      auto lowest_iter = middle_slice.find(middle_index);
      if(lowest_iter != middle_slice.end()){
        lowest_iter->second.push_back({lowest_index, nnz.get_data()});
      } else {
        middle_slice[middle_index] = {{lowest_index, nnz.get_data()}};
      }
    }
  }
};
