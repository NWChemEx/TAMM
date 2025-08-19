#include "coordinate.hpp"
#include "timer.hpp"
#include <omp.h>
#include <forward_list>
#include <fstream>
#include <list>
#include <random>
#include <cmath>
namespace tamm::fastcc{
template<class DT> class NNZ;

template <class DT> class CompactNNZ {
  CompactCordinate cord;
  DT data;

public:
  CompactNNZ(DT data, CompactCordinate cord) : data(data), cord(cord) {}
  DT get_data() const { return data; }
  CompactCordinate get_cord() const { return cord; }
  std::string to_string() {
      if constexpr (std::is_floating_point<DT>::value) {
          return cord.to_string() + ": " + std::to_string(data);
      }
      else {
          exit(1);
      }
    //return cord.to_string() + ": " + std::to_string(data);
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
  int* shape = nullptr;

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
  void set_shape(std::vector<int> &some_shape) {
    this->shape = new int[some_shape.size()];
    for (int i = 0; i < some_shape.size(); i++) {
        this->shape[i] = some_shape[i];
    }
  }
  void set_shape(int* some_shape) {
    this->shape = some_shape;
  }
  int* get_shape(){ // the pointer she told you not to worry about
      return shape;
  }
  NNZNode<DT>* get_head(){ // not safe, but when you really need head
      return head;
  }
  int compute_nnz_count(){
      return count;
  }
  void write_to_pointer(DT* destination, std::vector<int> gather_positions = {}){
      if(this->head == nullptr){
          std::cerr << "ListTensor is empty, cannot write to pointer" << std::endl;
          exit(1);
      }
      if(this->shape == nullptr){
          std::cerr << "Shape is not set, cannot write to pointer" << std::endl;
          exit(1);
      }
      if(gather_positions.size() > 0) {
          BoundedPosition gather(gather_positions);
          for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()) {
            destination[current->get_nnz().get_cord().gather_linearize(gather, shape)] =
              current->get_nnz().get_data();
          }
      }
      else {
          for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()) {
            destination[current->get_nnz().get_cord().linearize(shape)] =
              current->get_nnz().get_data();
          }
      }
  }
  int run_through_nnz(){
      if(this->head == nullptr){
          return 0;
      }
      int count = 0;
      for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()){
          count++;
      }
      return count;
  }
  CompactCordinate get_cord_at(int index){
      NNZNode<DT>* current = head;
      for(int i = 0; i < index; i++){
          current = current->get_next();
          if (current == nullptr){
              std::cerr << "Index out of bounds to get coordinate out of list tensor" << std::endl;
              exit(1);
          }
      }
      return current->get_nnz().get_cord();
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
  void write(std::string filename){
      std::ofstream myfile(filename);
      myfile.open(filename);
      for(int i = 0; i < this->get_dimensionality(); i++){
          myfile << this->shape[i] << " ";
      }
      myfile << "\n";
      myfile <<  this->to_string();
      myfile.close();
  }
      
  //void write(std::string filename){
  //    std::ofstream myfile(filename);
  //    myfile.open(filename);
  //    for(int i = 0; i < this->get_dimensionality(); i++){
  //        myfile << this->shape[i] << " ";
  //    }
  //    myfile << "\n";
  //    myfile <<  this->to_string();
  //    myfile.close();
  //}
  template<class RES, class OTHER>
  ListTensor<RES> multiply_3d(FastccTensor<OTHER>& other, BoundedPosition left_batch,
                              BoundedPosition left_contr, BoundedPosition left_ex,
                              CoOrdinate right_batch, CoOrdinate right_contr,
                              CoOrdinate right_ex);
  template<class RES, class OTHER>
  ListTensor<RES> multiply_3d(ListTensor<OTHER>& other, BoundedPosition left_batch,
                              BoundedPosition left_contr, BoundedPosition left_ex,
                              BoundedPosition right_batch, BoundedPosition right_contr,
                              BoundedPosition right_ex);
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
      result._infer_dimensionality();
      std::vector<int> my_shape;
      if(this->shape != nullptr){
          assert(this->dimensionality > 0);
          for(int i = 0; i < this->get_dimensionality(); i++){
              my_shape.push_back(this->shape[i]);
          }
          result.set_shape(my_shape);
      } else{
          result._infer_shape();
      }
      
      return result;
  }
  // drops the references held by this tensor.
  void drop() {
    head = nullptr;
    tail = nullptr;
    count = 0;
    thread_id = 0;
    dimensionality = 0;
    shape = nullptr;
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

  InputTensorMap3D(ListTensor<DT> &base, BoundedPosition outermost, BoundedPosition middle, BoundedPosition lowest, uint64_t max_outermost_val){
      assert(base.get_shape() != nullptr);
    indexed_tensor = (outermost_type)calloc(max_outermost_val, sizeof(middle_type));
    if(outermost.get_dimensionality() == 0){
      assert(max_outermost_val == 1);
    }
    for(int _i = 0; _i < max_outermost_val; _i++){
      indexed_tensor[_i] = middle_type();
    }
    for(NNZNode<DT>* current = base.get_head(); current != nullptr; current = current->get_next()){
      uint64_t outer_index = 0;
      if(outermost.get_dimensionality() > 0){
        outer_index = current->get_nnz().get_cord().gather_linearize(outermost, base.get_shape());
      }
      uint64_t middle_index = DNE;
      if(middle.get_dimensionality() > 0){
        middle_index = current->get_nnz().get_cord().gather_linearize(middle, base.get_shape());
      }
      uint64_t lowest_index = DNE;
      if(lowest.get_dimensionality() > 0){
        lowest_index = current->get_nnz().get_cord().gather_linearize(lowest, base.get_shape());
      }
      middle_type &middle_slice = indexed_tensor[outer_index];
      auto lowest_iter = middle_slice.find(middle_index);
      if(lowest_iter != middle_slice.end()){
        lowest_iter->second.push_back({lowest_index, current->get_nnz().get_data()});
      } else {
        lowest_type new_lowest = {{lowest_index, current->get_nnz().get_data()}};
        middle_slice[middle_index] = new_lowest;
      }
    }
  }
};
}
