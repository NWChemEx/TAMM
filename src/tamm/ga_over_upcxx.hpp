#pragma once

#if defined(USE_UPCXX)
#include <cassert>
#include <limits>
#include <upcxx/upcxx.hpp>

template<typename T = double>
class ga_over_upcxx_chunk_view {
private:
  T*      chunk;
  int64_t chunk_size[4];

  inline int64_t flatten(const int64_t i0_offset, const int64_t i1_offset, const int64_t i2_offset,
                         const int64_t i3_offset) const {
    return i0_offset * chunk_size[1] * chunk_size[2] * chunk_size[3] +
           i1_offset * chunk_size[2] * chunk_size[3] + i2_offset * chunk_size[3] + i3_offset;
  }

public:
  ga_over_upcxx_chunk_view(T* _chunk, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) {
    chunk         = _chunk;
    chunk_size[0] = dim0;
    chunk_size[1] = dim1;
    chunk_size[2] = dim2;
    chunk_size[3] = dim3;
  }

  inline T read(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3) const {
    return chunk[flatten(i0, i1, i2, i3)];
  }

  inline void write(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3, T val) {
    chunk[flatten(i0, i1, i2, i3)] = val;
  }

  inline void subtract(const int64_t i0, const int64_t i1, const int64_t i2, const int64_t i3,
                       T val) {
    chunk[flatten(i0, i1, i2, i3)] -= val;
  }

  inline int64_t get_chunk_size(int index) const { return chunk_size[index]; }
};

template<typename T = double>
class ga_over_upcxx_chunk {
private:
  int64_t              lo_coord[4];
  int64_t              chunk_size[4];
  upcxx::global_ptr<T> chunk;

public:
  ga_over_upcxx_chunk(int64_t lo0, int64_t lo1, int64_t lo2, int64_t lo3, int64_t chunk_size0,
                      int64_t chunk_size1, int64_t chunk_size2, int64_t chunk_size3,
                      upcxx::global_ptr<T> _chunk) {
    lo_coord[0] = lo0;
    lo_coord[1] = lo1;
    lo_coord[2] = lo2;
    lo_coord[3] = lo3;

    chunk_size[0] = chunk_size0;
    chunk_size[1] = chunk_size1;
    chunk_size[2] = chunk_size2;
    chunk_size[3] = chunk_size3;

    chunk = _chunk;
  }

  void destroy() { upcxx::delete_array(chunk); }

  T* local() {
    assert(chunk.is_local());
    return chunk.local();
  }

  ga_over_upcxx_chunk_view<T> local_view() {
    if(!chunk.is_local()) abort();
    return ga_over_upcxx_chunk_view(chunk.local(), chunk_size[0], chunk_size[1], chunk_size[2],
                                    chunk_size[3]);
  }

  void check_bounds_global(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const {
    if(!(i0 >= lo_coord[0] && i1 >= lo_coord[1] && i2 >= lo_coord[2]) && i3 >= lo_coord[3]) {
      abort();
    }
    if(!(i0 < lo_coord[0] + chunk_size[0] && i1 < lo_coord[1] + chunk_size[1] &&
         i2 < lo_coord[2] + chunk_size[2] && i3 < lo_coord[3] + chunk_size[3])) {
      abort();
    }
  }

  void check_bounds_local(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const {
    if(!(i0 >= 0 && i1 >= 0 && i2 >= 0 && i3 >= 0)) { abort(); }
    if(!(i0 < chunk_size[0] && i1 < chunk_size[1] && i2 < chunk_size[2] && i3 < chunk_size[3])) {
      abort();
    }
  }

  bool contains(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const {
    return i0 >= lo_coord[0] && i1 >= lo_coord[1] && i2 >= lo_coord[2] && i3 >= lo_coord[3] &&
           i0 < lo_coord[0] + chunk_size[0] && i1 < lo_coord[1] + chunk_size[1] &&
           i2 < lo_coord[2] + chunk_size[2] && i3 < lo_coord[3] + chunk_size[3];
  }

  inline int64_t flatten(const int64_t i0_offset, const int64_t i1_offset, const int64_t i2_offset,
                         const int64_t i3_offset) const {
    return i0_offset * chunk_size[1] * chunk_size[2] * chunk_size[3] +
           i1_offset * chunk_size[2] * chunk_size[3] + i2_offset * chunk_size[3] + i3_offset;
  }

  // Dimensions of in are high0-low0+1, high1-low1+1, high2-low2+1
  upcxx::future<> put_any(int64_t low0, int64_t low1, int64_t low2, int64_t low3, int64_t high0,
                          int64_t high1, int64_t high2, int64_t high3, T* buf) {
    return putget_any(low0, low1, low2, low3, high0, high1, high2, high3, buf, true);
  }

  // Dimensions of in are high0-low0+1, high1-low1+1, high2-low2+1
  upcxx::future<> get_any(int64_t low0, int64_t low1, int64_t low2, int64_t low3, int64_t high0,
                          int64_t high1, int64_t high2, int64_t high3, T* buf) {
    return putget_any(low0, low1, low2, low3, high0, high1, high2, high3, buf, false);
  }

  // Dimensions of in are high0-low0+1, high1-low1+1, high2-low2+1
  upcxx::future<> putget_any(int64_t low0, int64_t low1, int64_t low2, int64_t low3, int64_t high0,
                             int64_t high1, int64_t high2, int64_t high3, T* buf, bool is_put) {
    upcxx::future<> fut      = upcxx::make_future();
    bool            overlap0 = (low0 <= (lo_coord[0] + chunk_size[0] - 1) && lo_coord[0] <= high0);
    bool            overlap1 = (low1 <= (lo_coord[1] + chunk_size[1] - 1) && lo_coord[1] <= high1);
    bool            overlap2 = (low2 <= (lo_coord[2] + chunk_size[2] - 1) && lo_coord[2] <= high2);
    bool            overlap3 = (low3 <= (lo_coord[3] + chunk_size[3] - 1) && lo_coord[3] <= high3);
    if(!overlap0 || !overlap1 || !overlap2 || !overlap3) { return fut; }

    int64_t this_low0  = std::max(low0, lo_coord[0]);
    int64_t this_high0 = std::min(high0, lo_coord[0] + chunk_size[0] - 1);
    int64_t this_low1  = std::max(low1, lo_coord[1]);
    int64_t this_high1 = std::min(high1, lo_coord[1] + chunk_size[1] - 1);
    int64_t this_low2  = std::max(low2, lo_coord[2]);
    int64_t this_high2 = std::min(high2, lo_coord[2] + chunk_size[2] - 1);
    int64_t this_low3  = std::max(low3, lo_coord[3]);
    int64_t this_high3 = std::min(high3, lo_coord[3] + chunk_size[3] - 1);
    assert(this_low0 <= this_high0);
    assert(this_low1 <= this_high1);
    assert(this_low2 <= this_high2);
    assert(this_low3 <= this_high3);

    int64_t buf_offset0 = this_low0 - low0;
    int64_t buf_offset1 = this_low1 - low1;
    int64_t buf_offset2 = this_low2 - low2;
    int64_t buf_offset3 = this_low3 - low3;

    int64_t chunk_offset0 = this_low0 - lo_coord[0];
    int64_t chunk_offset1 = this_low1 - lo_coord[1];
    int64_t chunk_offset2 = this_low2 - lo_coord[2];
    int64_t chunk_offset3 = this_low3 - lo_coord[3];

    int64_t buf_dim0 = high0 - low0 + 1;
    int64_t buf_dim1 = high1 - low1 + 1;
    int64_t buf_dim2 = high2 - low2 + 1;
    int64_t buf_dim3 = high3 - low3 + 1;

    if(this_low3 == lo_coord[3] && this_high3 == lo_coord[3] + chunk_size[3] - 1 &&
       this_low2 == lo_coord[2] && this_high2 == lo_coord[2] + chunk_size[2] - 1 &&
       this_low1 == lo_coord[1] && this_high1 == lo_coord[1] + chunk_size[1] - 1 &&
       this_low3 == low3 && this_high3 == high3 && this_low2 == low2 && this_high2 == high2 &&
       this_low1 == low1 && this_high1 == high1) {
      /*
       * Sending whole chunk or some subset off the outermost
       * dimension. Do a single put of:
       *     [this_low0:this_high0,
       *      this_low1:this_high1,
       *      this_low2:this_high2]
       */
      if(is_put) {
        fut = upcxx::when_all(
          fut, upcxx::rput(&buf[(this_low0 - low0) * buf_dim1 * buf_dim2 * buf_dim3],
                           chunk + flatten(chunk_offset0, 0, 0, 0),
                           (this_high0 - this_low0 + 1) * buf_dim1 * buf_dim2 * buf_dim3));
      }
      else {
        fut = upcxx::when_all(
          fut, upcxx::rget(chunk + flatten(chunk_offset0, 0, 0, 0),
                           &buf[(this_low0 - low0) * buf_dim1 * buf_dim2 * buf_dim3],
                           (this_high0 - this_low0 + 1) * buf_dim1 * buf_dim2 * buf_dim3));
      }
    }
    else if(this_low3 == lo_coord[3] && this_high3 == lo_coord[3] + chunk_size[3] - 1 &&
            this_low2 == lo_coord[2] && this_high2 == lo_coord[2] + chunk_size[2] - 1 &&
            this_low3 == low3 && this_high3 == high3 && this_low2 == low2 && this_high2 == high2) {
      // Sending contiguous chunks of innermost two dimensions
      for(int64_t i = this_low0; i <= this_high0; i++) {
        if(is_put) {
          fut = upcxx::when_all(
            fut,
            upcxx::rput(
              &buf[(i - low0) * buf_dim1 * buf_dim2 * buf_dim3 + buf_offset1 * buf_dim2 * buf_dim3],
              chunk + flatten(i - lo_coord[0], chunk_offset1, 0, 0),
              (this_high1 - this_low1 + 1) * buf_dim2 * buf_dim3));
        }
        else {
          fut = upcxx::when_all(
            fut,
            upcxx::rget(
              chunk + flatten(i - lo_coord[0], chunk_offset1, 0, 0),
              &buf[(i - low0) * buf_dim1 * buf_dim2 * buf_dim3 + buf_offset1 * buf_dim2 * buf_dim3],
              (this_high1 - this_low1 + 1) * buf_dim2 * buf_dim3));
        }
      }
    }
    else {
      for(int64_t i = this_low0; i <= this_high0; i++) {
        for(int64_t j = this_low1; j <= this_high1; j++) {
          for(int64_t k = this_low2; k <= this_high2; k++) {
            if(is_put) {
              fut = upcxx::when_all(
                fut, upcxx::rput(&buf[(i - low0) * buf_dim1 * buf_dim2 * buf_dim3 +
                                      (j - low1) * buf_dim2 * buf_dim3 + (k - low2) * buf_dim3 +
                                      (this_low3 - low3)],
                                 chunk + flatten(i - lo_coord[0], j - lo_coord[1], k - lo_coord[2],
                                                 this_low3 - lo_coord[3]),
                                 this_high3 - this_low3 + 1));
            }
            else {
              fut = upcxx::when_all(
                fut, upcxx::rget(chunk + flatten(i - lo_coord[0], j - lo_coord[1], k - lo_coord[2],
                                                 this_low3 - lo_coord[3]),
                                 &buf[(i - low0) * buf_dim1 * buf_dim2 * buf_dim3 +
                                      (j - low1) * buf_dim2 * buf_dim3 + (k - low2) * buf_dim3 +
                                      (this_low3 - low3)],
                                 this_high3 - this_low3 + 1));
            }
          }
        }
      }
    }

    return fut;
  }

  void subtract(int64_t i0_off, int64_t i1_off, int64_t i2_off, int64_t i3_off, double val) {
    if(!chunk.is_local()) abort();

    double* ptr = chunk.local();
    ptr[flatten(i0_off, i1_off, i2_off, i3_off)] -= val;
  }

  void add(int64_t i0_off, int64_t i1_off, int64_t i2_off, int64_t i3_off, double val) {
    if(!chunk.is_local()) abort();

    double* ptr = chunk.local();
    ptr[flatten(i0_off, i1_off, i2_off, i3_off)] += val;
  }

  void minimum(double& out_val, int64_t& out_offset0, int64_t& out_offset1, int64_t& out_offset2,
               int64_t& out_offset3) {
    out_val = -1.0 * std::numeric_limits<double>::max();

    double* ptr = chunk.local();
    for(int i = 0; i < chunk_size[0]; i++) {
      for(int j = 0; j < chunk_size[1]; j++) {
        for(int k = 0; k < chunk_size[2]; k++) {
          for(int l = 0; l < chunk_size[3]; l++) {
            if(ptr[flatten(i, j, k, l)] < out_val) {
              out_val     = ptr[flatten(i, j, k, l)];
              out_offset0 = i;
              out_offset1 = j;
              out_offset2 = k;
              out_offset3 = l;
            }
          }
        }
      }
    }
  }

  void maximum(double& out_val, int64_t& out_i0, int64_t& out_i1, int64_t& out_i2,
               int64_t& out_i3) {
    out_val            = -1.0 * std::numeric_limits<double>::max();
    unsigned max_index = 0;

    const double*  ptr             = chunk.local();
    const unsigned chunk_nelements = chunk_size[0] * chunk_size[1] * chunk_size[2] * chunk_size[3];
    for(unsigned i = 0; i < chunk_nelements; i++) {
      if(out_val < ptr[i]) {
        out_val   = ptr[i];
        max_index = i;
      }
    }

    out_i0 = max_index / (chunk_size[1] * chunk_size[2] * chunk_size[3]);
    out_i1 = (max_index / (chunk_size[2] * chunk_size[3])) % chunk_size[1];
    out_i2 = (max_index / chunk_size[3]) % chunk_size[2];
    out_i3 = (max_index % chunk_size[3]);
  }

  int64_t* get_lo_coord() { return lo_coord; }
  int64_t  get_lo_coord(int dim) { return lo_coord[dim]; }
  int64_t* get_chunk_size() { return chunk_size; }
  int64_t  get_chunk_size(int dim) { return chunk_size[dim]; }
  int64_t  get_nelements() { return chunk_size[0] * chunk_size[1] * chunk_size[2] * chunk_size[3]; }

  bool same_coord(ga_over_upcxx_chunk<T>* other) {
    return lo_coord[0] == other->lo_coord[0] && lo_coord[1] == other->lo_coord[1] &&
           lo_coord[2] == other->lo_coord[2] && lo_coord[3] == other->lo_coord[3];
  }

  bool same_size_or_smaller(ga_over_upcxx_chunk<T>* other) {
    return chunk_size[0] <= other->chunk_size[0] && chunk_size[1] <= other->chunk_size[1] &&
           chunk_size[2] <= other->chunk_size[2] && chunk_size[3] <= other->chunk_size[3];
  }
};

template<typename T = double>
class ga_over_upcxx {
private:
  int rank;
  int nranks;

  int64_t        dims[4];
  int64_t        chunk_size[4];
  int64_t        chunks_per_dim[4];
  upcxx::team_id tid;

  size_t total_chunks;

  std::vector<ga_over_upcxx_chunk<T>*> local_chunks;
  std::vector<ga_over_upcxx_chunk<T>*> all_chunks;

public:
  typedef typename std::vector<ga_over_upcxx_chunk<T>*>::iterator chunk_iterator;

  ga_over_upcxx(int _ndims, int64_t* _dims, int64_t* _chunk_size, upcxx::team& t) {
    rank   = t.rank_me();
    nranks = t.rank_n();

    if(_ndims >= 1 && _ndims <= 4) {

      memcpy(dims, _dims, _ndims * sizeof(dims[0]));
      memcpy(chunk_size, _chunk_size, _ndims * sizeof(chunk_size[0]));

      for (int i = _ndims; i < 4; ++i) {
        dims[i] = 1;
        chunk_size[i] = 1;
      }
    }
    else {
      fprintf(stderr,
              "ga_over_upcxx only supports 1,2,3,4D tensors, got _ndims=%d\n", _ndims);
      abort();
    }

    tid = t.id();

    // Initial gueses for chunk_size and chunks per dimension
    for(int i = 0; i < 4; i++) {
      if(chunk_size[i] < 1) { chunk_size[i] = (dims[i] + nranks - 1) / nranks; }
      chunks_per_dim[i] = (dims[i] + chunk_size[i] - 1) / chunk_size[i];
    }

    total_chunks = chunks_per_dim[0] * chunks_per_dim[1] * chunks_per_dim[2] * chunks_per_dim[3];

    while(1) {
      // Find the largest chunks per dim
      int64_t most_chunks_per_dim     = chunks_per_dim[0];
      int     dim_most_chunks_per_dim = 0;
      for(int i = 0; i < 4; i++) {
        if(chunks_per_dim[i] > most_chunks_per_dim) {
          most_chunks_per_dim     = chunks_per_dim[i];
          dim_most_chunks_per_dim = i;
        }
      }

      // Try increasing that dimension's chunk size by 1
      int64_t proposed_chunk_size[4];
      int64_t proposed_chunks_per_dim[4];
      memcpy(proposed_chunk_size, chunk_size, 4 * sizeof(int64_t));
      proposed_chunk_size[dim_most_chunks_per_dim] += 1;
      for(int i = 0; i < 4; i++) {
        proposed_chunks_per_dim[i] =
          (dims[i] + proposed_chunk_size[i] - 1) / proposed_chunk_size[i];
      }
      int64_t proposed_total_chunks = proposed_chunks_per_dim[0] * proposed_chunks_per_dim[1] *
                                      proposed_chunks_per_dim[2] * proposed_chunks_per_dim[3];

      /*
       * If increasing that chunk size leads to fewer chunks than
       * ranks, break out.
       */
      if(proposed_total_chunks < nranks) { break; }

      // Update to the proposed chunk configuration
      memcpy(chunk_size, proposed_chunk_size, 4 * sizeof(int64_t));
      memcpy(chunks_per_dim, proposed_chunks_per_dim, 4 * sizeof(int64_t));
      total_chunks = proposed_total_chunks;
    }

    // Create each chunk on its owning rank.
    int owning_rank = 0;
    for(unsigned c = 0; c < total_chunks; c++) {
      int64_t chunk_coord[]     = {c / (chunks_per_dim[1] * chunks_per_dim[2] * chunks_per_dim[3]),
                                   (c / (chunks_per_dim[2] * chunks_per_dim[3])) % chunks_per_dim[1],
                                   (c / chunks_per_dim[3]) % chunks_per_dim[2], c % chunks_per_dim[3]};
      int64_t lo[]              = {chunk_coord[0] * chunk_size[0], chunk_coord[1] * chunk_size[1],
                                   chunk_coord[2] * chunk_size[2], chunk_coord[3] * chunk_size[3]};
      int64_t this_chunk_size[] = {chunk_size[0], chunk_size[1], chunk_size[2], chunk_size[3]};
      for(int i = 0; i < 4; i++) {
        if(lo[i] + this_chunk_size[i] > dims[i]) { this_chunk_size[i] = dims[i] - lo[i]; }
      }
      int64_t chunk_nelements =
        this_chunk_size[0] * this_chunk_size[1] * this_chunk_size[2] * this_chunk_size[3];

      upcxx::global_ptr<T> ptr;
      if(owning_rank == rank) {
        ptr = upcxx::new_array<T>(chunk_nelements);

        upcxx::dist_object<upcxx::global_ptr<T>> dobj(ptr, tid.here());
        upcxx::barrier(t);

        ga_over_upcxx_chunk<T>* new_chunk = new ga_over_upcxx_chunk<T>(
          lo[0], lo[1], lo[2], lo[3], this_chunk_size[0], this_chunk_size[1], this_chunk_size[2],
          this_chunk_size[3], ptr);
        all_chunks.push_back(new_chunk);
        local_chunks.push_back(new_chunk);
        upcxx::barrier(t);
      }
      else {
        upcxx::dist_object<upcxx::global_ptr<T>> dobj(ptr, tid.here());
        upcxx::barrier(t);
        ptr = dobj.fetch(owning_rank).wait();

        ga_over_upcxx_chunk<T>* new_chunk = new ga_over_upcxx_chunk<T>(
          lo[0], lo[1], lo[2], lo[3], this_chunk_size[0], this_chunk_size[1], this_chunk_size[2],
          this_chunk_size[3], ptr);
        all_chunks.push_back(new_chunk);
        upcxx::barrier(t);
      }
      owning_rank = (owning_rank + 1) % nranks;
    }
  }

  void destroy() {
    // upcxx::persona_scope master_scope(master_mtx,
    //         upcxx::master_persona());
    upcxx::barrier(tid.here());
    for(auto i = local_chunks.begin(), e = local_chunks.end(); i != e; i++) { (*i)->destroy(); }
    local_chunks.clear();
    all_chunks.clear();
    upcxx::barrier(tid.here());
  }

  void print() {
    assert(local_chunks.size() == 1);
    assert(all_chunks.size() == 1);
    assert(dims[3] == 1);
    for(int i = 0; i < dims[0]; i++) {
      for(int j = 0; j < dims[1]; j++) {
        for(int k = 0; k < dims[2]; k++) {
          int64_t dims[] = {1, 1, 1, 1};
          double  val;

          get(i, j, k, 0, i, j, k, 0, &val, dims);
          printf(" %.10f", val);
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }

  bool coord_is_local(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    for(auto i = local_chunks.begin(), e = local_chunks.end(); i != e; i++) {
      if((*i)->contains(i0, i1, i2, i3)) { return true; }
    }
    return false;
  }

  void zero() {
    for(auto i = local_chunks.begin(), e = local_chunks.end(); i != e; i++) {
      memset((*i)->local(), 0x00,
             (*i)->get_chunk_size(0) * (*i)->get_chunk_size(1) * (*i)->get_chunk_size(2) *
               (*i)->get_chunk_size(3) * sizeof(T));
    }
    upcxx::barrier(tid.here());
  }

  void copy(ga_over_upcxx* dst) {
    if(dims[0] != dst->dims[0] || dims[1] != dst->dims[1] || dims[2] != dst->dims[2] ||
       dims[3] != dst->dims[3]) {
      fprintf(stderr,
              "ga_over_upcxx::copy dims mismatch (%ld, %ld, "
              "%ld, %ld) (%ld, %ld, %ld, %ld)\n",
              dims[0], dims[1], dims[2], dims[3], dst->dims[0], dst->dims[1], dst->dims[2],
              dst->dims[3]);
      abort();
    }

    if(chunk_size[0] != dst->chunk_size[0] || chunk_size[1] != dst->chunk_size[1] ||
       chunk_size[2] != dst->chunk_size[2] || chunk_size[3] != dst->chunk_size[3]) {
      fprintf(stderr,
              "ga_over_upcxx::copy chunk_size mismatch (%ld, "
              "%ld, %ld, %ld) (%ld, %ld, %ld, %ld)\n",
              chunk_size[0], chunk_size[1], chunk_size[2], chunk_size[3], dst->chunk_size[0],
              dst->chunk_size[1], dst->chunk_size[2], dst->chunk_size[3]);
      abort();
    }

    if(local_chunks.size() != dst->local_chunks.size()) { abort(); }

    if(tid != dst->tid) { abort(); }

    for(unsigned i = 0; i < local_chunks.size(); i++) {
      memcpy(dst->local_chunks[i]->local(), local_chunks[i]->local(),
             local_chunks[i]->get_nelements() * sizeof(T));
    }
    upcxx::barrier(tid.here());
  }

  ga_over_upcxx_chunk<T>* find_chunk(int64_t lo0, int64_t lo1, int64_t lo2, int64_t lo3) {
    for(auto i = all_chunks.begin(), e = all_chunks.end(); i != e; i++) {
      ga_over_upcxx_chunk<T>* c = *i;
      if(c->contains(lo0, lo1, lo2, lo3)) { return c; }
    }
    return NULL;
  }

  void get(int64_t low0, int64_t low1, int64_t low2, int64_t low3, int64_t high0, int64_t high1,
           int64_t high2, int64_t high3, T* out, int64_t* out_dims) {
    upcxx::future<> fut = upcxx::make_future();
    // Check that specified output dimensions match size of fetched
    // region
    if(out_dims[0] != (high0 - low0 + 1) || out_dims[1] != (high1 - low1 + 1) ||
       out_dims[2] != (high2 - low2 + 1) || out_dims[3] != (high3 - low3 + 1)) {
      fprintf(stderr,
              "ga_over_upcxx::get size mismatch (%ld %ld "
              "%ld %ld) (%ld %ld %ld %ld)\n",
              out_dims[0], out_dims[1], out_dims[2], out_dims[3], high0 - low0 + 1,
              high1 - low1 + 1, high2 - low2 + 1, high3 - low3 + 1);
      abort();
    }

    int64_t lo_chunk_0 = low0 / chunk_size[0];
    int64_t lo_chunk_1 = low1 / chunk_size[1];
    int64_t lo_chunk_2 = low2 / chunk_size[2];
    int64_t lo_chunk_3 = low3 / chunk_size[3];
    int64_t hi_chunk_0 = high0 / chunk_size[0];
    int64_t hi_chunk_1 = high1 / chunk_size[1];
    int64_t hi_chunk_2 = high2 / chunk_size[2];
    int64_t hi_chunk_3 = high3 / chunk_size[3];

    for(int64_t i = lo_chunk_0; i <= hi_chunk_0; i++) {
      for(int64_t j = lo_chunk_1; j <= hi_chunk_1; j++) {
        for(int64_t k = lo_chunk_2; k <= hi_chunk_2; k++) {
          for(int64_t l = lo_chunk_3; l <= hi_chunk_3; l++) {
            int64_t chunk_offset = i * chunks_per_dim[1] * chunks_per_dim[2] * chunks_per_dim[3] +
                                   j * chunks_per_dim[2] * chunks_per_dim[3] +
                                   k * chunks_per_dim[3] + l;
            ga_over_upcxx_chunk<T>* chunk = all_chunks[chunk_offset];
            fut                           = upcxx::when_all(
                                        fut, chunk->get_any(low0, low1, low2, low3, high0, high1, high2, high3, out));
          }
        }
      }
    }
    fut.wait();
  }

  void put(int64_t low0, int64_t low1, int64_t low2, int64_t low3, int64_t high0, int64_t high1,
           int64_t high2, int64_t high3, T* in, int64_t* in_dims) {
    upcxx::future<> fut = upcxx::make_future();

    // Check that specified output dimensions match size of fetched
    // region
    if(in_dims[0] != (high0 - low0 + 1) || in_dims[1] != (high1 - low1 + 1) ||
       in_dims[2] != (high2 - low2 + 1) || in_dims[3] != (high3 - low3 + 1)) {
      fprintf(stderr,
              "ga_over_upcxx::put size mismatch (%ld %ld "
              "%ld %ld) (%ld %ld %ld %ld)\n",
              in_dims[0], in_dims[1], in_dims[2], in_dims[3], high0 - low0 + 1, high1 - low1 + 1,
              high2 - low2 + 1, high3 - low3 + 1);
      abort();
    }

    int64_t lo_chunk_0 = low0 / chunk_size[0];
    int64_t lo_chunk_1 = low1 / chunk_size[1];
    int64_t lo_chunk_2 = low2 / chunk_size[2];
    int64_t lo_chunk_3 = low3 / chunk_size[3];
    int64_t hi_chunk_0 = high0 / chunk_size[0];
    int64_t hi_chunk_1 = high1 / chunk_size[1];
    int64_t hi_chunk_2 = high2 / chunk_size[2];
    int64_t hi_chunk_3 = high3 / chunk_size[3];

    for(int64_t i = lo_chunk_0; i <= hi_chunk_0; i++) {
      for(int64_t j = lo_chunk_1; j <= hi_chunk_1; j++) {
        for(int64_t k = lo_chunk_2; k <= hi_chunk_2; k++) {
          for(int64_t l = lo_chunk_3; l <= hi_chunk_3; l++) {
            int64_t chunk_offset = i * chunks_per_dim[1] * chunks_per_dim[2] * chunks_per_dim[3] +
                                   j * chunks_per_dim[2] * chunks_per_dim[3] +
                                   k * chunks_per_dim[3] + l;
            ga_over_upcxx_chunk<T>* chunk = all_chunks[chunk_offset];
            fut                           = upcxx::when_all(fut, chunk->put_any(low0, low1, low2, low3, high0, high1, high2, high3, in));
          }
        }
      }
    }
    fut.wait();
  }

  void minimum(double& out_val, int64_t& out_i0, int64_t& out_i1, int64_t& out_i2,
               int64_t& out_i3) {
    double  min_val = std::numeric_limits<double>::max();
    int64_t i0      = 0;
    int64_t i1      = 0;
    int64_t i2      = 0;
    int64_t i3      = 0;

    for(auto i = local_chunks_begin(), e = local_chunks_end(); i != e; i++) {
      double  chunk_min_val;
      int64_t chunk_i0, chunk_i1, chunk_i2, chunk_i3;
      (*i)->minimum(chunk_min_val, chunk_i0, chunk_i1, chunk_i2, chunk_i3);
      if(chunk_min_val < min_val) {
        min_val = chunk_min_val;
        i0      = (*i)->get_lo_coord(0) + chunk_i0;
        i1      = (*i)->get_lo_coord(1) + chunk_i1;
        i2      = (*i)->get_lo_coord(2) + chunk_i2;
        i3      = (*i)->get_lo_coord(3) + chunk_i3;
      }
    }

    double global_min = upcxx::reduce_all(min_val, upcxx::op_fast_min, tid.here()).wait();
    if(global_min != min_val) { i0 = INT64_MAX; }
    int64_t global_i0 = upcxx::reduce_all(i0, upcxx::op_fast_min, tid.here()).wait();
    if(global_min != min_val || global_i0 != i0) { i1 = INT64_MAX; }
    int64_t global_i1 = upcxx::reduce_all(i1, upcxx::op_fast_min, tid.here()).wait();
    if(global_min != min_val || global_i0 != i0 || global_i1 != i1) { i2 = INT64_MAX; }
    int64_t global_i2 = upcxx::reduce_all(i2, upcxx::op_fast_min, tid.here()).wait();
    if(global_min != min_val || global_i0 != i0 || global_i1 != i1 || global_i2 != i2) {
      i3 = INT64_MAX;
    }
    int64_t global_i3 = upcxx::reduce_all(i3, upcxx::op_fast_min, tid.here()).wait();

    if(global_i0 == INT64_MAX || global_i1 == INT64_MAX || global_i2 == INT64_MAX ||
       global_i3 == INT64_MAX) {
      abort();
    }

    out_val = global_min;
    out_i0  = global_i0;
    out_i1  = global_i1;
    out_i2  = global_i2;
    out_i3  = global_i3;
  }

  void maximum(double& out_val, int64_t& out_i0, int64_t& out_i1, int64_t& out_i2,
               int64_t& out_i3) {
    double  max_val = -1.0 * std::numeric_limits<double>::max();
    int64_t i0      = 0;
    int64_t i1      = 0;
    int64_t i2      = 0;
    int64_t i3      = 0;

    for(auto i = local_chunks_begin(), e = local_chunks_end(); i != e; i++) {
      double  chunk_max_val;
      int64_t chunk_i0, chunk_i1, chunk_i2, chunk_i3;
      (*i)->maximum(chunk_max_val, chunk_i0, chunk_i1, chunk_i2, chunk_i3);
      if(chunk_max_val > max_val) {
        max_val = chunk_max_val;
        i0      = (*i)->get_lo_coord(0) + chunk_i0;
        i1      = (*i)->get_lo_coord(1) + chunk_i1;
        i2      = (*i)->get_lo_coord(2) + chunk_i2;
        i3      = (*i)->get_lo_coord(3) + chunk_i3;
      }
    }

    double  global_max = upcxx::reduce_all(max_val, upcxx::op_fast_max, tid.here()).wait();
    int64_t global_i   = ((global_max == max_val) ? (i0 * dims[1] * dims[2] * dims[3] +
                                                   i1 * dims[2] * dims[3] + i2 * dims[3] + i3)
                                                  : INT64_MAX);
    global_i           = upcxx::reduce_all(global_i, upcxx::op_fast_min, tid.here()).wait();

    int64_t global_i0 = global_i / (dims[1] * dims[2] * dims[3]);
    int64_t global_i1 = (global_i / (dims[2] * dims[3])) % dims[1];
    int64_t global_i2 = (global_i / dims[3]) % dims[2];
    int64_t global_i3 = global_i % dims[3];

    out_val = global_max;
    out_i0  = global_i0;
    out_i1  = global_i1;
    out_i2  = global_i2;
    out_i3  = global_i3;
  }

  chunk_iterator local_chunks_begin() { return local_chunks.begin(); }
  chunk_iterator local_chunks_end() { return local_chunks.end(); }
};

class atomic_counter_over_upcxx {
private:
  int                            rank;
  int                            nranks;
  upcxx::global_ptr<int64_t>     gptr;
  upcxx::atomic_domain<int64_t>* ad_i64;
  upcxx::team_id                 tid;

public:
  atomic_counter_over_upcxx(upcxx::team& team, int64_t init_val = 0) {
    rank   = team.rank_me();
    nranks = team.rank_n();

    ad_i64 = new upcxx::atomic_domain<int64_t>({upcxx::atomic_op::fetch_add}, team);

    if(rank == 0) { gptr = upcxx::new_array<int64_t>(1); }
    else { gptr = upcxx::new_array<int64_t>(1); }

    if(!gptr) {
      std::cerr << "Error doing allocation?" << std::endl;
      abort();
    }
    gptr.local()[0] = init_val;
    gptr            = upcxx::broadcast(gptr, 0, team).wait();

    tid = team.id();

    upcxx::barrier(team);
  }

  void destroy() {
    upcxx::barrier(tid.here());
    if(rank == 0) { upcxx::delete_array(gptr); }
    ad_i64->destroy();
    upcxx::barrier(tid.here());
  }

  int64_t fetch_add(int64_t val) {
    return ad_i64->fetch_add(gptr, val, std::memory_order_relaxed).wait();
  }
};

#endif
