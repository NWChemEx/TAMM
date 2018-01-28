#ifndef TAMMY_PROC_GROUP_H_
#define TAMMY_PROC_GROUP_H_

#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include <map>
#include <vector>

#include "types.h"

namespace tammy {

/**
 * @brief Wrapper to MPI communicator and related operations.
 */
class ProcGroup {
 public:
  //ProcGroup() = default;
  ProcGroup(const ProcGroup&) = default;
  ProcGroup& operator = (const ProcGroup&) = default;
  ~ProcGroup() = default;

  explicit ProcGroup(MPI_Comm comm = MPI_COMM_NULL)
      : comm_{comm},
        is_valid_{comm != MPI_COMM_NULL} { }

  /**
   * Is it a valid communicator (i.e., not MPI_COMM_NULL)
   * @return true is wrapped MPI communicator is not MPI_COMM_NULL
   */
  bool is_valid() const {
    return is_valid_;
  }

  /**
   * Rank of invoking process
   * @return rank of invoking process in the wrapped communicator
   */
  Proc rank() const {
    int rank;
    EXPECTS(is_valid());
    MPI_Comm_rank(comm_, &rank);
    return Proc{rank};
  }

  /**
   * Number of ranks in the wrapped communicator
   * @return Size of the wrapped communicator
   */
  Proc size() const {
    int nranks;
    EXPECTS(is_valid());
    MPI_Comm_size(comm_, &nranks);
    return Proc{nranks};
  }

  /**
   * Access the underlying MPI communicator
   * @return the wrapped MPI communicator
   */
  MPI_Comm comm() const {
    return comm_;
  }

  /**
   * Duplicate/clone the wrapped MPI communicator
   * @return A copy.
   * @note This is a collective call on the wrapped communicator
   * @todo Rename this call to clone_coll() to indicate this is a collective call.
   */
  ProcGroup clone() const {
    EXPECTS(is_valid());
    MPI_Comm comm_out{MPI_COMM_NULL};
    MPI_Comm_dup(comm_, &comm_out);
    return ProcGroup{comm_out};
  }

  /**
   * Free the wrapped communicator
   */
  void destroy() {
    if(is_valid()) {
      MPI_Comm_free(&comm_);
    }
    comm_ = MPI_COMM_NULL;
    is_valid_ = false;
  }

  /**
   * Barrier on the wrapped communicator.
   */
  void barrier() {
    MPI_Barrier(comm_);
  }
  
  Proc rank_translate(Proc proc, const ProcGroup& pg2) {
    EXPECTS(is_valid());
    MPI_Group group1, group2;
    int ranks1{static_cast<int>(proc.value())};
    int ranks2{MPI_PROC_NULL};
    MPI_Comm_group(comm_, &group1);
    MPI_Comm_group(pg2.comm_, &group2);
    MPI_Group_translate_ranks(group1, 1, &ranks1, group2, &ranks2);
    assert(ranks2 != MPI_PROC_NULL);
    return Proc{ranks2};
  }
  

 private:
  MPI_Comm comm_;// = MPI_COMM_NULL;
  bool is_valid_;

  friend bool operator == (const ProcGroup& lhs, const ProcGroup& rhs) {
    int result;
    MPI_Comm_compare(lhs.comm_, rhs.comm_, &result);
    return result == MPI_IDENT;
  }
  
  friend  bool operator != (const ProcGroup& lhs, const ProcGroup& rhs) {
    return !(lhs == rhs);
  }  
};  // class ProcGroup



} // namespace tammy


#endif // TAMMY_PROC_GROUP_H_

