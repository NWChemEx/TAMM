#ifndef TAMM_PROC_GROUP_H_
#define TAMM_PROC_GROUP_H_

#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include <map>
#include <vector>

#include "tamm/types.hpp"

namespace tamm {

/**
 * @brief Wrapper to MPI communicator and related operations.
 */
class ProcGroup {
 public:
  //ProcGroup() = default;
  ProcGroup() : mpi_comm_(std::make_shared<MPI_Comm>(MPI_COMM_NULL)), is_valid_(false){
    // MPI_Comm* comm_out = new MPI_Comm();
    // *comm_out = MPI_COMM_NULL;
    // mpi_comm_.reset(comm_out);
  }
  ProcGroup(MPI_Comm comm) {
    //std::shared_ptr<MPI_Comm> comm_out = std::make_shared<MPI_Comm> (comm);
    // mpi_comm_.reset(comm_out.get());
    //MPI_Comm* comm_out = new MPI_Comm();
    //*comm_out = comm;
    //mpi_comm_.reset(comm_out);
    mpi_comm_.reset(new MPI_Comm(comm));
    is_valid_=(comm != MPI_COMM_NULL);
  }
  static ProcGroup create_coll(MPI_Comm mpi_comm) {
    MPI_Comm* comm_out = new MPI_Comm();
    MPI_Comm_dup(mpi_comm, comm_out);
    ProcGroup pg;
    pg.mpi_comm_.reset(comm_out, deleter);
    pg.is_valid_ = true;
    return pg;
  }

  ProcGroup(const ProcGroup&) = default;
  ProcGroup(ProcGroup&& pg)  // TBD: check if this can be default
      : mpi_comm_{std::move(pg.mpi_comm_)} {}

  ProcGroup& operator=(const ProcGroup&) = default;

  //ProcGroup(ProcGroup&&) = default;
  ProcGroup& operator=(ProcGroup&&) = default;
  ~ProcGroup() = default;

  //explicit ProcGroup(MPI_Comm comm = MPI_COMM_NULL)
  //    : comm_{comm},
  //      is_valid_{comm != MPI_COMM_NULL} { }

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
    //MPI_Comm_rank(comm_, &rank);
    MPI_Comm_rank(*mpi_comm_, &rank);
    return Proc{rank};
  }

  /**
   * Number of ranks in the wrapped communicator
   * @return Size of the wrapped communicator
   */
  Proc size() const {
    int nranks;
    EXPECTS(is_valid());
    //MPI_Comm_size(comm_, &nranks);
    MPI_Comm_size(*mpi_comm_, &nranks);
    return Proc{nranks};
  }

  /**
   * Access the underlying MPI communicator
   * @return the wrapped MPI communicator
   */
  MPI_Comm comm() const {
    //return comm_;
    return *mpi_comm_;
  }

  /**
   * Duplicate/clone the wrapped MPI communicator
   * @return A copy.
   * @note This is a collective call on the wrapped communicator
   * @todo Rename this call to clone_coll() to indicate this is a collective call.
   */
  ProcGroup clone_coll() const { return create_coll(*mpi_comm_); }
  
  //ProcGroup clone() const {
  //  EXPECTS(is_valid());
  //  MPI_Comm comm_out{MPI_COMM_NULL};
  //  MPI_Comm_dup(comm_, &comm_out);
  //  return ProcGroup{comm_out};
  //}

void destroy_coll() { 
  MPI_Comm_free(mpi_comm_.get());
  is_valid_ = false;
  }
  /**
   * Free the wrapped communicator
   */
 /* void destroy() {
    if(is_valid()) {
      MPI_Comm_free(&comm_);
    }
    comm_ = MPI_COMM_NULL;
    is_valid_ = false;
  }*/

  /**
   * Barrier on the wrapped communicator.
   */
  void barrier() {
    //MPI_Barrier(comm_);
    MPI_Barrier(*mpi_comm_);
  }
  
  Proc rank_translate(Proc proc, const ProcGroup& pg2) {
    EXPECTS(is_valid());
    MPI_Group group1, group2;
    int ranks1{static_cast<int>(proc.value())};
    int ranks2{MPI_PROC_NULL};
    //MPI_Comm_group(comm_, &group1);
    MPI_Comm_group(*mpi_comm_, &group1);
    MPI_Comm_group(*pg2.mpi_comm_, &group2);
    MPI_Group_translate_ranks(group1, 1, &ranks1, group2, &ranks2);
    assert(ranks2 != MPI_PROC_NULL);
    return Proc{ranks2};
  }
  

 private:
  //MPI_Comm comm_;// = MPI_COMM_NULL;
  std::shared_ptr<MPI_Comm> mpi_comm_;
  bool is_valid_;

  static void deleter(MPI_Comm* mpi_comm) {
    assert(*mpi_comm == MPI_COMM_NULL);
    delete mpi_comm;
  }

  static void deleter_comm(MPI_Comm* mpi_comm) {
    delete mpi_comm;
  }

  friend bool operator == (const ProcGroup& lhs, const ProcGroup& rhs) {
    int result;
    MPI_Comm_compare(*lhs.mpi_comm_, *rhs.mpi_comm_, &result);
    return result == MPI_IDENT;
  }
  
  friend  bool operator != (const ProcGroup& lhs, const ProcGroup& rhs) {
    return !(lhs == rhs);
  }
};  // class ProcGroup


} // namespace tamm


#endif // TAMM_PROC_GROUP_H_

