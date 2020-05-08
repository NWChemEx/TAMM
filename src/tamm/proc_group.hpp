#ifndef TAMM_PROC_GROUP_H_
#define TAMM_PROC_GROUP_H_

#include <ga.h>
#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include <map>
#include <vector>
#include "ga-mpi.h"

#include "tamm/types.hpp"

namespace tamm {

class ProcGroup {
 public:
  /**
   * @brief Construct a new Proc Group object
   * 
   */
  ProcGroup() : pginfo_{std::make_shared<ProcGroupInfo>()} {}
  ProcGroup(const ProcGroup&) = default;
  ProcGroup(ProcGroup&& pg) : pginfo_{std::move(pg.pginfo_)} {}
  ProcGroup& operator=(ProcGroup pg) {
    using std::swap;
    swap(*this, pg);
    return *this;
  }
  ~ProcGroup() = default;

  /**
   * @brief Construct a new Proc Group object by wrapping the given MPI
   * communicator and GA process group
   *
   * @param mpi_comm MPI communicator
   * @param ga_pg Corresponding GA process group
   *
   * @pre @param ga_pg corresponds to @param mpi_comm
   */
  ProcGroup(MPI_Comm mpi_comm, int ga_pg)
      : pginfo_{std::make_shared<ProcGroupInfo>()} {
    pginfo_->mpi_comm_ = mpi_comm;
    pginfo_->created_mpi_comm_ = false;
    pginfo_->ga_pg_ = ga_pg;
    pginfo_->created_ga_pg_ = false;
    pginfo_->is_valid_ = (mpi_comm != MPI_COMM_NULL);
  }

  /**
   * @brief Collectively create a ProcGroup from the given communicator
   *
   * @param mpi_comm Communication to be used as a basis of the ProcGroup
   * @return ProcGroup New ProcGroup object that duplicates @param mpi_comm and
   * creates the corresponding GA process group
   */
  static ProcGroup create_coll(MPI_Comm mpi_comm) {
    MPI_Comm comm_out;
    MPI_Comm_dup(mpi_comm, &comm_out);
    ProcGroup pg;
    pg.pginfo_->mpi_comm_ = comm_out;
    pg.pginfo_->created_mpi_comm_ = true;
    pg.pginfo_->ga_pg_ = create_ga_process_group_coll(mpi_comm);
    pg.pginfo_->created_ga_pg_ = true;
    pg.pginfo_->is_valid_ = (mpi_comm != MPI_COMM_NULL);
    return pg;
  }

  /**
   * @brief Check if the given process group is valid
   * 
   * @return true if it is a non-null process group (not MPI_COMM_NULL)
   * @return false otherwise
   */
  bool is_valid() const { return pginfo_->is_valid_; }

  /**
   * Number of ranks in the wrapped communicator
   * @return Size of the wrapped communicator
   *
   * @pre is_valid()
   */
  Proc size() const {
    int nranks;
    EXPECTS(is_valid());
    MPI_Comm_size(pginfo_->mpi_comm_, &nranks);
    return Proc{nranks};
  }

  /**
   * Access the underlying MPI communicator
   * @return the wrapped MPI communicator
   *
   * @pre is_valid()
   */
  MPI_Comm comm() const {
    EXPECTS(is_valid());
    return pginfo_->mpi_comm_;
  }

  /**
   * @brief Obtained the underlying GA process group
   * 
   * @return int Underlying GA process group
   * 
   * @pre is_valid()
   */
  int ga_pg() const {
    EXPECTS(is_valid());
    return pginfo_->ga_pg_;
  }

  /**
   * Rank of invoking process
   * @return rank of invoking process in the wrapped communicator
   */
  Proc rank() const {
    int rank;
    EXPECTS(is_valid());
    MPI_Comm_rank(pginfo_->mpi_comm_, &rank);
    return Proc{rank};
  }

  /**
   * Collectivelu clone the given process group
   * @return A copy of this process group
   */
  ProcGroup clone_coll() const { return create_coll(pginfo_->mpi_comm_); }

  /**
   * @brief Collectively destroy this communicator.
   *
   * @post !is_valid()
   * @post comm() == MPI_COMM_NULL
   * @post underlying GA process group in invalid
   *
   */
  void destroy_coll() {
    EXPECTS(is_valid());
    if (pginfo_->created_mpi_comm_) {
      MPI_Comm_free(&pginfo_->mpi_comm_);
      pginfo_->created_mpi_comm_ = false;
    }
    if (pginfo_->created_ga_pg_) {
      GA_Pgroup_destroy(pginfo_->ga_pg_);
      pginfo_->created_ga_pg_ = false;
    }
    *pginfo_ = ProcGroupInfo{};  // cleanup
  }

  /**
   * Barrier on the wrapped communicator.
   *
   * @pre is_valid()
   */
  void barrier() {
    EXPECTS(is_valid());
    GA_Pgroup_sync(pginfo_->ga_pg_);
  }

  /**
   * @brief Translate a rank from this process group to anothe process group
   * 
   * @param proc rank in this process group
   * @param pg2 Process group to be translated into
   * @return Proc rank of @param proc in ProcGroup @param pg2
   * 
   * @pre is_valid()
   * @pre proc >=0 && proc < size()
   * @pre pg2.is_valid
   * @pre proc (w.r.t. this) exists in pg2
   */
  Proc rank_translate(Proc proc, const ProcGroup& pg2) {
    EXPECTS(is_valid());
    EXPECTS(pg2.is_valid());
    MPI_Group group1, group2;
    int ranks1{static_cast<int>(proc.value())};
    int ranks2{MPI_PROC_NULL};
    // MPI_Comm_group(comm_, &group1);
    MPI_Comm_group(pginfo_->mpi_comm_, &group1);
    MPI_Comm_group(pg2.pginfo_->mpi_comm_, &group2);
    MPI_Group_translate_ranks(group1, 1, &ranks1, group2, &ranks2);
    assert(ranks2 != MPI_PROC_NULL);
    return Proc{ranks2};
  }

  /**
   * @brief Create a GA process group correspondig to MPI_COMM_SELF, or just
   * access it.
   *
   * @param create Create the GA pgroup if true
   * @return int Return the created GA pgroup
   */
  static int self_ga_pgroup(bool create = false) {
    static int ga_pg_self = -1;
    static bool created = false;
    if (!create) {
      EXPECTS(created);
      return ga_pg_self;
    }
    ga_pg_self = create_ga_process_group_coll(MPI_COMM_SELF);
    created = true;
    return ga_pg_self;
  }

 private:
  /**
   * Create a GA process group corresponding to the given proc group
   * @param pg TAMM process group
   * @return GA processes group on this TAMM process group
   */
  static int create_ga_process_group_coll(MPI_Comm comm) {
    int nranks;
    MPI_Comm_size(comm, &nranks);
    MPI_Group group, group_world;
    int ranks[nranks], ranks_world[nranks];
    MPI_Comm_group(comm, &group);

    MPI_Comm_group(GA_MPI_Comm(), &group_world);

    for (int i = 0; i < nranks; i++) {
      ranks[i] = i;
    }
    MPI_Group_translate_ranks(group, nranks, ranks, group_world, ranks_world);

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(GA_Pgroup_get_world());
    int ga_pg = GA_Pgroup_create(ranks_world, nranks);
    GA_Pgroup_set_default(ga_pg_default);
    return ga_pg;
  }

  /**
   * @brief Swap contents of two given objects
   * 
   * @param first Object to be swapped
   * @param second Object to be swapped
   */
  friend void swap(ProcGroup& first, ProcGroup& second) {
    using std::swap;
    swap(first.pginfo_, second.pginfo_);
  }

  /**
   * @brief Object holding ProcGroup's shared state. It trackes whether the MPI
   * communicator and GA pgroup were created/dup-ed and this have to be
   * destructed.
   *
   */
  struct ProcGroupInfo {
    MPI_Comm mpi_comm_ = MPI_COMM_NULL; /**< Wrapped MPI communicator */
    bool created_mpi_comm_ = false;     /**< Was mpi_comm_ duplicated/created */
    int ga_pg_ = -1;                    /**< Corresponding GA communicator */
    bool created_ga_pg_ = false;        /**< Was this GA pgroup created */
    bool is_valid_ =
        false; /**< Is this object valid, i.e., mpi_comm_ != MPI_COMM_NULL */
  };
  std::shared_ptr<ProcGroupInfo> pginfo_; /**< Shared ProcGroupInfo state */

  /**
   * @brief Equality operator
   * 
   * @param lhs LHS to be compared
   * @param rhs RHS to be compared
   * @return true if lhs == rhs
   * @return false otherwise
   */
  friend bool operator==(const ProcGroup& lhs, const ProcGroup& rhs) {
    int result;
    MPI_Comm_compare(lhs.pginfo_->mpi_comm_, rhs.pginfo_->mpi_comm_, &result);
    return result == MPI_IDENT;
  }

  /**
   * @brief Inequality operator
   * 
   * @param lhs LHS to be compared
   * @param rhs RHS to be compared
   * @return true if lhs != rhs
   * @return false ptherwise
   */
  friend bool operator!=(const ProcGroup& lhs, const ProcGroup& rhs) {
    return !(lhs == rhs);
  }
};  // class ProcGroup

#if 0
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
    pg.ga_pg_ = create_ga_process_group_coll(mpi_comm);
    return pg;
  }

  ProcGroup(const ProcGroup&) = default;
  ProcGroup(ProcGroup&& pg)  // TBD: check if this can be default
      : mpi_comm_{std::move(pg.mpi_comm_)},
      ga_pg_{pg.ga_pg_} {}

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

  int ga_pg() const {
    return ga_pg_;
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
  GA_Pgroup_destroy(ga_pg_);
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
    //MPI_Barrier(*mpi_comm_);
    GA_Pgroup_sync(ga_pg_);
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
  /**
   * Create a GA process group corresponding to the given proc group
   * @param pg TAMM process group
   * @return GA processes group on this TAMM process group
   */
  static int create_ga_process_group_coll(MPI_Comm comm) {
    int nranks;
    MPI_Comm_size(comm, &nranks);
    MPI_Group group, group_world;
    int ranks[nranks], ranks_world[nranks];
    MPI_Comm_group(comm, &group);

    MPI_Comm_group(GA_MPI_Comm(), &group_world);

    for (int i = 0; i < nranks; i++) {
      ranks[i] = i;
    }
    MPI_Group_translate_ranks(group, nranks, ranks, group_world, ranks_world);

    int ga_pg_default = GA_Pgroup_get_default();
    GA_Pgroup_set_default(GA_Pgroup_get_world());
    int ga_pg = GA_Pgroup_create(ranks_world, nranks);
    GA_Pgroup_set_default(ga_pg_default);
    return ga_pg;
  }

  // MPI_Comm comm_;// = MPI_COMM_NULL;
  std::shared_ptr<MPI_Comm> mpi_comm_;
  int ga_pg_;
  bool is_valid_;

  static void deleter(MPI_Comm* mpi_comm) {
    EXPECTS(*mpi_comm != MPI_COMM_NULL);
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
#endif

}  // namespace tamm

#endif  // TAMM_PROC_GROUP_H_
