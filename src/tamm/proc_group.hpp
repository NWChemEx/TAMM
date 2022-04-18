#pragma once

#include <pthread.h>
#include <cassert>
#include <map>
#include <vector>
#include <memory>

#include "tamm/types.hpp"

#include <upcxx/upcxx.hpp>

//extern std::mutex master_mtx;

namespace tamm {

#if 1
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
  ProcGroup(upcxx::team* team)
      : pginfo_{std::make_shared<ProcGroupInfo>()} {
    pginfo_->team_ = team;
    pginfo_->created_team_ = false;
    pginfo_->is_valid_ = true;

    // pginfo_->recvd_ops = new upcxx::dist_object<int64_t>(0, *team);
    {
    //upcxx::persona_scope master_scope(master_mtx,
    //        upcxx::master_persona());
    pginfo_->recvd_ops = new upcxx::dist_object<int64_t>(0, *team);
    }

    pginfo_->pending_put_futures = upcxx::make_future();
  }

  static ProcGroup create_world_coll() {
#ifdef UPCXX_VERSION
      return create_coll(upcxx::world());
#else
      return create_coll(GA_MPI_Comm());
#endif
  }

  /**
   * @brief Collectively create a ProcGroup from the given communicator
   *
   * @param mpi_comm Communication to be used as a basis of the ProcGroup
   * @return ProcGroup New ProcGroup object that duplicates @param mpi_comm and
   * creates the corresponding GA process group
   */
  static ProcGroup create_coll(upcxx::team& team) {
    return ProcGroup(&team);
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
    EXPECTS(is_valid());
    return pginfo_->team_->rank_n();
  }

  /**
   * Access the underlying MPI communicator
   * @return the wrapped MPI communicator
   *
   * @pre is_valid()
   */
  upcxx::team* team() const {
    EXPECTS(is_valid());
    return pginfo_->team_;
  }

  /**
   * Rank of invoking process
   * @return rank of invoking process in the wrapped communicator
   */
  Proc rank() const {
    EXPECTS(is_valid());
    return pginfo_->team_->rank_me();
  }

  /**
   * Collectivelu clone the given process group
   * @return A copy of this process group
   */
  ProcGroup clone_coll() const { return create_coll(*pginfo_->team_); }

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
    if (pginfo_->created_team_) {
        {
            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            pginfo_->team_->destroy();
        }
      pginfo_->created_team_ = false;
    }
    *pginfo_ = ProcGroupInfo{};  // cleanup
  }

  /**
   * Barrier on the wrapped communicator.
   *
   * @pre is_valid()
   */

  void add_op(int rank) {
      pginfo_->sent_ops += 1;
  }

  void add_put_fut(upcxx::future<> &f) {
      pginfo_->pending_put_futures = upcxx::when_all(pginfo_->pending_put_futures, f);
  }

  upcxx::dist_object<int64_t>* get_recvd_ops_object() { return pginfo_->recvd_ops; }

  void barrier() {
    EXPECTS(is_valid());
    pginfo_->pending_put_futures.wait();
    pginfo_->pending_put_futures = upcxx::make_future();

    int64_t total_sends;
    {
        //upcxx::persona_scope master_scope(master_mtx,
        //        upcxx::master_persona());
        total_sends = upcxx::reduce_all<int64_t>(pginfo_->sent_ops,
                upcxx::op_fast_add, *(pginfo_->team_)).wait();
    }

    // total_sends = upcxx::master_persona().lpc([&] {
    //         return upcxx::reduce_all<int64_t>(pginfo_->sent_ops,
    //             upcxx::op_fast_add, *(pginfo_->team_));
    //         }).wait();

    pginfo_->sent_ops = 0;

    int64_t total_recvs;
    do {
        int64_t local_recvd = *(*(pginfo_->recvd_ops));

        {
            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            total_recvs = upcxx::reduce_all<int64_t>(local_recvd,
                    upcxx::op_fast_add, *(pginfo_->team_)).wait();
        }
        // total_recvs = upcxx::master_persona().lpc([&] {
        //             return upcxx::reduce_all<int64_t>(local_recvd,
        //                 upcxx::op_fast_add, *(pginfo_->team_));
        //         }).wait();
    } while (total_recvs < total_sends);

    if (total_recvs != total_sends) {
        abort();
    }

    *(*(pginfo_->recvd_ops)) = 0;
    //upcxx::persona_scope master_scope(master_mtx,
    //        upcxx::master_persona());
    upcxx::barrier(*(pginfo_->team_));
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

    int this_rank = proc.value();
    int world_rank = (*pginfo_->team_)[this_rank];
    
    int other_rank = pg2.pginfo_->team_->from_world(world_rank);
    return Proc{other_rank};
  }

  /**
   * @brief Translate all ranks from this proc group to ranks in @param pg2 
   *
   * @param pg2 Proc group to which this group's ranks need to be translated
   * @return std::vector<Proc> Translated rank for each rank in this group. -1
   * indicates a rank in this group that is not in @param pg2
   */
  std::vector<Proc> rank_translate(const ProcGroup& pg2) {
    EXPECTS(is_valid());
    EXPECTS(pg2.is_valid());
    const size_t nranks = size().value();
    std::vector<Proc> ret(nranks);
    for(int i=0; i<nranks; i++) {
      ret[i] = rank_translate(i, pg2);
    }
    return ret;
  }

  template <typename T>
  void broadcast(T *buf, int root) {
    upcxx::broadcast(buf, 1, root, *pginfo_->team_).wait();
  }

  template <typename T>
  void broadcast(T *buf, int buflen, int root) {
    upcxx::broadcast(buf, buflen, root, *pginfo_->team_).wait();
  }

  template <typename T>
  void gather(const T *sbuf, upcxx::global_ptr<T> rbuf) {
    gather(sbuf, 1, rbuf, 1);
  }

  template <typename T>
  void gather(const T *sbuf, int scount, upcxx::global_ptr<T> rbuf, int rcount) {
    EXPECTS(scount == rcount);
    EXPECTS(sbuf != nullptr || scount == 0);
    if (sbuf) {
        upcxx::rput(sbuf, rbuf + (scount * rank().value()), scount).wait();
    }
    upcxx::barrier(*pginfo_->team_);
  }

  template <typename T>
  void gatherv(const T *sbuf, int scount, upcxx::global_ptr<T> rbuf,
          const int* rcounts, upcxx::global_ptr<int> displacements) {
    upcxx::barrier(*pginfo_->team_); // ensure that displacements is populated on root
    int my_displacement = upcxx::rget(displacements + rank().value()).wait();
    EXPECTS(sbuf != nullptr || scount == 0);
    if (sbuf) {
        upcxx::rput(sbuf, rbuf + my_displacement, scount).wait();
    }
    upcxx::barrier(*pginfo_->team_); // wait for everyone to rput

    // MPI_Gatherv(sbuf, scount, mpi_type<T>(), rbuf, rcounts, displacements, mpi_type<T>(), root, pginfo_->mpi_comm_);
  }  

  template <typename T>
  void allgather(const T *sbuf, T *rbuf) {
     throw std::runtime_error("allgather unsupported");
    // MPI_Allgather(sbuf, 1, mpi_type<T>(), rbuf, 1, mpi_type<T>(), pginfo_->mpi_comm_);
  }

  template <typename T>
  void allgather(const T *sbuf, int scount, T *rbuf, int rcount) {
     throw std::runtime_error("allgather unsupported");
    // MPI_Allgather(sbuf, scount, mpi_type<T>(), rbuf, rcount, mpi_type<T>(), pginfo_->mpi_comm_);
  }

  template <typename T, typename opT>
  T reduce(const T *buf, opT op, int root) {
    T result{};
    upcxx::reduce_one(buf, &result, 1, op, root, *pginfo_->team_).wait();
    return result;
  }

  template <typename T, typename opT>
  void reduce(const T *sbuf, T *rbuf, int count, opT op, int root) {
      upcxx::reduce_one(sbuf, rbuf, count, op, root, *pginfo_->team_).wait();
  }

  template <typename T, typename opT>
  T allreduce(const T *buf, opT op) {
    T result{};
    upcxx::reduce_all(buf, &result, 1, op, *pginfo_->team_).wait();
    return result;
  }

  template <typename T, typename opT>
  void allreduce(const T *sbuf, T *rbuf, int count, opT op) {
    upcxx::reduce_all(sbuf, rbuf, count, op, *pginfo_->team_).wait();
  }

 private:
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
    upcxx::team* team_ = NULL; /**< Wrapped MPI communicator */
    bool created_team_ = false;     /**< Was mpi_comm_ duplicated/created */
    bool is_valid_ =
        false; /**< Is this object valid, i.e., mpi_comm_ != MPI_COMM_NULL */
    int64_t sent_ops = 0;
    upcxx::dist_object<int64_t> *recvd_ops = NULL;
    upcxx::future<> pending_put_futures;
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
    return lhs.pginfo_->team_->id() == rhs.pginfo_->team_->id();
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

#else
/**
 * @brief Wrapper to MPI communicator and related operations.
 */
class ProcGroup {
 public:
  ProcGroup() : team_(nullptr), is_valid_(false), pending_put_futures(upcxx::make_future()) { }

  ProcGroup(upcxx::team* team) : team_(team), is_valid_(true), pending_put_futures(upcxx::make_future()) {
    recvd_ops = new upcxx::dist_object<int64_t>(0, *team);
    sent_ops = (int64_t*)malloc(sizeof(*sent_ops));
    assert(sent_ops);
    *sent_ops = 0;
  }

  static ProcGroup create_coll(upcxx::team& team) {
    return ProcGroup(&team);
  }

  ProcGroup(const ProcGroup&) = default;
  ProcGroup(ProcGroup&& pg)  // TBD: check if this can be default
      : team_{pg.team_}, is_valid_(pg.is_valid_), sent_ops(pg.sent_ops),
      recvd_ops(pg.recvd_ops), pending_put_futures(pg.pending_put_futures) {
  }

  ProcGroup& operator=(const ProcGroup&) = default;

  ProcGroup& operator=(ProcGroup&&) = default;
  ~ProcGroup() = default;

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
    // return Proc{t.rank_me()};
    return team_->rank_me();
  }

  /**
   * Number of ranks in the wrapped communicator
   * @return Size of the wrapped communicator
   */
  Proc size() const {
    // return Proc{t.rank_n()};
    return team_->rank_n();
  }

  /**
   * Access the underlying MPI communicator
   * @return the wrapped MPI communicator
   */
  upcxx::team* team() const {
    return team_;
  }

  /**
   * Duplicate/clone the wrapped MPI communicator
   * @return A copy.
   * @note This is a collective call on the wrapped communicator
   * @todo Rename this call to clone_coll() to indicate this is a collective call.
   */
  ProcGroup clone_coll() const {
    return create_coll(*team_);
  }
  
  void destroy_coll() {
        {
            //upcxx::persona_scope master_scope(master_mtx,
            //        upcxx::master_persona());
            upcxx::team& t = *team_;
            t.destroy();
        }
    is_valid_ = false;
  }

  void add_op(int rank) {
      *sent_ops += 1;
  }

  void add_put_fut(upcxx::future<> &f) {
      pending_put_futures = upcxx::when_all(pending_put_futures, f);
  }

  upcxx::dist_object<int64_t>* get_recvd_ops_object() { return recvd_ops; }

  /**
   * Barrier on the wrapped communicator.
   */
  void barrier() {
    pending_put_futures.wait();
    pending_put_futures = upcxx::make_future();
    int64_t total_sends = upcxx::reduce_all<int64_t>(*sent_ops, upcxx::op_fast_add, *team_).wait();

    *sent_ops = 0;

    int64_t total_recvs;
    do {
        int64_t local_recvd = *(*recvd_ops);
        total_recvs = upcxx::reduce_all<int64_t>(local_recvd, upcxx::op_fast_add, *team_).wait();
    } while (total_recvs < total_sends);

    if (total_recvs != total_sends) {
        abort();
    }

    *(*recvd_ops) = 0;
    upcxx::barrier(*team_);
  }

  void broadcast(void *buf, int buflen, int root) {
      upcxx::broadcast((uint8_t *)buf, buflen, root, *team_).wait();
  }

  /*
   * Given a rank 'proc' in this ProcGroup, return its rank in another ProcGroup
   * pg2.
   */
  Proc rank_translate(Proc proc, const ProcGroup& pg2) {
    EXPECTS(is_valid());

    int this_rank = proc.value();
    int world_rank = (*team_)[this_rank];
    
    int other_rank = pg2.team_->from_world(world_rank);
    return Proc{other_rank};
  }
  
 private:
  upcxx::future<> pending_put_futures;
  upcxx::team* team_;
  bool is_valid_;

  // Incremented every time an RPC runs at this rank
  upcxx::dist_object<int64_t> *recvd_ops;
  int64_t *sent_ops;

  friend bool operator == (const ProcGroup& lhs, const ProcGroup& rhs) {
    return lhs.team_->id() == rhs.team_->id();
  }
  
  friend  bool operator != (const ProcGroup& lhs, const ProcGroup& rhs) {
    return !(lhs == rhs);
  }
};  // class ProcGroup
#endif

}  // namespace tamm
