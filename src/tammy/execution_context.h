#ifndef TAMMY_EXECUTION_CONTEXT_H_
#define TAMMY_EXECUTION_CONTEXT_H_


namespace tammy {

class Scheduler;
/**
 * @brief Wrapper class to hold information during execution.
 *
 * This class holds the choice of default memory manager, distribution, irrep, etc.
 *
 * @todo Should spin_restricted be wrapper by this class? Or should it always default to false?
 */
class ExecutionContext {
 public:
  ExecutionContext() = default;
  ExecutionContext(const ExecutionContext&) = default;
  ExecutionContext(ExecutionContext&&) = default;
  ExecutionContext& operator=(const ExecutionContext&) = default;
  ExecutionContext& operator=(ExecutionContext&&) = default;
      
  ~ExecutionContext() {}

  /**
   * Construct a scheduler object
   * @return Scheduler object
   */
  Scheduler& scheduler() {
    // return Scheduler{};
  }

  void allocate() {
    //no-op
  }

 void deallocate() {
    //no-op
  }

  
  // private:
  //   Scheduler _scheduler;

}; // class ExecutionContext

} // namespace tammy

#endif // TAMMY_EXECUTION_CONTEXT_H_
