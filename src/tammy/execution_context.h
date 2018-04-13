#ifndef TAMMY_EXECUTION_CONTEXT_H_
#define TAMMY_EXECUTION_CONTEXT_H_

#include "scheduler.h"

namespace tammy {

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
      
  ~ExecutionContext() {
    
  }

  /**
   * Construct a scheduler object
   * @return Scheduler object
   */
  Scheduler scheduler() {
    return Scheduler{};
  }

  void allocate() {
    //no-op
  }

 void deallocate() {
    //no-op
  }

  
// private:

}; // class ExecutionContext

} // namespace tammy

#endif // TAMMY_EXECUTION_CONTEXT_H_
