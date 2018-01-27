#ifndef TAMMY_ERRORS_H_
#define TAMMY_ERRORS_H_

#include <cassert>

namespace tammy {

/**
 * @brief Mark code options that are not yet implemented.
 */
#define NOT_IMPLEMENTED() do {                        \
  std::cout<<"ERROR: "                                \
           <<"file:"<<__FILE__                        \
           <<"function:"<<__func__                    \
           <<" line:"<<__LINE__                       \
           <<". This is not implemented\n";           \
  } while(0)

/**
 * @brief Mark code paths that should be unreachable
 */
#define UNREACHABLE() do {                            \
    std::cout<<"ERROR: "                              \
             <<"file:"<<__FILE__                      \
             <<"function:"<<__func__                  \
             <<" line:"<<__LINE__                     \
             <<". This line should be unreachable\n"; \
  } while(0)

/**
 * @brief Wrapper for assertion checking.
 *
 * This is meant to identify preconditions and possibly include additional operations (e.g., an error message).
 */
#define EXPECTS(cond) assert(cond)

} //namespace tammy

#endif // TAMMY_ERRORS_H_

