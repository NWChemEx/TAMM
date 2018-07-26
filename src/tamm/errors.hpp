#ifndef TAMM_ERRORS_HPP_
#define TAMM_ERRORS_HPP_

#include <cassert>
#include <iostream>

namespace tamm {

/**
 * @brief Mark code options that are not yet implemented.
 */
#define NOT_IMPLEMENTED()                                                   \
    do {                                                                    \
        std::cerr << "ERROR: "                                              \
                  << "file:" << __FILE__ << "function:" << __func__         \
                  << " line:" << __LINE__ << ". This is not implemented\n"; \
    } while(0)

/**
 * @brief Mark code options that are not yet allowed.
 */
#define NOT_ALLOWED()                                                       \
    do {                                                                    \
        std::cerr << "ERROR: "                                              \
                  << "file:" << __FILE__ << "function:" << __func__         \
                  << " line:" << __LINE__ << ". This is not implemented\n"; \
    } while(0)

/**
 * @brief Mark code paths that should be unreachable
 */
#define UNREACHABLE()                                               \
    do {                                                            \
        std::cerr << "ERROR: "                                      \
                  << "file:" << __FILE__ << "function:" << __func__ \
                  << " line:" << __LINE__                           \
                  << ". This line should be unreachable\n";         \
    } while(0)

/**
 * @brief Wrapper for assertion checking.
 *
 * This is meant to identify preconditions and possibly include additional
 * operations (e.g., an error message).
 */
//#define EXPECTS(cond) assert(cond)
#define EXPECTS(cond)                                                          \
    do {                                                                       \
        if(!(cond)) {                                                          \
            std::cerr<<"EXPECTS failed. Condition: "<<#cond<<"\n";               \
            throw std::string{"EXPECT condition failed: "}+std::string{#cond}; \
        }                                                                      \
    } while(0)

} // namespace tamm

#endif // TAMM_ERRORS_HPP_
