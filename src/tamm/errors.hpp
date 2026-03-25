#pragma once

// C++20: std::source_location replaces __FILE__/__LINE__ macro patchwork.
// Macro wrappers are retained for ABI / call-site compatibility.

#include <cassert>
#include <iostream>
#include <source_location>  // C++20
#include <stdexcept>
#include <string>
#include <string_view>

namespace tamm {

// ---------------------------------------------------------------------------
// Core diagnostic helpers (inline, no macros)
// ---------------------------------------------------------------------------

/// Throw a descriptive runtime_error with full source location.
[[noreturn]] inline void
tamm_error(std::string_view msg,
           std::source_location loc = std::source_location::current()) {
  std::string full = std::string{loc.file_name()} + ':' +
                     std::to_string(loc.line()) +
                     " in '" + loc.function_name() + "': " +
                     std::string{msg};
  std::cerr << "TAMM ERROR: " << full << '\n';
  throw std::runtime_error(full);
}

/// Precondition check: throws if cond is false.
inline void
tamm_expects(bool cond, std::string_view msg,
             std::source_location loc = std::source_location::current()) {
  if (!cond) [[unlikely]] {
    std::string full = std::string{loc.file_name()} + ':' +
                       std::to_string(loc.line()) +
                       " in '" + loc.function_name() + "': EXPECTS failed [" +
                       std::string{msg} + ']';
    std::cerr << full << '\n';
    throw std::runtime_error(full);
  }
}

/// Precondition check with custom message.
inline void
tamm_expects_str(bool cond, std::string_view cond_str, std::string_view user_msg,
                 std::source_location loc = std::source_location::current()) {
  if (!cond) [[unlikely]] {
    std::string full = std::string{loc.file_name()} + ':' +
                       std::to_string(loc.line()) +
                       " in '" + loc.function_name() + "': " +
                       std::string{cond_str} + " -- " + std::string{user_msg};
    std::cerr << full << '\n';
    throw std::runtime_error(full);
  }
}

// ---------------------------------------------------------------------------
// Macro wrappers (preserved for call-site compatibility)
// source_location is captured automatically at each call site.
// ---------------------------------------------------------------------------

// clang-format off

/**
 * @brief Wrapper for assertion checking.
 *
 * This is meant to identify preconditions and possibly include additional
 * operations (e.g., an error message).
 */
#define EXPECTS(cond)          tamm_expects((cond), #cond)

/**
 * @brief Wrapper for assertion checking with a custom string message.
 */
#define EXPECTS_STR(cond, str) tamm_expects_str((cond), #cond, str)

#define EXPECTS_NOTHROW(cond)  assert(cond)

/**
 * @brief Mark code options that are not yet implemented.
 */
#define NOT_IMPLEMENTED()                                               \
    tamm_error("Not implemented")

/**
 * @brief Mark code options that are not yet allowed.
 */
#define NOT_ALLOWED()                                                   \
    tamm_error("Not allowed")

/**
 * @brief Mark code paths that should be unreachable.
 *
 * Uses C++23 std::unreachable() when available, otherwise aborts via
 * tamm_error + __builtin_unreachable for optimizer hints.
 */
#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#  include <utility>
#  define UNREACHABLE() (std::unreachable())
#else
#  define UNREACHABLE()                                                 \
    do {                                                                \
        tamm_error("Reached supposedly unreachable code");              \
        __builtin_unreachable();                                        \
    } while(0)
#endif

// clang-format on

void tamm_terminate(std::string msg);

} // namespace tamm
