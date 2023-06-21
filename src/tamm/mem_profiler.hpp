#pragma once

namespace tamm {

class MemProfiler {
private:
  MemProfiler() {}

public:
  int    alloc_counter          = 0;
  int    dealloc_counter        = 0;
  double mem_allocated          = 0.0;
  double mem_deallocated        = 0.0;
  double max_in_single_allocate = 0.0;
  double max_total_allocated    = 0.0;

  inline static MemProfiler& instance() {
    static MemProfiler mem_prof;
    return mem_prof;
  }

  MemProfiler(const MemProfiler&)            = delete;
  MemProfiler& operator=(const MemProfiler&) = delete;
  MemProfiler(MemProfiler&&)                 = delete;
  MemProfiler& operator=(MemProfiler&&)      = delete;
};

} // namespace tamm
