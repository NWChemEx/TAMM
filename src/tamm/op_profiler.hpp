#pragma once

namespace tamm {

class OpProfiler {
private:
  OpProfiler() {}

public:
  int    mult_counter  = 0;
  double multOpTime    = 0;
  double addOpTime     = 0;
  double setOpTime     = 0;
  double allocOpTime   = 0;
  double deallocOpTime = 0;
  double tgetTime      = 0;
  double taddTime      = 0;
  double twaitTime     = 0;
  double tBCTime       = 0;
  double tcopyTime     = 0;
  double tbarrierTime  = 0;

  double multOpGetTime  = 0;
  double multOpAddTime  = 0;
  double multOpWaitTime = 0;
  double multOpCopyTime = 0;
  double multOpBCTime   = 0;

  // Lazy-tensor block construction. Charged only at the outermost lazy read (lazyDepth),
  // so a lazy tensor whose source is itself lazy is not counted twice.
  double  lazyBlockTime  = 0;
  int64_t lazyBlockCount = 0;
  int     lazyDepth      = 0;

  inline static OpProfiler& instance() {
    static OpProfiler op_prof;
    return op_prof;
  }

  OpProfiler(const OpProfiler&)            = delete;
  OpProfiler& operator=(const OpProfiler&) = delete;
  OpProfiler(OpProfiler&&)                 = delete;
  OpProfiler& operator=(OpProfiler&&)      = delete;
};

} // namespace tamm
