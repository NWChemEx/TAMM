#ifndef TAMMX_PROC_GROUP_H_
#define TAMMX_PROC_GROUP_H_

namespace tammx {

// @@todo implement
class ProcGroup {
 public:
  Proc size() {
    return Proc{1};
  }
  Proc rank() {
    return Proc{0};
  }
};

} // namespace tammx


#endif // TAMMX_PROC_GROUP_H_

