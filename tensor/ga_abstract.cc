#include "ga_abstract.h"
#include "ga.h"

using namespace std;

namespace tamm {
namespace gmem {
Handle NULL_HANDLE = (int)0;
char global_ops[6][7] = {"+", "*", "max", "min", "absmin", "absmax"};

bool Handle::valid() { return value != 0; }

// FIXME: Make explicit for c++11
Handle::Handle(uint64_t conversion) : value{conversion} {};
Handle::operator uint64_t() const { return value; }

Handle create(Types type, int size, char *name) {
  Handle handle;
  int ga_type;
  switch (type) {
    case Int:
      ga_type = C_INT;
      break;
    case Double:
      ga_type = MT_C_DBL;
      break;
  }
  handle.value = NGA_Create(ga_type, 1, &size, name, NULL);
  assert(handle.value != 0);

  return handle;
}

void zero(Handle handle) { NGA_Zero(handle.value); }

int64_t atomic_fetch_add(Handle handle, int pos, long amount) {
  return (int64_t)NGA_Read_inc(handle.value, &pos, amount);
}

uint64_t ranks() { return GA_Nnodes(); }
void sync() { GA_Sync(); }

void destroy(Handle handle) { GA_Destroy(handle.value); }

void get(Handle handle, void *buf, int start, int stop) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;

  NGA_Get(handle.value, lo, hi, buf, &tmp);
}

void get(Handle handle, void *buf, int start, int stop, Wait_Handle &wait) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;
  // ga_nbhdl_t conversion = (ga_ndhdl_t)(uint64_t) wait.value;
  NGA_NbGet(handle.value, lo, hi, buf, &tmp, (ga_nbhdl_t *)&wait.value);
}

void wait(Wait_Handle &wait) {
  if (wait.value) {
    // ga_nbhdl_t conversion = (ga_ndhdl_t)(uint64_t) wait.value;
    NGA_NbWait((ga_nbhdl_t *)&wait.value);
  }
}

void acc(Handle handle, void *buf, int start, int stop) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;

  double alpha = 1.0;
  NGA_Acc(handle.value, lo, hi, buf, &tmp, &alpha);
}

void op(double x[], int n, Operation op) { GA_Dgop(x, n, global_ops[op]); }
}
}
