//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include "tensor/gmem.h"
#include <ga.h>

namespace tamm {
namespace gmem {
Handle NULL_HANDLE = static_cast<Handle>(0);
char global_ops[6][7] = {"+", "*", "max", "min", "absmin", "absmax"};

bool Handle::valid() { return value != 0; }

// FIXME: Make explicit for c++11
Handle::Handle(uintptr_t conversion) : value{conversion} {};
Handle::operator uintptr_t() const { return value; }

Handle create(Types type, int64_t size, std::string name) {
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
  /// @todo: const_cast is bad. Fix it later
  handle.value = NGA_Create64(ga_type, 1, &size, const_cast<char*>(name.c_str()), nullptr);
  assert(handle.value != 0);

  return handle;
}

void zero(Handle handle) { NGA_Zero(static_cast<int>(handle.value)); }

int64_t atomic_fetch_add(Handle handle, int pos, int amount) {
  return static_cast<int64_t>(
      NGA_Read_inc(static_cast<int>(handle.value), &pos, amount));
}

uint64_t ranks() { return GA_Nnodes(); }
void sync() { GA_Sync(); }

void destroy(Handle handle) { GA_Destroy(handle.value); }

void get(Handle handle, void *buf, int start, int stop) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;

  NGA_Get(static_cast<int>(handle.value), lo, hi, buf, &tmp);
}

void get(Handle handle, void *buf, int start, int stop, Wait_Handle *wait) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;
  NGA_NbGet(static_cast<int>(handle.value), lo, hi, buf, &tmp,
            reinterpret_cast<ga_nbhdl_t *>(wait->value));
}

void wait(const Wait_Handle &wait) {
  if (wait.value) {
    NGA_NbWait(reinterpret_cast<ga_nbhdl_t *>(wait.value));
  }
}

void acc(Handle handle, void *buf, int start, int stop) {
  int lo[2] = {0, start};
  int hi[2] = {0, stop};
  int tmp = 0;

  double alpha = 1.0;
  NGA_Acc(static_cast<int>(handle.value), lo, hi, buf, &tmp, &alpha);
}

void op(double x[], int n, Operation op) { GA_Dgop(x, n, global_ops[op]); }
}  // namespace gmem
}  // namespace tamm
