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
#ifndef TAMM_TENSOR_GMEM_H_
#define TAMM_TENSOR_GMEM_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string>

namespace tamm {
namespace gmem {
struct Handle {
  Handle() {}
  bool valid();

  // FIXME: Make explicit for c++11
  explicit Handle(uintptr_t conversion);
  operator uintptr_t() const;

  uintptr_t value;
};

struct Wait_Handle {
  uintptr_t value;
};

extern Handle NULL_HANDLE;

enum Types { Int, Double };

enum Operation { Plus = 0, Multiply, Max, Min, AbsMax, AbsMin };

Handle create(Types type, int64_t size, std::string name);
uint64_t rank();
uint64_t ranks();
int64_t atomic_fetch_add(Handle handle, int pos, int amount);
void zero(Handle handle);
void sync();
void destroy(Handle handle);
void get(Handle handle, void* buf, int start, int stop);
void get(Handle handle, void* buf, int start, int stop, Wait_Handle* wait);
void acc(Handle handle, void* buf, int start, int stop);
void op(double x[], int n, Operation op);
void wait(const Wait_Handle& wait);
void destroy(Handle handle);
}  // namespace gmem
}  // namespace tamm

#endif  // TAMM_TENSOR_GMEM_H_
