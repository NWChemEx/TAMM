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
#ifndef TAMM_TENSOR_INPUT_H_
#define TAMM_TENSOR_INPUT_H_

#include <string>
#include "tensor/define.h"
#include "tensor/expression.h"
#include "tensor/tensor.h"

#define MAX_TENSOR_DIMS 8
#define MAX_INDEX_NAMES 32

namespace tamm {

static const char *OSTR = "O";
static const char *VSTR = "V";
static const char *NSTR = "N";

typedef struct {
  int nindices;
  IndexName names[MAX_INDEX_NAMES];
} Range2Index;

typedef struct { std::string name; /*name for this range*/ } RangeEntry;

typedef struct {
  std::string name; /*name of this index*/
  int range_id;     /*index into the RangeEntry struct*/
} IndexEntry;

typedef struct {
  std::string name;
  int range_ids
      [MAX_TENSOR_DIMS]; /*dimensions in terms of index into RangeEntry struct*/
  int ndim, nupper;
} TensorEntry;

/* tc[tc_ids] = alpha * ta[ta_ids]*/
typedef struct {
  // int tc, ta; //tc and ta in terms of index into TensorEntry structs
  std::string tc, ta;
  double alpha;
  int tc_ids[MAX_TENSOR_DIMS];  // index labels for tc in terms of index into
                                // IndexEntry struct
  int ta_ids[MAX_TENSOR_DIMS];  // index labels for tc in terms of index into
                                // IndexEntry struct
} AddOp;

/* tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]*/
typedef struct {
  // int tc, ta, tb; //tensors identified by index into TensorEntry structs
  std::string tc, ta, tb;
  double alpha;
  int tc_ids[MAX_TENSOR_DIMS];
  int ta_ids[MAX_TENSOR_DIMS];
  int tb_ids[MAX_TENSOR_DIMS];
} MultOp;

typedef enum { OpTypeAdd, OpTypeMult } OpType;

typedef struct {
  int op_id;
  OpType optype;
  AddOp add;
  MultOp mult;
} OpEntry;

typedef struct {
  OpType optype;
  Assignment add;
  Multiplication mult;
} Operation;
};  // namespace tamm

#endif  // TAMM_TENSOR_INPUT_H_
