#ifndef __ctce_input_h__
#define __ctce_input_h__

#include "define.h"
#include "tensor.h"
#include "expression.h"

#define MAX_TENSOR_DIMS 8
#define MAX_INDEX_NAMES 32

namespace ctce {

static const char *OSTR = "O";
static const char *VSTR = "V";
static const char *NSTR = "N";

typedef struct {
	int nindices;
	IndexName names[MAX_INDEX_NAMES];
} Range2Index;

/* typedef enum { */
/* 	TENSOR_INPUT, */
/* 	TENSOR_OUTPUT, */
/* 	TENSOR_TEMP */
/* } TensorType; */

typedef struct {
	char * name; /*name for this range*/
	RangeType rt; /*type for range*/
} RangeEntry;

typedef struct {
	char * name; /*name of this index*/
	RangeEntry *range;  /*range declaration for this index*/
	IndexName index;
} IndexEntry;

typedef struct {
	char * name;
	RangeEntry *dims[MAX_TENSOR_DIMS];
	int ndim, nupper;
	Tensor *tensor;
} TensorEntry;

/* tc[tc_ids] = alpha * ta[ta_ids]*/
typedef struct {
	TensorEntry *tc, *ta;
	double alpha;
	IndexEntry *tc_ids[MAX_TENSOR_DIMS];
	IndexEntry *ta_ids[MAX_TENSOR_DIMS];
} AddOp;

/* tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]*/
typedef struct {
	TensorEntry *tc, *ta, *tb;
	double alpha;
	IndexEntry *tc_ids[MAX_TENSOR_DIMS];
	IndexEntry *ta_ids[MAX_TENSOR_DIMS];	
	IndexEntry *tb_ids[MAX_TENSOR_DIMS];	
} MultOp;

typedef enum {
	OpTypeAdd,
	OpTypeMult
} OpType;

typedef struct {
	OpType optype;
	void *op_entry; /*AddOp or MultOp*/
  Assignment add; /*Assignment of Multiplication*/
  Multiplication mult;
} Operation;

void input_initialize(int num_ranges, RangeEntry *ranges,
											int num_indices, IndexEntry *indices,
											int num_tensors, TensorEntry *tensors,
											int num_operations, Operation *ops);

void input_ops_initialize(int num_ranges, RangeEntry *ranges,
                          int num_indices, IndexEntry *indices,
                          int num_tensors, TensorEntry *tensors,
                          int num_operations, Operation *ops);

};

#endif /*__ctce_input_h__*/


