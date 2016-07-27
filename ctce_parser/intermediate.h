#ifndef __CTCE_INTERMEDIATE_H__
#define __CTCE_INTERMEDIATE_H__

#include "absyn.h"
#include "ctce_vector.h"

#define MAX_TENSOR_DIMS 8
#define MAX_INDEX_NAMES 32


//typedef struct Equations_ *Equations;
typedef struct Range2Index_ *Range2Index;
typedef struct IndexEntry_ *IndexEntry;
typedef struct RangeEntry_ *RangeEntry;
typedef struct TensorEntry_ *TensorEntry;
typedef struct OpEntry_ *OpEntry;
typedef struct AddOp_ *AddOp;
typedef struct MultOp_ *MultOp;




//static const char *OSTR = "O";
//static const char *VSTR = "V";
//static const char *NSTR = "N";



//void tensors_and_ops(Equations &eqs,
//                     std::vector<Tensor> &tensors,
//                     std::vector<Operation> &ops);

/* range of the indices, also used in triangular iterator */
typedef enum {
    TO, TV, TN, RANGE_UB
} RangeType;

typedef enum {
    P1B, P2B, P3B, P4B, P5B, P6B, P7B, P8B, P9B, P10B, P11B, P12B,
    H1B, H2B, H3B, H4B, H5B, H6B, H7B, H8B, H9B, H10B, H11B, H12B
} IndexName;

typedef enum {
    pIndex, hIndex
} IndexType;

typedef enum {
    V_tensor, T_tensor, F_tensor,
    iV_tensor, iT_tensor, iF_tensor,
    iVT_tensor, iVF_tensor, iTF_tensor
} TensorType;

typedef struct {
    vector range_entries;
    vector index_entries;
    vector tensor_entries;
    vector op_entries;
} Equations ;


struct Range2Index_ {
    int nindices;
    IndexName names[MAX_INDEX_NAMES];
} ;

struct RangeEntry_ {
    char *name; /*name for this range*/
} ;

struct IndexEntry_ {
    char *name; /*name of this index*/
    int range_id; /*index into the RangeEntry struct*/
} ;

struct TensorEntry_ {
    char *name;
    int range_ids[MAX_TENSOR_DIMS]; /*dimensions in terms of index into RangeEntry struct*/
    int ndim, nupper;
} ;

/* tc[tc_ids] = alpha * ta[ta_ids]*/
struct AddOp_ {
    int tc, ta; //tc and ta in terms of index into TensorEntry structs
    double alpha;
    int tc_ids[MAX_TENSOR_DIMS]; /*index labels for tc in terms of index into IndexEntry struct*/
    int ta_ids[MAX_TENSOR_DIMS]; /*index labels for tc in terms of index into IndexEntry struct*/
} ;

/* tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]*/
struct MultOp_ {
    int tc, ta, tb; //tensors identified by index into TensorEntry structs
    double alpha;
    int tc_ids[MAX_TENSOR_DIMS];
    int ta_ids[MAX_TENSOR_DIMS];
    int tb_ids[MAX_TENSOR_DIMS];
} ;

typedef enum {
    OpTypeAdd,
    OpTypeMult
} OpType;

struct OpEntry_ {
    OpType optype;
    AddOp add;
    MultOp mult;
} ;


void generate_intermediate_ast(Equations *eqn, TranslationUnit root);

void generate_intermediate_CompoundElem(Equations *eqn, CompoundElem celem);

void generate_intermediate_Elem(Equations *eqn, Elem el);

void generate_intermediate_Decl(Equations *eqn, Decl d);

void generate_intermediate_Stmt(Equations *eqn, Stmt s);

void generate_intermediate_Exp(Equations *eqn, Exp exp);

void generate_intermediate_ExpList(Equations *eqn, ExpList expList, string am);

void generate_intermediate_DeclList(Equations *eqn, DeclList dl);


#endif /*__CTCE_INTERMEDIATE_H__*/

