#ifndef __TAMM_INTERMEDIATE_H__
#define __TAMM_INTERMEDIATE_H__

#include "absyn.h"
#include <vector>

#define MAX_TENSOR_DIMS 8
#define MAX_INDEX_NAMES 32


typedef struct AddOp_ *AddOp;
typedef struct MultOp_ *MultOp;

class RangeEntry {
public:
    char *name; /*name for this range*/
};

class IndexEntry {
public:
    char *name; /*name of this index*/
    int range_id; /*index into the RangeEntry struct*/
};

class TensorEntry {
public:
    char *name;
    int range_ids[MAX_TENSOR_DIMS]; /*dimensions in terms of index into RangeEntry struct*/
    int ndim, nupper;
};

typedef enum {
    OpTypeAdd,
    OpTypeMult
} OpType;

class OpEntry {
public:
    int op_id;
    OpType optype;
    AddOp add;
    MultOp mult;
};

typedef struct {
    std::vector<RangeEntry*> range_entries;
    std::vector<IndexEntry*> index_entries;
    std::vector<TensorEntry*> tensor_entries;
    std::vector<OpEntry*> op_entries;
} Equations;

/* tc[tc_ids] = alpha * ta[ta_ids]*/
struct AddOp_ {
    int tc, ta; //tc and ta in terms of index into TensorEntry structs
    double alpha;
    int tc_ids[MAX_TENSOR_DIMS]; /*index labels for tc in terms of index into IndexEntry struct*/
    int ta_ids[MAX_TENSOR_DIMS]; /*index labels for tc in terms of index into IndexEntry struct*/
};

/* tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]*/
struct MultOp_ {
    int tc, ta, tb; //tensors identified by index into TensorEntry structs
    double alpha;
    int tc_ids[MAX_TENSOR_DIMS];
    int ta_ids[MAX_TENSOR_DIMS];
    int tb_ids[MAX_TENSOR_DIMS];
};


void generate_intermediate_ast(Equations *eqn, TranslationUnit* root);

void generate_intermediate_CompoundElem(Equations *eqn, CompoundElem* celem);

void generate_intermediate_Elem(Equations *eqn, Elem* el);

void generate_intermediate_Decl(Equations *eqn, Decl* d);

void generate_intermediate_Stmt(Equations *eqn, Stmt* s);

void generate_intermediate_Exp(Equations *eqn, Exp* exp);

void generate_intermediate_ExpList(Equations *eqn, ExpList* expList, tamm_string am);

void generate_intermediate_DeclList(Equations *eqn, DeclList* dl);

void collectArrayRefs(Exp* exp, std::vector<Exp*> &arefs, double *alpha);

//tamm_string_array collectExpIndices(Exp* exp, int* first_ref); //Get each index only once

void getIndexIDs(Equations *eqn, Exp* e, int *);

void getTensorIDs(Equations *eqn, Exp* exp, int *tid);

#endif /*__TAMM_INTERMEDIATE_H__*/

