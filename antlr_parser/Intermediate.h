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

#ifndef __TAMM_INTERMEDIATE_H__
#define __TAMM_INTERMEDIATE_H__

#include "Absyn.h"
#include <vector>

namespace tamm{

class RangeEntry {
public:
    std::string name; /*name for this range*/
};

class IndexEntry {
public:
    std::string name; /*name of this index*/
    const int range_id; /*index into the RangeEntry struct*/
};

class TensorEntry {
public:
    std::string name;
    int ndim, nupper;
    std::vector<int> range_ids; /*dimensions in terms of index into RangeEntry struct*/
    
};


/* tc[tc_ids] = alpha * ta[ta_ids]*/
class AddOp {
    const int tc, ta; //tc and ta in terms of index into TensorEntry structs
    const double alpha;
   /*index labels for tc in terms of index into IndexEntry struct*/
    std::vector<int> tc_ids;
    std::vector<int> ta_ids;
};

/* tc[tc_ids] += alpha * ta[ta_ids] * tb[tb_ids]*/
class MultOp {
    public:
        const int tc, ta, tb; //tensors identified by index into TensorEntry structs
        const double alpha;
        std::vector<int> tc_ids;
        std::vector<int> ta_ids;
        std::vector<int> tb_ids;
};

typedef enum {
    OpTypeAdd,
    OpTypeMult
} OpType;

class OpEntry {
public:
    int op_id;
    OpType optype;
    AddOp* add;
    MultOp* mult;
};

class Equations {
    public:
        std::vector<RangeEntry*> range_entries;
        std::vector<IndexEntry*> index_entries;
        std::vector<TensorEntry*> tensor_entries;
        std::vector<OpEntry*> op_entries;
};


void generate_equations(Equations* const equation, CompilationUnit* const root);

// void generate_intermediate_CompoundElem(Equations *eqn, CompoundElem* celem);

// void generate_intermediate_Elem(Equations *eqn, Element* el);

// void generate_intermediate_Decl(Equations *eqn, Declaration* d);

// void generate_intermediate_Stmt(Equations *eqn, Stmt* s);

// void generate_intermediate_Exp(Equations *eqn, Exp* exp);

// void generate_intermediate_ExpList(Equations *eqn, ExpList* expList, tamm_string am);

// void generate_intermediate_DeclList(Equations *eqn, DeclList* dl);

// void collectArrayRefs(Exp* exp, std::vector<Exp*> &arefs, int *num_adds);

// //tamm_string_array collectExpIndices(Exp* exp, int* first_ref); //Get each index only once

// void getIndexIDs(Equations *eqn, Exp* e, int *);

// void getTensorIDs(Equations *eqn, Exp* exp, int *tid);

}
#endif /*__TAMM_INTERMEDIATE_H__*/

