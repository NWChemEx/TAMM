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
    const std::string range_name; /*name for this range*/
    RangeEntry(const std::string range_name): range_name(range_name) {}
    ~RangeEntry() {}    
};

class IndexEntry {
public:
    const std::string index_name; /*name of this index*/
    const int range_id; /*index into the RangeEntry vector */
    IndexEntry(const std::string index_name, const int range_id): index_name(index_name), range_id(range_id) {}
    ~IndexEntry() {}
};

class TensorEntry {
public:
    const std::string tensor_name;
    const int ndim, nupper;
    std::vector<int> range_ids; /*dimensions in terms of index into RangeEntry vector */
    TensorEntry(const std::string tensor_name, const int ndim, const int nupper)
                : tensor_name(tensor_name), ndim(ndim), nupper(nupper) {}
    ~TensorEntry() {}
    
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


void generate_equations(CompilationUnit* const root, Equations* const equation);

void generate_equations_CompoundElement(const CompoundElement* const ce, Equations* const equations);

void generate_equations_Element(Element* const element, Equations* const equations);

void generate_equations_DeclarationList(const DeclarationList* const decllist, Equations* const equations);

void generate_equations_Statement(Statement* const statement, Equations* const equations);

void generate_equations_AssignStatement(const AssignStatement* const statement, Equations* const equations);

void generate_equations_Declaration(Declaration* const declaration, Equations* const equations);

// void collectArrayRefs(Exp* exp, std::vector<Exp*> &arefs, int *num_adds);

// //tamm_string_array collectExpIndices(Exp* exp, int* first_ref); //Get each index only once

// void getIndexIDs(Equations *eqn, Exp* e, int *);

// void getTensorIDs(Equations *eqn, Exp* exp, int *tid);

}
#endif /*__TAMM_INTERMEDIATE_H__*/

