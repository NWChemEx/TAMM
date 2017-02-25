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

#include "Intermediate.h"
#include "Util.h"
#include <cassert>

namespace tamm {

static int op_id = 1;

void generate_equations(CompilationUnit* const root, Equations* const equations) {
    std::vector<RangeEntry*> &re = equations->range_entries;
    re.push_back(new RangeEntry("O"));
    re.push_back(new RangeEntry("V"));
    re.push_back(new RangeEntry("N"));

    for (auto &ce: root->celist)
        generate_equations_CompoundElement(ce, equations);
}

void generate_equations_CompoundElement(const CompoundElement* const ce, Equations* const equations) {
    const ElementList* const element_list = ce->elist;
    for (auto &elem: element_list->elist) 
        generate_equations_Element(elem, equations);
}

void generate_equations_Element(Element* const element, Equations* const equations) {    
     if (element == nullptr) return; /// TODO: Can this happen ?

     //if (element->getElementType() == Element::kDeclaration);
     if (DeclarationList* const dl = dynamic_cast<DeclarationList*>(element))
            generate_equations_DeclarationList(dl, equations);
     else if (Statement* const statement = dynamic_cast<Statement*>(element))
            generate_equations_Statement(statement, equations);
     else ;
//             std::cerr << "Not a Declaration or Statement!\n";
//             std::exit(EXIT_FAILURE);
}

void generate_equations_DeclarationList(const DeclarationList* const decllist, Equations* const equations) {    
    for (auto &decl: decllist->dlist)
        generate_equations_Declaration(decl, equations);
}

void generate_equations_Declaration(Declaration* const declaration, Equations* const equations) { 
    if (RangeDeclaration* const rdecl = dynamic_cast<RangeDeclaration*>(declaration)) ;
    else if (IndexDeclaration* const idecl = dynamic_cast<IndexDeclaration*>(declaration)) {
            const std::string index_name = idecl->index_name->name;
            int rid = 0;
            if (index_name == "O") rid = 0;
            else if (index_name == "V") rid = 1;
            else if (index_name == "N") rid = 2;
            equations->index_entries.push_back(new IndexEntry(index_name, rid));
    }

    else if (ArrayDeclaration* const adecl = dynamic_cast<ArrayDeclaration*>(declaration)) {
        TensorEntry* te = new TensorEntry(adecl->tensor_name->name,
            adecl->upper_indices.size()+adecl->lower_indices.size(), adecl->upper_indices.size());
        for (auto &ui: adecl->upper_indices)
            if(ui->name == "O") te->range_ids.push_back(0);
            else if(ui->name == "V") te->range_ids.push_back(1);
            else if(ui->name == "N") te->range_ids.push_back(2);
        for (auto &li: adecl->lower_indices)
            if(li->name == "O") te->range_ids.push_back(0);
            else if(li->name == "V") te->range_ids.push_back(1);
            else if(li->name == "N") te->range_ids.push_back(2);

        equations->tensor_entries.push_back(te);
    }
}


/// Get integer ids for indexes in array refs - Look up in index_entries vector
void get_index_ids(Equations* const equations, Array* const aref, std::vector<int>& tensor_ids) {
    IndexEntry* index_entry = nullptr;
    for(auto &index: aref->indices) {
        int index_pos = 0;
        for (auto &ientry: equations->index_entries) {
            index_entry = dynamic_cast<IndexEntry*>(ientry);
            if (index->name == index_entry->index_name) {
                tensor_ids.push_back(index_pos);
                break;
            }
            index_pos++;
        }
    }    
}


const int get_tensor_id(Equations* const equations, Array* const aref) {
    int tid = -1;
    const std::string tensor_ref_name = aref->tensor_name->name;
    int tensor_pos = 0;
    TensorEntry* tensor_entry = nullptr;
    for (auto &tentry: equations->tensor_entries) {
        tensor_entry = dynamic_cast<TensorEntry*>(tentry);
        if (tensor_ref_name == tensor_entry->tensor_name) {
            tid = tensor_pos;
            break;
        }
        tensor_pos++;
    }    
    assert(tid!=-1);
    return tid;
}


void generate_equations_Statement(Statement* const statement, Equations* const equations) {    
    if (AssignStatement* const as = dynamic_cast<AssignStatement*>(statement)) 
        generate_equations_AssignStatement(as, equations);
}

void generate_equations_AssignStatement(const AssignStatement* const statement, Equations* const equations) { 
        Array* const lhs_tref = statement->lhs;
        Expression* const rhs  = statement->rhs;

        std::vector<Array*> rhs_arefs;
        std::vector<NumConst*> rhs_consts;
        get_all_refs_from_expression(rhs, rhs_arefs, rhs_consts);

        const int num_rhs_arefs = rhs_arefs.size();
        assert(rhs_consts.size() == 0 || rhs_consts.size()==1);
        assert(num_rhs_arefs ==1 || num_rhs_arefs == 2);
        //assert ((Addition* const add = dynamic_cast<Addition*>(rhs)));
        Addition* const add = dynamic_cast<Addition*>(rhs);
        const std::vector<std::string> add_operators = add->add_operators;
        assert(add_operators.size() == 0 || add_operators.size() == 1);
        const bool first_op = add->first_op;

        float alpha = 1.0;
        if (add_operators.size()==1) if (add_operators.at(0) == "-") alpha = -1.0;
        if (rhs_consts.size()==1) {
            NumConst* const nc = dynamic_cast<NumConst*>(rhs_consts.at(0));
            alpha = alpha * nc->value;
        }
        /// An Add or Mult can have only one add_operator and one constant
        if  (num_rhs_arefs == 1){
            /// Add Op
            AddOp* const aop = new AddOp();
            aop->alpha = alpha;

            get_index_ids(equations,lhs_tref,aop->tc_ids);
            get_index_ids(equations,rhs_arefs.at(0),aop->ta_ids);

            assert(aop->tc_ids.size() == lhs_tref->indices.size());
            assert(aop->ta_ids.size() == rhs_arefs.at(0)->indices.size());

            aop->tc = get_tensor_id(equations,lhs_tref);
            aop->ta = get_tensor_id(equations,rhs_arefs.at(0));

            equations->op_entries.push_back(new OpEntry(op_id,OpTypeAdd,aop,nullptr));
            op_id++;

        }
        else if (num_rhs_arefs == 2){
            /// Mult Op
            MultOp* const mop = new MultOp();
            mop->alpha = alpha;

            get_index_ids(equations,lhs_tref,mop->tc_ids);
            get_index_ids(equations,rhs_arefs.at(0),mop->ta_ids);
            get_index_ids(equations,rhs_arefs.at(1),mop->tb_ids);

            assert(mop->tc_ids.size() == lhs_tref->indices.size());
            assert(mop->ta_ids.size() == rhs_arefs.at(0)->indices.size());
            assert(mop->tb_ids.size() == rhs_arefs.at(1)->indices.size());

            mop->tc = get_tensor_id(equations,lhs_tref);
            mop->ta = get_tensor_id(equations,rhs_arefs.at(0));
            mop->tb = get_tensor_id(equations,rhs_arefs.at(1));

            equations->op_entries.push_back(new OpEntry(op_id, OpTypeMult, nullptr, mop));
            op_id++;

        }

        else ; /// Feature not implemented. print num_rhs_arefs

}

}