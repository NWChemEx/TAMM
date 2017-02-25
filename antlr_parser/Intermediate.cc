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

namespace tamm {

// class ArrayRefAlpha {
// public:
//     double alpha;
//     Exp* aref;
// };

// ArrayRefAlpha* make_ArrayRefAlpha(double alpha, Exp* aref) {
//     ArrayRefAlpha* p = new ArrayRefAlpha();
//     p->alpha = alpha;
//     p->aref = aref;
//     return p;
// }

// IndexEntry* make_IndexEntry(tamm_string name, int range_id) {
//     IndexEntry* p = new IndexEntry();
//     p->name = name;
//     p->range_id = range_id;
//     return p;
// }

// TensorEntry* make_TensorEntry(tamm_string name, int ndim, int nupper) {
//     TensorEntry* p = new TensorEntry();
//     p->name = name;
//     p->ndim = ndim;
//     p->nupper = nupper;
//     return p;
// }

// OpEntry* make_OpEntry(int op_id, OpType ot, AddOp ao, MultOp mo) {
//     OpEntry* p = new OpEntry();
//     p->op_id = op_id;
//     p->optype = ot;
//     p->add = ao;
//     p->mult = mo;
//     return p;
// }


// AddOp make_AddOp(int tc, int ta, double alpha) {
//     AddOp p = (AddOp)tce_malloc(sizeof(*p));
//     p->ta = ta;
//     p->tc = tc;
//     p->alpha = alpha;
//     return p;
// }

// MultOp make_MultOp(int tc, int ta, int tb, double alpha) {
//     MultOp p = (MultOp)tce_malloc(sizeof(*p));
//     p->ta = ta;
//     p->tc = tc;
//     p->tb = tb;
//     p->alpha = alpha;
//     return p;
// }

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

void generate_equations_Statement(Statement* const statement, Equations* const equations) {    
    if (AssignStatement* const as = dynamic_cast<AssignStatement*>(statement)) 
        generate_equations_AssignStatement(as, equations);
}

void generate_equations_AssignStatement(const AssignStatement* const statement, Equations* const equations) { 
    ;
}

// void generate_intermediate_Stmt(Equations *eqn, Stmt* s) {
//     std::vector<Exp*> lhs_aref, rhs_allref;
//     std::vector<ArrayRefAlpha*> rhs_aref;
//     int num_adds = 1;
//     switch (s->kind) {
//       case Stmt::is_AssignStmt: {
//         num_adds = 1;
//         collectArrayRefs(s->u.AssignStmt.lhs, lhs_aref, &num_adds);
//         int i = 0;
// //            for (i = 0; i < vector_count(&lhs_aref); i++) {
// //                Exp* e = vector_get(&lhs_aref, i);
// //                std::cout << e->u.Array.name << " ";
// //            }
//         collectArrayRefs(s->u.AssignStmt.rhs, rhs_allref, &num_adds);

// //        int ignore_first_ref = 0;
// //        if (strcmp(s->u.AssignStmt.astype, "+=") == 0 || strcmp(s->u.AssignStmt.astype, "-=") == 0)
// //          ignore_first_ref = 1;

// //        tce_string_array lhs_indices = (tce_string_array)collectExpIndices(s->u.AssignStmt.lhs, &ignore_first_ref);
// //        tce_string_array rhs_indices = (tce_string_array)collectExpIndices(s->u.AssignStmt.rhs, &ignore_first_ref);

// //            print_index_list(lhs_indices);
// //            std::cout << "="
// //            print_index_list(rhs_indices);

// //            int tc_ids[MAX_TENSOR_DIMS];
// //            getIndexIDs(eqn, vector_get(&lhs_aref,0), tc_ids);

// //            for (i=0;i<MAX_TENSOR_DIMS;i++)
// //                if(tc_ids[i]!=-1) std::cout << tc_ids[i] ", ";


//         int rhs_aref_count = 0;
//         for (i = 0; i < rhs_allref.size(); i++) {
//           Exp* e = (Exp*) rhs_allref.at(i);
//           if (e->kind == Exp::is_NumConst) {
//             Exp* e1 = (Exp*)rhs_allref.at(i + 1);
//             if (e1->kind == Exp::is_ArrayRef) {
//               rhs_aref_count++;
//               rhs_aref.push_back(make_ArrayRefAlpha(e->u.NumConst.value * e->coef * e1->coef, e1));
//               i++;
//             }
//           } else {
//             rhs_aref_count++;
//             Exp* e1 = (Exp*)rhs_allref.at(i);
//             rhs_aref.push_back(make_ArrayRefAlpha(1.0 * e1->coef, e1));
//           }
//         }

//         bool rhs_first_ref = false;
// //        tce_string_array rhs_first_ref_indices = (tce_string_array)collectExpIndices(
// //            ((ArrayRefAlpha) vector_get(&rhs_aref, 0))->aref, &ignore_first_ref);

//         if (rhs_aref.size() > 1) {
//           Exp* tc_exp = (Exp*)lhs_aref.at(0);
//           Exp* ta_exp = ((ArrayRefAlpha*) rhs_aref.at(0))->aref;
//           if (strcmp(tc_exp->u.Array.name, ta_exp->u.Array.name) == 0) rhs_first_ref = true;
//         }

//         //Exp* tcp = vector_get(&lhs_aref, 0);
//         //std::cout << "name = " << tcp->u.Array.name << std::endl;

// //            tamm_bool isAMOp = (exact_compare_index_lists(lhs_indices, rhs_indices));
// //            //a1121[p3,h1,p2,h2] = t_vo[p3,h1] * t_vo[p2,h2];
// //            tamm_bool firstRefInd = (lhs_indices->length > rhs_first_ref_indices->length);
// //            tamm_bool isEqInd = (lhs_indices->length == rhs_indices->length);
// //            tamm_bool isAddOp = isEqInd && isAMOp;
// //            tamm_bool isMultOp =
// //                    (lhs_indices->length < rhs_indices->length) || (isEqInd && !isAMOp) || (isEqInd && firstRefInd);

//         bool isAddOp = false;
//         bool isMultOp = false;

//         if (rhs_first_ref) rhs_aref_count -= 1;

//         if (rhs_aref_count == 2) isMultOp = true;
        
//         else if (rhs_aref_count == 1 || rhs_aref_count > 2) isAddOp = true;

//         if (isMultOp) {
//           //std::cout << " == MULT OP\n";

//           Exp* tc_exp = (Exp*)lhs_aref.at(0);
//           int ta_ind = 0;
//           if (rhs_first_ref) ta_ind++;

//           MultOp mop = make_MultOp(0, 0, 0, ((ArrayRefAlpha*) rhs_aref.at(ta_ind))->alpha);

//           Exp* ta_exp = ((ArrayRefAlpha*) rhs_aref.at(ta_ind))->aref;
//           Exp* tb_exp = ((ArrayRefAlpha*) rhs_aref.at(ta_ind + 1))->aref;

//           getIndexIDs(eqn, tc_exp, mop->tc_ids);
//           getIndexIDs(eqn, ta_exp, mop->ta_ids);
//           getIndexIDs(eqn, tb_exp, mop->tb_ids);

//           getTensorIDs(eqn, tc_exp, &mop->tc);
//           getTensorIDs(eqn, ta_exp, &mop->ta);
//           getTensorIDs(eqn, tb_exp, &mop->tb);

//           eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeMult, nullptr, mop));
//           op_id++;

//         } else if (isAddOp) {

//           //std::cout << " == ADD OP\n";

//           Exp* tc_exp = (Exp*) lhs_aref.at(0);

//           int ta_ind = 0;
//           if (rhs_first_ref) ta_ind++;

//           AddOp mop = make_AddOp(0, 0, ((ArrayRefAlpha*) rhs_aref.at(ta_ind))->alpha);

//           Exp* ta_exp = ((ArrayRefAlpha*) rhs_aref.at(ta_ind))->aref;

//           getIndexIDs(eqn, tc_exp, mop->tc_ids);
//           getIndexIDs(eqn, ta_exp, mop->ta_ids);

//           getTensorIDs(eqn, tc_exp, &mop->tc);
//           getTensorIDs(eqn, ta_exp, &mop->ta);

//           eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeAdd, mop, nullptr));
//           op_id++;

//           if (rhs_aref.size() > ta_ind + 1) {
//             int k;
//             for (k = ta_ind + 1; k < rhs_aref.size(); k++) {
//               AddOp aop = make_AddOp(0, 0, ((ArrayRefAlpha*) rhs_aref.at(k))->alpha);

//               Exp* ta_exp = ((ArrayRefAlpha*) rhs_aref.at(k))->aref;

//               getIndexIDs(eqn, tc_exp, aop->tc_ids);
//               getIndexIDs(eqn, ta_exp, aop->ta_ids);

//               getTensorIDs(eqn, tc_exp, &aop->tc);
//               getTensorIDs(eqn, ta_exp, &aop->ta);

//               eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeAdd, aop, nullptr));
//               op_id++;
//             }
//           }

//         } else {
//           std::cerr <<  "NEITHER ADD OR MULT OP.. THIS SHOULD NOT HAPPEN!\n";
//           std::exit(EXIT_FAILURE);
//         }
//       }
//             break;
//         default: {
//           std::cerr <<  "Not an Assignment Statement!\n";
//           std::exit(EXIT_FAILURE);
//         }
//     }
// }


// void getTensorIDs(Equations *eqn, Exp* exp, int *tid) {
//     if (exp->kind == Exp::is_ArrayRef) {
//         tamm_string aname = exp->u.Array.name;
//         int j;
//         TensorEntry* ient = nullptr;
//         for (j = 0; j < eqn->tensor_entries.size(); j++) {
//             ient = (TensorEntry*)eqn->tensor_entries.at(j);
//             if (strcmp(aname, ient->name) == 0) {
//                 *tid = j;
//                 break;
//             }
//         }
//     }
// }


// void getIndexIDs(Equations *eqn, Exp* exp, int *tc_ids) {

//     int i;
//     for (i = 0; i < MAX_TENSOR_DIMS; i++) tc_ids[i] = -1;
//     if (exp->kind == Exp::is_ArrayRef) {
//         tamm_string *aind = exp->u.Array.indices;
//         int len = exp->u.Array.length;
//         int j;
//         int ipos = 0;
//         IndexEntry* ient = nullptr;
//         for (i = 0; i < len; i++) {
//             for (j = 0; j < eqn->index_entries.size(); j++) {
//                 ient = (IndexEntry*)eqn->index_entries.at(j);
//                 if (strcmp(aind[i], ient->name) == 0) {
//                     tc_ids[ipos] = j;
//                     ipos++;
//                     break;
//                 }
//             }
//         }

//     }
// }



// void generate_intermediate_Exp(Equations *eqn, Exp* exp) {
//     switch (exp->kind) {
//       case Exp::is_Parenth: {
//         generate_intermediate_Exp(eqn, exp->u.Parenth.exp);
//       }
//             break;
//         case Exp::is_NumConst: {}
//             //std::cout << exp->u.NumConst.value << " ";
//             break;
//         case Exp::is_ArrayRef: {}
//             //std::cout << "%s[%s] ", exp->u.Array.name, combine_indices(exp->u.Array.indices, exp->u.Array.length);
//             break;
//         case Exp::is_Addition: {
//           generate_intermediate_ExpList(eqn, exp->u.Addition.subexps, "+");
//         }
//             break;
//         case Exp::is_Multiplication: {
//           generate_intermediate_ExpList(eqn, exp->u.Multiplication.subexps, "*");
//         }
//             break;
//         default: {
//           std::cerr <<  "Not a valid Expression!\n";
//           std::exit(EXIT_FAILURE);
//         }
//     }
// }






}