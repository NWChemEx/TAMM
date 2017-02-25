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

#include "Semant.h"
#include "Error.h"
#include "Entry.h"
#include <cassert>
#include <algorithm>

namespace tamm {

void type_check(const CompilationUnit* const root, SymbolTable* const context) {
    for (auto &ce: root->celist)
        check_CompoundElement(ce, context);
}

void check_CompoundElement(const CompoundElement* const ce, SymbolTable* const context) {
    const ElementList* const element_list = ce->elist;
    for (auto &elem: element_list->elist) 
        check_Element(elem, context);
}

void check_Element(Element* const element, SymbolTable* const context) {    
     if (element == nullptr) return; /// TODO: Can this happen ?

     //if (element->getElementType() == Element::kDeclaration);
     if (DeclarationList* const dl = dynamic_cast<DeclarationList*>(element))
            check_DeclarationList(dl, context);
     else if (Statement* const statement = dynamic_cast<Statement*>(element))
            check_Statement(statement, context);
     else ;
//             std::cerr << "Not a Declaration or Statement!\n";
//             std::exit(EXIT_FAILURE);
}

void check_DeclarationList(const DeclarationList* const decllist, SymbolTable* const context) {
    for (auto &decl: decllist->dlist)
        check_Declaration(decl, context);
}

void check_Statement(Statement* const statement, SymbolTable* const context) {
    if (AssignStatement* const as = dynamic_cast<AssignStatement*>(statement)) ;
        //check_AssignStatement(as,context);
    else ; //error- cannot happen
}

void check_AssignStatement(const AssignStatement* const statement, SymbolTable* const context) {
//     switch (s->kind) {
//         case Stmt::is_AssignStmt:
//             check_Exp(s->u.AssignStmt.lhs, symtab);
//             //std::cout << " " << s->u.AssignStmt.astype << " "; //astype not needed since we flatten. keep it for now.
//             check_Exp(s->u.AssignStmt.rhs, symtab);
//             if (s->u.AssignStmt.lhs->kind != Exp::is_ArrayRef) {
//                 std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
//                           << ": LHS of assignment must be an array reference\n";
//                 std::exit(EXIT_FAILURE);
//             } else if (s->u.AssignStmt.lhs->coef < 0) {
//                 std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
//                           << ": LHS array reference cannot be negative\n";
//                 std::exit(EXIT_FAILURE);
//             }

// //    UNCOMMENT FOR DEBUG ONLY
// //    print_index_list(getIndices(s->u.AssignStmt.lhs));
// //    std::cout << " = ";
// //    print_index_list(getIndices(s->u.AssignStmt.rhs));
// //    std::cout << "\n";
//             if (!compare_index_lists(getIndices(s->u.AssignStmt.lhs), getIndices(s->u.AssignStmt.rhs))) {
//                 std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
//                           << ": LHS and RHS of assignment must have equal (non-summation) index sets\n";
//                 std::exit(EXIT_FAILURE);
//             }

// //    tamm_string_array lhs_aref = collectArrayRefs(s->u.AssignStmt.lhs);
// //    tamm_string_array rhs_arefs = collectArrayRefs(s->u.AssignStmt.rhs);
// //    if (exists_index(rhs_arefs->list,rhs_arefs->length,lhs_aref->list[0])){
// //        std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno << ": array " << lhs_aref->list[0] << " cannot be assigned after being previously referenced\n";
// //        std::exit(EXIT_FAILURE);
// //    }
//             break;
//         default:
//             std::cerr << "Not an Assignment Statement!\n";
//             std::exit(EXIT_FAILURE);
//     }
}


void check_range(const Identifier* const range_var, SymbolTable* const context) {

    const std::string range_name = range_var->name;
        /// RANGE VARIABLE MAY or MAY NOT BE DEFINED
        //   if (context->get(range_name) == nullptr) {
        //         std::string range_var_error = "Range variable " + range_name + " is not defined";
        //         Error(idecl->line, idecl->range_id->position, range_var_error);
        //    }
    const std::vector<std::string> allowed_ranges{"O", "V", "N"};
    if (std::find(allowed_ranges.begin(), allowed_ranges.end(), range_name) == allowed_ranges.end()) {
        std::string range_error = "Range " + range_name + " is not supported. Can only be one of O, V, N";
        Error(range_var->line, range_var->position, range_error);
    }
}

void check_Declaration(Declaration* const declaration, SymbolTable* const context) {
if (RangeDeclaration* const rdecl = dynamic_cast<RangeDeclaration*>(declaration)) {
           const std::string range_var = rdecl->name->name;
           if (context->get(range_var) != nullptr) {
                std::string range_decl_error = "Range " + range_var + " is already defined";
                Error(rdecl->line, rdecl->name->position, range_decl_error);
           }
           assert(rdecl->value > 0); /// This should be fixed in ast builder.
           context->put(range_var, new Entry(new RangeType(rdecl->value)));
}

else if (IndexDeclaration* const idecl = dynamic_cast<IndexDeclaration*>(declaration)) {
           const std::string index_name = idecl->index_name->name;
           const std::string range_name = idecl->range_id->name;
           if (context->get(index_name) != nullptr) {
                std::string index_decl_error = "Index " + index_name + " is already defined";
                Error(idecl->line, idecl->index_name->position, index_decl_error);
           }

           check_range(idecl->range_id, context);
           context->put(index_name, new Entry(new IndexType(idecl->range_id)));
}
else if (ArrayDeclaration* const adecl = dynamic_cast<ArrayDeclaration*>(declaration)) {

        const std::string tensor_name = adecl->tensor_name->name;
        if (context->get(tensor_name) != nullptr){
            std::string array_decl_error = "Tensor " + tensor_name + " is already defined";
            Error(adecl->line, adecl->tensor_name->position, array_decl_error);
        }

        for (auto &upper: adecl->upper_indices) check_range(upper, context);
        for (auto &lower: adecl->lower_indices) check_range(lower, context);

        context->put(tensor_name, new Entry(new TensorType(adecl->upper_indices,adecl->lower_indices)));
    }
}




// void verifyArrayRefName(SymbolTable &symtab, tamm_string name, int line_no) {
//     if (symtab.find(name) == symtab.end()) {
//         std::cerr << "Error at line " << line_no << ": array " << name << " is not defined\n";
//         std::exit(EXIT_FAILURE);
//     }
// }

// void verifyIndexRef(SymbolTable &symtab, tamm_string name, int line_no) {
//     if (symtab.find(name) == symtab.end()) {
//         std::cerr << "Error at line " << line_no << ": index " << name << " is not defined\n";
//         std::exit(EXIT_FAILURE);
//     }
// }

// void verifyArrayRef(SymbolTable &symtab, tamm_string name, tamm_string *inds, int len, int line_no) {
//     verifyArrayRefName(symtab, name, line_no);
//     for (int i = 0; i < len; i++) verifyIndexRef(symtab, inds[i], line_no);
// }


// void check_Exp(Exp *exp, SymbolTable &symtab) {
//     tamm_string_array inames;
//     ExpList *el = nullptr;
//     int clno = exp->lineno;
//     switch (exp->kind) {
//         case Exp::is_Parenth: {
//             check_Exp(exp->u.Parenth.exp, symtab);
//         }
//             break;
//         case Exp::is_NumConst: {
//         }
//             //std::cout << exp->u.NumConst.value << " ";
//             break;
//         case Exp::is_ArrayRef: {
//             verifyArrayRef(symtab, exp->u.Array.name, exp->u.Array.indices, exp->u.Array.length, clno);
//             inames = getIndices(exp);
//             tamm_string_array all_ind1 = inames;
//             tamm_string_array rnames;

//             for (auto i1: all_ind1)
//                 rnames.push_back(symtab[i1]);

//             tamm_string_array rnamesarr = rnames;
//             tamm_string ulranges = symtab[exp->u.Array.name];
//             tamm_string_array ulr = stringToList(ulranges);

//             if (!check_array_usage(ulr, rnamesarr)) {
//                 std::cerr << "Error at line " << clno << ": array reference " << exp->u.Array.name << "["
//                           << combine_indices(all_ind1) << "]"
//                           << " must have index structure of " << exp->u.Array.name << "[" << combine_indices(ulr)
//                           << "]\n";
//                 std::exit(EXIT_FAILURE);
//             }
//             //Check for repetitive indices in an array reference
//             tamm_string_array uind1;

//             for (auto i1: all_ind1) {
//                 if (!exists_index(uind1, i1))
//                     uind1.push_back(i1);
//             }

//             tamm_string_array up_ind(exp->u.Array.length);
//             for (int i = 0; i < exp->u.Array.length; i++)
//                 up_ind[i] = exp->u.Array.indices[i];

//             for (auto i1:uind1) {
//                 if (count_index(all_ind1, i1) > 1) {
//                     std::cerr << "Error at line " << clno << ": repetitive index " << i1 << " in array reference "
//                               << exp->u.Array.name << "[" << combine_indices(up_ind) << "]\n";
//                     std::exit(EXIT_FAILURE);
//                 }
//             }
//         }
//             break;
//         case Exp::is_Addition: {
//             check_ExpList(exp->u.Addition.subexps, symtab);
//             inames = getIndices(exp);
//             el = exp->u.Addition.subexps;
//             while (el != nullptr) {
//                 tamm_string_array op_inames = getIndices(el->head);
//                 if (!compare_index_lists(inames, op_inames)) {
//                     std::cerr << "Error at line " << clno
//                               << ": subexpressions of an addition must have equal index sets\n";
//                     std::exit(EXIT_FAILURE);
//                 }

//                 el = el->tail;
//             }
//             break;
//             case Exp::is_Multiplication:
//                 check_ExpList(exp->u.Multiplication.subexps, symtab);
//             el = exp->u.Multiplication.subexps;
//             tamm_string_array all_ind;

//             while (el != nullptr) {
//                 tamm_string_array se = getIndices(el->head);
//                 for (auto i: se)
//                     all_ind.push_back(i);
//                 el = el->tail;
//             }

//             tamm_string_array uind;
//             for (auto i: all_ind) {
//                 if (!exists_index(uind, i))
//                     uind.push_back(i);
//             }

//             for (auto i:uind) {
//                 if (count_index(all_ind, i) > 2) {
//                     std::cerr << "Error at line " << clno << ": summation index " << i <<
//                               " must occur exactly twice in a multiplication\n";
//                     std::exit(EXIT_FAILURE);
//                 }
//             }
//         }
//             break;
//         default: {
//             std::cerr << "Not a valid Expression!\n";
//             std::exit(EXIT_FAILURE);
//         }
//     }
// }

// //get non-summation indices only
// tamm_string_array getIndices(Exp *exp) {
//     ExpList *el = nullptr;
//     tamm_string_array p;
//     switch (exp->kind) {
//         case Exp::is_Parenth: {
//             return getIndices(exp->u.Parenth.exp);
//         }
//         case Exp::is_NumConst: {
//             return p;
//         }
//         case Exp::is_ArrayRef: {
//             tamm_string_array up_ind(exp->u.Array.length);
//             for (int i = 0; i < exp->u.Array.length; i++)
//                 up_ind[i] = strdup(exp->u.Array.indices[i]);
//             return up_ind;
//         }
//         case Exp::is_Addition: {
//             return getIndices(exp->u.Addition.subexps->head);
//         }
//         case Exp::is_Multiplication: {
//             el = exp->u.Multiplication.subexps;
//             tamm_string_array all_ind;
//             while (el != nullptr) {
//                 tamm_string_array se = getIndices(el->head);
//                 for (auto i: se) {
//                     all_ind.push_back(i);
//                 }
//                 el = el->tail;
//             }

//             tamm_string_array uind;
//             for (auto i: all_ind) {
//                 if (count_index(all_ind, i) == 1)
//                     uind.push_back(i);
//             }

//             tamm_string_array uniq_ind;
//             for (auto i: uind) uniq_ind.push_back(strdup(i));
//             return uniq_ind;
//         }

//         default: {
//             std::cerr << "Not a valid Expression!\n";
//             std::exit(EXIT_FAILURE);
//         }
//     }
// }


// void print_ExpList(ExpList *expList, tamm_string am) {
//     ExpList *elist = expList;
//     while (elist != nullptr) {
//         print_Exp(elist->head);
//         elist = elist->tail;
//         if (elist != nullptr) std::cout << am << " ";
//     }
// }


// void print_Exp(Exp *exp) {
//     switch (exp->kind) {
//         case Exp::is_Parenth:
//             print_Exp(exp->u.Parenth.exp);
//             break;
//         case Exp::is_NumConst:
//             std::cout << exp->u.NumConst.value << " ";
//             break;
//         case Exp::is_ArrayRef: {
//             tamm_string_array up_ind(exp->u.Array.length);
//             for (int i = 0; i < exp->u.Array.length; i++)
//                 up_ind[i] = exp->u.Array.indices[i];
//             std::cout << exp->u.Array.name << "[" << combine_indices(up_ind) << "] ";
//             break;
//         }
//         case Exp::is_Addition:
//             print_ExpList(exp->u.Addition.subexps, "+");
//             break;
//         case Exp::is_Multiplication:
//             print_ExpList(exp->u.Multiplication.subexps, "*");
//             break;
//         default:
//             std::cerr << "Not a valid Expression!\n";
//             std::exit(EXIT_FAILURE);
//     }
// }


// //get all indices only once
// tamm_string_array getUniqIndices(Exp *exp) {
//     ExpList *el = nullptr;
//     tamm_string_array p;
//     switch (exp->kind) {
//         case Exp::is_Parenth: {
//             return getUniqIndices(exp->u.Parenth.exp);
//         }

//         case Exp::is_NumConst: {
//             return p;
//         }

//         case Exp::is_ArrayRef: {
//             tamm_string_array up_ind(exp->u.Array.length);
//             for (int i = 0; i < exp->u.Array.length; i++)
//                 up_ind[i] = strdup(exp->u.Array.indices[i]);
//             return up_ind;
//         }

//         case Exp::is_Addition: {
//             return getUniqIndices(exp->u.Addition.subexps->head);
//         }

//         case Exp::is_Multiplication: {
//             el = exp->u.Multiplication.subexps;
//             tamm_string_array all_ind;
//             while (el != nullptr) {
//                 tamm_string_array se = getUniqIndices(el->head);
//                 for (auto elem: se) {
//                     all_ind.push_back(elem);
//                 }
//                 el = el->tail;
//             }

//             tamm_string_array uind;
//             for (auto i: all_ind) {
//                 if (!exists_index(uind, i))
//                     uind.push_back(i);
//             }

//             tamm_string_array uniq_ind;
//             for (auto i: uind) uniq_ind.push_back(strdup(i));
//             return uniq_ind;
//         }

//         default: {
//             std::cerr << "Not a valid Expression!\n";
//             std::exit(EXIT_FAILURE);
//         }
//     }
// }

}