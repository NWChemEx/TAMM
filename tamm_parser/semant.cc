#include "semant.h"

void check_ast(TranslationUnit *root, SymbolTable &symtab) {
    CompoundElemList *celist = root->celist;
    while (celist != nullptr) {
        check_CompoundElem(celist->head, symtab);
        celist = celist->tail;
    }
}

void check_CompoundElem(CompoundElem *celem, SymbolTable &symtab) {
    ElemList *elist = celem->elist;
    while (elist != nullptr) {
        check_Elem(elist->head, symtab);
        elist = elist->tail;
    }
}

void check_Elem(Elem *elem, SymbolTable &symtab) {
    Elem *e = elem;
    if (e == nullptr) return;

    switch (e->kind) {
        case Elem::is_DeclList:
            check_DeclList(elem->u.d, symtab);
            break;
        case Elem::is_Statement:
            check_Stmt(e->u.s, symtab);
            break;
        default:
            std::cerr << "Not a Declaration or Statement!\n";
            std::exit(EXIT_FAILURE);
    }
}

void check_DeclList(DeclList *decllist, SymbolTable &symtab) {
    DeclList *dl = decllist;
    while (dl != nullptr) {
        check_Decl(dl->head, symtab);
        dl = dl->tail;
    }
}

void verifyVarDecl(SymbolTable &symtab, tamm_string name, int line_no) {
    if (symtab.find(name) != symtab.end()) {
        std::cerr << "Error at line " << line_no << ": " << name << " is already defined\n";
        std::exit(EXIT_FAILURE);
    }
}

void verifyRangeRef(SymbolTable &symtab, tamm_string name, int line_no) {
//    if (symtab.find(name) == symtab.end()){
//        std::cerr << "Error at line " << line_no << ":range variable " << name << " is not defined\n";
//        std::exit(EXIT_FAILURE);
//    }
    const int rno = 3;
    tamm_string_array ranges = {"O", "V", "N"};
    if (!exists_index(ranges, name)) {
        std::cerr << "Error at line " << line_no << ": range " << name << " is not supported. " <<
                  "Can only be one of " << combine_indices(ranges) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_Decl(Decl *d, SymbolTable &symtab) {
    switch (d->kind) {
        case Decl::is_RangeDecl: {
            verifyVarDecl(symtab, d->u.RangeDecl.name, d->lineno);
            if (d->u.RangeDecl.value <= 0) {
                std::cerr << "Error at line " << d->lineno << ": " << d->u.RangeDecl.value
                          << " is not a positive integer\n";
                std::exit(EXIT_FAILURE);
            }
            symtab.insert(SymbolTable::value_type(
                    std::string(d->u.RangeDecl.name), constcharToChar((std::to_string(d->u.RangeDecl.value)).c_str())));
        }
            break;
        case Decl::is_IndexDecl: {
            verifyVarDecl(symtab, d->u.IndexDecl.name, d->lineno);
            verifyRangeRef(symtab, d->u.IndexDecl.rangeID, d->lineno);
            symtab.insert(SymbolTable::value_type(std::string(d->u.IndexDecl.name), (d->u.IndexDecl.rangeID)));
        }
            break;
        case Decl::is_ArrayDecl: {
            tamm_string_array up_ind(d->u.ArrayDecl.ulen);
            for (int i = 0; i < d->u.ArrayDecl.ulen; i++)
                up_ind[i] = d->u.ArrayDecl.upperIndices[i];

            tamm_string_array lo_ind(d->u.ArrayDecl.llen);
            for (int i = 0; i < d->u.ArrayDecl.llen; i++)
                lo_ind[i] = d->u.ArrayDecl.lowerIndices[i];

            verifyVarDecl(symtab, d->u.ArrayDecl.name, d->lineno);
            tamm_string comb_index_list = combine_indexLists(up_ind, lo_ind);
            //std::cout << d->u.ArrayDecl.name << " -> " << comb_index_list << std::endl;
            int i = 0;
            tamm_string *ind_list = d->u.ArrayDecl.upperIndices;
            for (i = 0; i < d->u.ArrayDecl.ulen; i++) verifyRangeRef(symtab, ind_list[i], d->lineno);
            ind_list = d->u.ArrayDecl.lowerIndices;
            for (i = 0; i < d->u.ArrayDecl.llen; i++) verifyRangeRef(symtab, ind_list[i], d->lineno);

            symtab.insert(SymbolTable::value_type(std::string(d->u.ArrayDecl.name), (comb_index_list)));
        }
            break;
        default: {
            std::cerr << "Not a valid Declaration!\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

void check_Stmt(Stmt *s, SymbolTable &symtab) {
    switch (s->kind) {
        case Stmt::is_AssignStmt:
            check_Exp(s->u.AssignStmt.lhs, symtab);
            //std::cout << " " << s->u.AssignStmt.astype << " "; //astype not needed since we flatten. keep it for now.
            check_Exp(s->u.AssignStmt.rhs, symtab);
            if (s->u.AssignStmt.lhs->kind != Exp::is_ArrayRef) {
                std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
                          << ": LHS of assignment must be an array reference\n";
                std::exit(EXIT_FAILURE);
            } else if (s->u.AssignStmt.lhs->coef < 0) {
                std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
                          << ": LHS array reference cannot be negative\n";
                std::exit(EXIT_FAILURE);
            }

//    UNCOMMENT FOR DEBUG ONLY
//    print_index_list(getIndices(s->u.AssignStmt.lhs));
//    std::cout << " = ";
//    print_index_list(getIndices(s->u.AssignStmt.rhs));
//    std::cout << "\n";
            if (!compare_index_lists(getIndices(s->u.AssignStmt.lhs), getIndices(s->u.AssignStmt.rhs))) {
                std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno
                          << ": LHS and RHS of assignment must have equal (non-summation) index sets\n";
                std::exit(EXIT_FAILURE);
            }

//    tamm_string_array lhs_aref = collectArrayRefs(s->u.AssignStmt.lhs);
//    tamm_string_array rhs_arefs = collectArrayRefs(s->u.AssignStmt.rhs);
//    if (exists_index(rhs_arefs->list,rhs_arefs->length,lhs_aref->list[0])){
//        std::cerr << "Error at line " << s->u.AssignStmt.lhs->lineno << ": array " << lhs_aref->list[0] << " cannot be assigned after being previously referenced\n";
//        std::exit(EXIT_FAILURE);
//    }
            break;
        default:
            std::cerr << "Not an Assignment Statement!\n";
            std::exit(EXIT_FAILURE);
    }
}


void check_ExpList(ExpList *expList, SymbolTable &symtab) {
    ExpList *elist = expList;
    while (elist != nullptr) {
        check_Exp(elist->head, symtab);
        elist = elist->tail;
    }
}


void verifyArrayRefName(SymbolTable &symtab, tamm_string name, int line_no) {
    if (symtab.find(name) == symtab.end()) {
        std::cerr << "Error at line " << line_no << ": array " << name << " is not defined\n";
        std::exit(EXIT_FAILURE);
    }
}

void verifyIndexRef(SymbolTable &symtab, tamm_string name, int line_no) {
    if (symtab.find(name) == symtab.end()) {
        std::cerr << "Error at line " << line_no << ": index " << name << " is not defined\n";
        std::exit(EXIT_FAILURE);
    }
}

void verifyArrayRef(SymbolTable &symtab, tamm_string name, tamm_string *inds, int len, int line_no) {
    verifyArrayRefName(symtab, name, line_no);
    int i = 0;
    for (i = 0; i < len; i++) verifyIndexRef(symtab, inds[i], line_no);
}


void check_Exp(Exp *exp, SymbolTable &symtab) {
    tamm_string_array inames;
    ExpList *el = nullptr;
    int clno = exp->lineno;
    switch (exp->kind) {
        case Exp::is_Parenth: {
            check_Exp(exp->u.Parenth.exp, symtab);
        }
            break;
        case Exp::is_NumConst: {
        }
            //std::cout << exp->u.NumConst.value << " ";
            break;
        case Exp::is_ArrayRef: {
            verifyArrayRef(symtab, exp->u.Array.name, exp->u.Array.indices, exp->u.Array.length, clno);
            inames = getIndices(exp);
            tamm_string_array all_ind1 = inames;
            tamm_string_array rnames;

            for (auto i1: all_ind1)
                rnames.push_back(symtab[i1]);

            tamm_string_array rnamesarr = rnames;
            tamm_string ulranges = symtab[exp->u.Array.name];
            tamm_string_array ulr = stringToList(ulranges);

            if (!check_array_usage(ulr, rnamesarr)) {
                std::cerr << "Error at line " << clno << ": array reference " << exp->u.Array.name << "["
                          << combine_indices(all_ind1) << "]"
                          << " must have index structure of " << exp->u.Array.name << "[" << combine_indices(ulr)
                          << "]\n";
                std::exit(EXIT_FAILURE);
            }
            //Check for repetitive indices in an array reference
            tamm_string_array uind1;

            for (auto i1: all_ind1) {
                if (!exists_index(uind1, i1))
                    uind1.push_back(i1);
            }

            tamm_string_array up_ind(exp->u.Array.length);
            for (int i = 0; i < exp->u.Array.length; i++)
                up_ind[i] = exp->u.Array.indices[i];

            for (auto i1:uind1) {
                if (count_index(all_ind1, i1) > 1) {
                    std::cerr << "Error at line " << clno << ": repetitive index " << i1 << " in array reference "
                              << exp->u.Array.name << "[" << combine_indices(up_ind) << "]\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
            break;
        case Exp::is_Addition: {
            check_ExpList(exp->u.Addition.subexps, symtab);
            inames = getIndices(exp);
            el = exp->u.Addition.subexps;
            while (el != nullptr) {
                tamm_string_array op_inames = getIndices(el->head);
                if (!compare_index_lists(inames, op_inames)) {
                    std::cerr << "Error at line " << clno
                              << ": subexpressions of an addition must have equal index sets\n";
                    std::exit(EXIT_FAILURE);
                }

                el = el->tail;
            }
            break;
            case Exp::is_Multiplication:
                check_ExpList(exp->u.Multiplication.subexps, symtab);
            el = exp->u.Multiplication.subexps;
            tamm_string_array all_ind;

            while (el != nullptr) {
                tamm_string_array se = getIndices(el->head);
                for (auto i: se)
                    all_ind.push_back(i);
                el = el->tail;
            }

            tamm_string_array uind;
            for (auto i: all_ind) {
                if (!exists_index(uind, i))
                    uind.push_back(i);
            }

            for (auto i:uind) {
                if (count_index(all_ind, i) > 2) {
                    std::cerr << "Error at line " << clno << ": summation index " << i <<
                              " must occur exactly twice in a multiplication\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }
            break;
        default: {
            std::cerr << "Not a valid Expression!\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

//get non-summation indices only
tamm_string_array getIndices(Exp *exp) {
    ExpList *el = nullptr;
    tamm_string_array p;
    switch (exp->kind) {
        case Exp::is_Parenth: {
            return getIndices(exp->u.Parenth.exp);
        }
        case Exp::is_NumConst: {
            return p;
        }
        case Exp::is_ArrayRef: {
            tamm_string_array up_ind(exp->u.Array.length);
            for (int i = 0; i < exp->u.Array.length; i++)
                up_ind[i] = strdup(exp->u.Array.indices[i]);
            return up_ind;
        }
        case Exp::is_Addition: {
            return getIndices(exp->u.Addition.subexps->head);
        }
        case Exp::is_Multiplication: {
            el = exp->u.Multiplication.subexps;
            tamm_string_array all_ind;
            while (el != nullptr) {
                tamm_string_array se = getIndices(el->head);
                for (auto i: se) {
                    all_ind.push_back(i);
                }
                el = el->tail;
            }

            tamm_string_array uind;
            for (auto i: all_ind) {
                if (count_index(all_ind, i) == 1)
                    uind.push_back(i);
            }

            tamm_string_array uniq_ind;
            for (auto i: uind) uniq_ind.push_back(strdup(i));
            return uniq_ind;
        }

        default: {
            std::cerr << "Not a valid Expression!\n";
            std::exit(EXIT_FAILURE);
        }
    }
}


void print_ExpList(ExpList *expList, tamm_string am) {
    ExpList *elist = expList;
    while (elist != nullptr) {
        print_Exp(elist->head);
        elist = elist->tail;
        if (elist != nullptr) std::cout << am << " ";
    }
}


void print_Exp(Exp *exp) {
    switch (exp->kind) {
        case Exp::is_Parenth:
            print_Exp(exp->u.Parenth.exp);
            break;
        case Exp::is_NumConst:
            std::cout << exp->u.NumConst.value << " ";
            break;
        case Exp::is_ArrayRef: {
            tamm_string_array up_ind(exp->u.Array.length);
            for (int i = 0; i < exp->u.Array.length; i++)
                up_ind[i] = exp->u.Array.indices[i];
            std::cout << exp->u.Array.name << "[" << combine_indices(up_ind) << "] ";
            break;
        }
        case Exp::is_Addition:
            print_ExpList(exp->u.Addition.subexps, "+");
            break;
        case Exp::is_Multiplication:
            print_ExpList(exp->u.Multiplication.subexps, "*");
            break;
        default:
            std::cerr << "Not a valid Expression!\n";
            std::exit(EXIT_FAILURE);
    }
}


//get all indices only once
tamm_string_array getUniqIndices(Exp *exp) {
    ExpList *el = nullptr;
    tamm_string_array p;
    switch (exp->kind) {
        case Exp::is_Parenth: {
            return getUniqIndices(exp->u.Parenth.exp);
        }

        case Exp::is_NumConst: {
            return p;
        }

        case Exp::is_ArrayRef: {
            tamm_string_array up_ind(exp->u.Array.length);
            for (int i = 0; i < exp->u.Array.length; i++)
                up_ind[i] = strdup(exp->u.Array.indices[i]);
            return up_ind;
        }

        case Exp::is_Addition: {
            return getUniqIndices(exp->u.Addition.subexps->head);
        }

        case Exp::is_Multiplication: {
            el = exp->u.Multiplication.subexps;
            tamm_string_array all_ind;
            while (el != nullptr) {
                tamm_string_array se = getUniqIndices(el->head);
                for (auto elem: se) {
                    all_ind.push_back(elem);
                }
                el = el->tail;
            }

            tamm_string_array uind;
            for (auto i: all_ind) {
                if (!exists_index(uind, i))
                    uind.push_back(i);
            }

            tamm_string_array uniq_ind;
            for (auto i: uind) uniq_ind.push_back(strdup(i));
            return uniq_ind;
        }

        default: {
            std::cerr << "Not a valid Expression!\n";
            std::exit(EXIT_FAILURE);
        }
    }
}


//tamm_string_array collectArrayRefs(Exp exp) {
//    ExpList* el = nullptr;
//    tamm_string_array p = nullptr;
//    int c = 0, i = 0;
//    switch (exp->kind) {
//        case is_Parenth:
//            return collectArrayRefs(exp->u.Parenth.exp);
//            break;
//        case is_NumConst:
//            return nullptr;
//            break;
//        case is_ArrayRef:
//            p = tce_malloc(sizeof(*p));
//            p->list = tce_malloc(sizeof(string) * 1);
//            p->list[0] = exp->u.Array.name;
//            p->length = 1;
//            return p;
//            break;
//        case is_Addition:
//            c = 0, i = 0;
//            el = (exp->u.Addition.subexps);
//            string allrefs[65535];
//            while (el != nullptr) {
//                tamm_string_array cref = collectArrayRefs(el->head);
//                for (i = 0; i < cref->length; i++) {
//                    allrefs[c] = cref->list[i];
//                    c++;
//                }
//                el = el->tail;
//            }
//            i = 0;
//            p = tce_malloc(sizeof(*p));
//            p->list = tce_malloc(sizeof(string) * c);
//            p->length = c;
//            for (i = 0; i < c; i++) {
//                p->list[i] = allrefs[i];
//            }
//            return p;
//            break;
//        case is_Multiplication:
//            c = 0, i = 0;
//            el = (exp->u.Multiplication.subexps);
//            string allrefs1[65535];
//            while (el != nullptr) {
//                tamm_string_array cref = collectArrayRefs(el->head);
//                if (cref != nullptr) {
//                    for (i = 0; i < cref->length; i++) {
//                        allrefs1[c] = cref->list[i];
//                        c++;
//                    }
//                }
//                el = el->tail;
//            }
//            i = 0;
//            p = tce_malloc(sizeof(*p));
//            p->list = tce_malloc(sizeof(string) * c);
//            p->length = c;
//            for (i = 0; i < c; i++) {
//                p->list[i] = allrefs1[i];
//            }
//            return p;
//            break;
//        default:
//            std::cerr << "Not a valid Expression!\n";
//            std::exit(EXIT_FAILURE);
//    }
//}
//
