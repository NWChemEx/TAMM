#include "semant.h"

void check_ast(TranslationUnit* root, SymbolTable symtab) {
    CompoundElemList* celist = root->celist;
    while (celist != nullptr) {
        check_CompoundElem(celist->head, symtab);
        celist = celist->tail;
    }
    celist = nullptr;
}

void check_CompoundElem(CompoundElem celem, SymbolTable symtab) {
    ElemList *elist = celem->elist;
    while (elist != nullptr) {
        check_Elem(elist->head, symtab);
        elist = elist->tail;
    }
    elist = nullptr;
}

void check_Elem(Elem elem, SymbolTable symtab) {
    Elem e = elem;
    if (e == nullptr) return;

    switch (e->kind) {
        case Elem_::is_DeclList:
            check_DeclList(elem->u.d, symtab);
            break;
        case Elem_::is_Statement:
            check_Stmt(e->u.s, symtab);
            break;
        default:
            fprintf(stderr, "Not a Declaration or Statement!\n");
            exit(0);
    }
}

void check_DeclList(DeclList* decllist, SymbolTable symtab) {
    DeclList* dl = decllist;
    while (dl != nullptr) {
        check_Decl(dl->head, symtab);
        dl = dl->tail;
    }
}

void verifyVarDecl(SymbolTable symtab, tamm_string name, int line_no) {
    if (ST_contains(symtab, name)) {
        fprintf(stderr, "Error at line %d: %s is already defined\n", line_no, name);
        exit(2);
    }
}

void verifyRangeRef(SymbolTable symtab, tamm_string name, int line_no) {
//    if (!ST_contains(symtab,name)){
//        fprintf(stderr,"Error at line %d: range variable %s is not defined\n", line_no, name);
//        //exit(2);
//    }
    const int rno = 3;
    tamm_string ranges[] = {"O", "V", "N"};
    if (!exists_index(ranges, rno, name)) {
        fprintf(stderr, "Error at line %d: range %s is not supported. Can only be one of %s\n", line_no, name,
                combine_indices(ranges, rno));
        exit(2);
    }
}

void check_Decl(Decl d, SymbolTable symtab) {
    switch (d->kind) {
      case Decl_::is_RangeDecl: {
        verifyVarDecl(symtab, d->u.RangeDecl.name, d->lineno);
        if (d->u.RangeDecl.value <= 0) {
          fprintf(stderr, "Error at line %d: %d is not a positive integer\n", d->lineno, d->u.RangeDecl.value);
          exit(2);
        }
        ST_insert(symtab, d->u.RangeDecl.name, int_str(d->u.RangeDecl.value));
      }
        break;
      case Decl_::is_IndexDecl: {
        verifyVarDecl(symtab, d->u.IndexDecl.name, d->lineno);
        verifyRangeRef(symtab, d->u.IndexDecl.rangeID, d->lineno);
        ST_insert(symtab, d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
      }
        break;
      case Decl_::is_ArrayDecl: {
        verifyVarDecl(symtab, d->u.ArrayDecl.name, d->lineno);
        tamm_string comb_index_list = combine_indexLists(d->u.ArrayDecl.upperIndices, d->u.ArrayDecl.ulen,
                                                         d->u.ArrayDecl.lowerIndices, d->u.ArrayDecl.llen);
        //printf("%s -> %s\n", d->u.ArrayDecl.name, comb_index_list);
        int i = 0;
        tamm_string *ind_list = d->u.ArrayDecl.upperIndices;
        for (i = 0; i < d->u.ArrayDecl.ulen; i++) verifyRangeRef(symtab, ind_list[i], d->lineno);
        ind_list = d->u.ArrayDecl.lowerIndices;
        for (i = 0; i < d->u.ArrayDecl.llen; i++) verifyRangeRef(symtab, ind_list[i], d->lineno);

        ST_insert(symtab, d->u.ArrayDecl.name, comb_index_list);
      }
        break;
      default:
      {
        fprintf(stderr, "Not a valid Declaration!\n");
        exit(0);
      }
    }
}

void check_Stmt(Stmt s, SymbolTable symtab) {
    switch (s->kind) {
        case Stmt_::is_AssignStmt:
            check_Exp(s->u.AssignStmt.lhs, symtab);
            //printf(" %s ", s->u.AssignStmt.astype); //astype not needed since we flatten. keep it for now.
            check_Exp(s->u.AssignStmt.rhs, symtab);
            if (s->u.AssignStmt.lhs->kind != Exp_::is_ArrayRef) {
                fprintf(stderr, "Error at line %d: LHS of assignment must be an array reference\n",
                        s->u.AssignStmt.lhs->lineno);
                exit(2);
            }
            else if (s->u.AssignStmt.lhs->coef < 0) {
                fprintf(stderr, "Error at line %d: LHS array reference cannot be negative\n",
                        s->u.AssignStmt.lhs->lineno);
                exit(2);
            }

//    UNCOMMENT FOR DEBUG ONLY
//    print_index_list(getIndices(s->u.AssignStmt.lhs));
//    printf(" = ");
//    print_index_list(getIndices(s->u.AssignStmt.rhs));
//    printf("\n");
            if (!compare_index_lists(getIndices(s->u.AssignStmt.lhs), getIndices(s->u.AssignStmt.rhs))) {
                fprintf(stderr,
                        "Error at line %d: LHS and RHS of assignment must have equal (non-summation) index sets\n",
                        s->u.AssignStmt.lhs->lineno);
                exit(2);
            }

//    tce_string_array lhs_aref = collectArrayRefs(s->u.AssignStmt.lhs);
//    tce_string_array rhs_arefs = collectArrayRefs(s->u.AssignStmt.rhs);
//    if (exists_index(rhs_arefs->list,rhs_arefs->length,lhs_aref->list[0])){
//        fprintf(stderr,"Error at line %d: array %s cannot be assigned after being previously referenced\n",
//        		    s->u.AssignStmt.lhs->lineno, lhs_aref->list[0]);
//        exit(2);
//    }
            break;
        default:
            fprintf(stderr, "Not an Assignment Statement!\n");
            exit(0);
    }
}


void check_ExpList(ExpList* expList, SymbolTable symtab) {
    ExpList* elist = expList;
    while (elist != nullptr) {
        check_Exp(elist->head, symtab);
        elist = elist->tail;
    }
    elist = nullptr;
}


void verifyArrayRefName(SymbolTable symtab, tamm_string name, int line_no) {
    if (!ST_contains(symtab, name)) {
        fprintf(stderr, "Error at line %d: array %s is not defined\n", line_no, name);
        exit(2);
    }
}

void verifyIndexRef(SymbolTable symtab, tamm_string name, int line_no) {
    if (!ST_contains(symtab, name)) {
        fprintf(stderr, "Error at line %d: index %s is not defined\n", line_no, name);
        exit(2);
    }
}

void verifyArrayRef(SymbolTable symtab, tamm_string name, tamm_string *inds, int len, int line_no) {
    verifyArrayRefName(symtab, name, line_no);
    int i = 0;
    for (i = 0; i < len; i++) verifyIndexRef(symtab, inds[i], line_no);
}


void check_Exp(Exp exp, SymbolTable symtab) {
    tce_string_array inames = nullptr;
    ExpList* el = nullptr;
    int clno = exp->lineno;
    switch (exp->kind) {
        case Exp_::is_Parenth: {
          check_Exp(exp->u.Parenth.exp, symtab);
        }
            break;
        case Exp_::is_NumConst: {}
            //printf("%f ",exp->u.NumConst.value);
            break;
        case Exp_::is_ArrayRef: {
          verifyArrayRef(symtab, exp->u.Array.name, exp->u.Array.indices, exp->u.Array.length, clno);
          inames = getIndices(exp);
          int tot_len1 = inames->length;
          tamm_string *all_ind1 = inames->list;
          int i1 = 0, ui1 = 0;
          tamm_string *rnames = (tamm_string *) tce_malloc(sizeof(tamm_string) * tot_len1);

          for (i1 = 0; i1 < tot_len1; i1++) {
            rnames[i1] = ST_get(symtab, all_ind1[i1]);
          }

          tce_string_array rnamesarr = (tce_string_array) tce_malloc(sizeof(*rnamesarr));
          rnamesarr->list = rnames;
          rnamesarr->length = tot_len1;
          tamm_string ulranges = ST_get(symtab, exp->u.Array.name);
          tce_string_array ulr = (tce_string_array) stringToList(ulranges);

          if (!check_array_usage(ulr, rnamesarr)) {
            fprintf(stderr, "Error at line %d: array reference %s[%s] must have index structure of %s[%s]\n", clno,
                    exp->u.Array.name, combine_indices(all_ind1, tot_len1), exp->u.Array.name,
                    combine_indices(ulr->list, ulr->length));
            exit(2);
          }
          //Check for repetitive indices in an array reference
          tamm_string *uind1 = (tamm_string*) tce_malloc(sizeof(tamm_string) * tot_len1);

          i1 = 0, ui1 = 0;
          for (i1 = 0; i1 < tot_len1; i1++) {
            if (!exists_index(uind1, ui1, all_ind1[i1])) {
              uind1[ui1] = all_ind1[i1];
              ui1++;
            }
          }

          for (i1 = 0; i1 < ui1; i1++) {
            if (count_index(all_ind1, tot_len1, uind1[i1]) > 1) {
              fprintf(stderr, "Error at line %d: repetitive index %s in array reference %s[%s]\n",
                      clno, uind1[i1], exp->u.Array.name,
                      combine_indices(exp->u.Array.indices, exp->u.Array.length));
              exit(2);
            }
          }
        }
            break;
        case Exp_::is_Addition: {
          check_ExpList(exp->u.Addition.subexps, symtab);
          inames = getIndices(exp);
          el = exp->u.Addition.subexps;
          while (el != nullptr) {
            tce_string_array op_inames = getIndices(el->head);
            if (!compare_index_lists(inames, op_inames)) {
              fprintf(stderr, "Error at line %d: subexpressions of an addition must have equal index sets\n",
                      clno);
              exit(2);
            }
            op_inames = nullptr;
            el = el->tail;
          }
          break;
          case Exp_::is_Multiplication:check_ExpList(exp->u.Multiplication.subexps, symtab);

          el = exp->u.Multiplication.subexps;
          int tot_len = 0;
          while (el != nullptr) {
            //print_Exp(el->head);
            tce_string_array se = getIndices(el->head);
            if (se != nullptr) tot_len += se->length;
            se = nullptr;
            el = el->tail;
          }

          el = exp->u.Multiplication.subexps;
          tamm_string *all_ind = (tamm_string*) tce_malloc(sizeof(tamm_string) * tot_len);

          int i = 0, ui = 0;
          while (el != nullptr) {
            tce_string_array se = getIndices(el->head);
            i = 0;
            if (se != nullptr) {
              for (i = 0; i < se->length; i++) {
                all_ind[ui] = se->list[i];
                ui++;
              }
            }
            se = nullptr;
            el = el->tail;
          }
          assert(ui == tot_len);
          tamm_string *uind = (tamm_string*) tce_malloc(sizeof(tamm_string) * tot_len);

          i = 0, ui = 0;
          for (i = 0; i < tot_len; i++) {
            if (!exists_index(uind, ui, all_ind[i])) {
              uind[ui] = all_ind[i];
              ui++;
            }
          }

          for (i = 0; i < ui; i++) {
            if (count_index(all_ind, tot_len, uind[i]) > 2) {
              fprintf(stderr,
                      "Error at line %d: summation index %s must occur exactly twice in a multiplication\n", clno,
                      uind[i]);
              exit(2);
            }
          }
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Expression!\n");
          exit(0);
        }
    }
}

//get non-summation indices only
tce_string_array getIndices(Exp exp) {
    ExpList* el = nullptr;
    tce_string_array p = nullptr;
    switch (exp->kind) {
        case Exp_::is_Parenth: {
          return getIndices(exp->u.Parenth.exp);
        }
            break;
        case Exp_::is_NumConst: {
          return nullptr;
        }
            break;
        case Exp_::is_ArrayRef: {
          p = (tce_string_array) tce_malloc(sizeof(*p));
          p->list = replicate_indices(exp->u.Array.indices, exp->u.Array.length);
          p->length = exp->u.Array.length;
          return p;
        }
            break;
        case Exp_::is_Addition: {
          return getIndices(exp->u.Addition.subexps->head);
        }
            break;
        case Exp_::is_Multiplication: {
          el = exp->u.Multiplication.subexps;
          int tot_len = 0;
          while (el != nullptr) {
            //print_Exp(el->head);
            tce_string_array se = (tce_string_array) getIndices(el->head);
            if (se != nullptr) tot_len += se->length;
            se = nullptr;
            el = el->tail;
          }

          el = exp->u.Multiplication.subexps;
          tamm_string *all_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * tot_len);

          int i = 0, ui = 0;
          while (el != nullptr) {
            tce_string_array se = (tce_string_array) getIndices(el->head);
            i = 0;
            if (se != nullptr) {
              for (i = 0; i < se->length; i++) {
                all_ind[ui] = se->list[i];
                ui++;
              }
            }
            se = nullptr;
            el = el->tail;
          }
          assert(ui == tot_len);
          tamm_string *uind = (tamm_string *) tce_malloc(sizeof(tamm_string) * tot_len);

          i = 0, ui = 0;
//  	for (i=0;i<tot_len;i++){
//  		if(!exists_index(uind,ui,all_ind[i])) {
//  			uind[ui] = all_ind[i];
//  			ui++;
//  		}
//  	}

          for (i = 0; i < tot_len; i++) {
            if (count_index(all_ind, tot_len, all_ind[i]) == 1) {
              uind[ui] = all_ind[i];
              ui++;
            }
          }

          tamm_string *uniq_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * ui);
          for (i = 0; i < ui; i++) uniq_ind[i] = strdup(uind[i]);

//  	free(all_ind);
//  	all_ind = nullptr;
          //uind = nullptr;

          p = (tce_string_array) tce_malloc(sizeof(*p));
          p->list = uniq_ind;
          p->length = ui;

          return p;
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Expression!\n");
          exit(0);
        }
    }
}


void print_ExpList(ExpList* expList, tamm_string am) {
    ExpList* elist = expList;
    while (elist != nullptr) {
        print_Exp(elist->head);
        elist = elist->tail;
        if (elist != nullptr) printf("%s ", am);
    }
    elist = nullptr;
}


void print_Exp(Exp exp) {
    switch (exp->kind) {
        case Exp_::is_Parenth:
            print_Exp(exp->u.Parenth.exp);
            break;
        case Exp_::is_NumConst:
            printf("%f ", exp->u.NumConst.value);
            break;
        case Exp_::is_ArrayRef:
            printf("%s[%s] ", exp->u.Array.name, combine_indices(exp->u.Array.indices, exp->u.Array.length));
            break;
        case Exp_::is_Addition:
            print_ExpList(exp->u.Addition.subexps, "+");
            break;
        case Exp_::is_Multiplication:
            print_ExpList(exp->u.Multiplication.subexps, "*");
            break;
        default:
            fprintf(stderr, "Not a valid Expression!\n");
            exit(0);
    }
}


//get all indices only once
tce_string_array getUniqIndices(Exp exp) {
    ExpList* el = nullptr;
    tce_string_array p = nullptr;
    switch (exp->kind) {
        case Exp_::is_Parenth: {
          return getUniqIndices(exp->u.Parenth.exp);
        }
            break;
        case Exp_::is_NumConst: {
          return nullptr;
        }
            break;
        case Exp_::is_ArrayRef: {
          p = (tce_string_array) tce_malloc(sizeof(*p));
          p->list = replicate_indices(exp->u.Array.indices, exp->u.Array.length);
          p->length = exp->u.Array.length;
          return p;
        }
            break;
        case Exp_::is_Addition: {
          return getUniqIndices(exp->u.Addition.subexps->head);
        }
            break;
        case Exp_::is_Multiplication: {
          el = exp->u.Multiplication.subexps;
          int tot_len = 0;
          while (el != nullptr) {
            //print_Exp(el->head);
            tce_string_array se = (tce_string_array) getUniqIndices(el->head);
            if (se != nullptr) tot_len += se->length;
            se = nullptr;
            el = el->tail;
          }

          el = exp->u.Multiplication.subexps;
          tamm_string *all_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * tot_len);

          int i = 0, ui = 0;
          while (el != nullptr) {
            tce_string_array se = (tce_string_array) getUniqIndices(el->head);
            i = 0;
            if (se != nullptr) {
              for (i = 0; i < se->length; i++) {
                all_ind[ui] = se->list[i];
                ui++;
              }
            }
            se = nullptr;
            el = el->tail;
          }
          assert(ui == tot_len);
          tamm_string *uind = (tamm_string *) tce_malloc(sizeof(tamm_string) * tot_len);

          i = 0, ui = 0;

          for (i = 0; i < tot_len; i++) {
            if (!exists_index(uind, ui, all_ind[i])) {
              uind[ui] = all_ind[i];
              ui++;
            }
          }

          tamm_string *uniq_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * ui);
          for (i = 0; i < ui; i++) uniq_ind[i] = strdup(uind[i]);

          p = (tce_string_array) tce_malloc(sizeof(*p));
          p->list = uniq_ind;
          p->length = ui;

          return p;
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Expression!\n");
          exit(0);
        }
    }
}


//tce_string_array collectArrayRefs(Exp exp) {
//    ExpList* el = nullptr;
//    tce_string_array p = nullptr;
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
//                tce_string_array cref = collectArrayRefs(el->head);
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
//                tce_string_array cref = collectArrayRefs(el->head);
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
//            fprintf(stderr, "Not a valid Expression!\n");
//            exit(0);
//    }
//}
//
