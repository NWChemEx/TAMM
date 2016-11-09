#include "intermediate.h"
#include "semant.h"
#include "absyn.h"

//void make_Equations(Equations p) {
//    p = tce_malloc(sizeof(*p));
//    vector_init(&p->index_entries);
//    vector_init(&p->range_entries);
//    vector_init(&p->op_entries);
//    vector_init(&p->tensor_entries);
//    //return p;
//}


int op_id = 1;
typedef struct ArrayRefAlpha_ *ArrayRefAlpha;


struct ArrayRefAlpha_ {
    double alpha;
    Exp aref;
};

ArrayRefAlpha make_ArrayRefAlpha(double alpha, Exp aref) {
    ArrayRefAlpha p = (ArrayRefAlpha) tce_malloc(sizeof(*p));
    p->alpha = alpha;
    p->aref = aref;
    return p;
}

RangeEntry make_RangeEntry(tamm_string name) {
    RangeEntry p = (RangeEntry)tce_malloc(sizeof(*p));
    p->name = name;
    return p;
}

IndexEntry make_IndexEntry(tamm_string name, int range_id) {
    IndexEntry p = (IndexEntry)tce_malloc(sizeof(*p));
    p->name = name;
    p->range_id = range_id;
    return p;
}

TensorEntry make_TensorEntry(tamm_string name, int ndim, int nupper) {
    TensorEntry p = (TensorEntry)tce_malloc(sizeof(*p));
    p->name = name;
    p->ndim = ndim;
    p->nupper = nupper;
    return p;
}


AddOp make_AddOp(int tc, int ta, double alpha) {
    AddOp p = (AddOp)tce_malloc(sizeof(*p));
    p->ta = ta;
    p->tc = tc;
    p->alpha = alpha;
    return p;
}

MultOp make_MultOp(int tc, int ta, int tb, double alpha) {
    MultOp p = (MultOp)tce_malloc(sizeof(*p));
    p->ta = ta;
    p->tc = tc;
    p->tb = tb;
    p->alpha = alpha;
    return p;
}

OpEntry make_OpEntry(int op_id, OpType ot, AddOp ao, MultOp mo) {
    OpEntry p = (OpEntry)tce_malloc(sizeof(*p));
    p->op_id = op_id;
    p->optype = ot;
    p->add = ao;
    p->mult = mo;
    return p;
}

void generate_intermediate_ast(Equations *eqn, TranslationUnit* root) {
//    &eqn->index_entries;
//    &eqn->range_entries;
//    &eqn->op_entries;
//    &eqn->tensor_entries;

    std::vector<RangeEntry> &re = eqn->range_entries;
    re.push_back(make_RangeEntry("O"));
    re.push_back(make_RangeEntry("V"));
    re.push_back(make_RangeEntry("N"));

    CompoundElemList* celist = root->celist;
    while (celist != nullptr) {
        generate_intermediate_CompoundElem(eqn, celist->head);
        celist = celist->tail;
    }
    celist = nullptr;
}

void generate_intermediate_CompoundElem(Equations *eqn, CompoundElem* celem) {
    ElemList *elist = celem->elist;
    while (elist != nullptr) {
        generate_intermediate_Elem(eqn, elist->head);
        elist = elist->tail;
    }
    elist = nullptr;
}

void generate_intermediate_Elem(Equations *eqn, Elem* elem) {
    Elem* e = elem;
    if (e == nullptr) return;

    switch (e->kind) {
        case Elem::is_DeclList:
            generate_intermediate_DeclList(eqn, elem->u.d);
            break;
      case Elem::is_Statement:
            generate_intermediate_Stmt(eqn, e->u.s);
            break;
        default:
            fprintf(stderr, "Not a Declaration or Statement!\n");
            exit(0);
    }
}

void generate_intermediate_DeclList(Equations *eqn, DeclList* decllist) {
    DeclList* dl = decllist;
    while (dl != nullptr) {
        generate_intermediate_Decl(eqn, dl->head);
        dl = dl->tail;
    }
}

void generate_intermediate_Decl(Equations *eqn, Decl d) {
    switch (d->kind) {
      case Decl_::is_RangeDecl: {
        //fprintf(eqn, "range %s : %d;\n", d->u.RangeDecl.name, d->u.RangeDecl.value);
      }
            break;
        case Decl_::is_IndexDecl: {
          int rid=0;
          if (strcmp(d->u.IndexDecl.rangeID, "O") == 0) rid = 0;
          else if (strcmp(d->u.IndexDecl.rangeID, "V") == 0) rid = 1;
          else if (strcmp(d->u.IndexDecl.rangeID, "N") == 0) rid = 2;

          eqn->index_entries.push_back(make_IndexEntry(d->u.IndexDecl.name, rid));
          //fprintf(eqn, "index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
        }
            break;
        case Decl_::is_ArrayDecl: {
          TensorEntry te = make_TensorEntry(d->u.ArrayDecl.name, d->u.ArrayDecl.ulen + d->u.ArrayDecl.llen, d->u.ArrayDecl.ulen);
          int rid = 0;
          for (rid = 0; rid < d->u.ArrayDecl.ulen; rid++) {
            tamm_string range = d->u.ArrayDecl.upperIndices[rid];
            te->range_ids[rid] = 0;
            if (strcmp(range, "V") == 0) te->range_ids[rid] = 1;
            else if (strcmp(range, "N") == 0) te->range_ids[rid] = 2;
          }

          int lid = rid;
          for (rid = 0; rid < d->u.ArrayDecl.llen; rid++) {
            tamm_string range = d->u.ArrayDecl.lowerIndices[rid];
            te->range_ids[lid] = 0;
            if (strcmp(range, "V") == 0) te->range_ids[lid] = 1;
            else if (strcmp(range, "N") == 0) te->range_ids[lid] = 2;
            lid++;
          }

          eqn->tensor_entries.push_back(te);
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Declaration!\n");
          exit(0);
        }
    }
}

void generate_intermediate_Stmt(Equations *eqn, Stmt s) {
    std::vector<Exp> lhs_aref, rhs_allref;
    std::vector<ArrayRefAlpha> rhs_aref;
    double alpha = 1;
    switch (s->kind) {
      case Stmt_::is_AssignStmt: {
        alpha = 1;
        collectArrayRefs(s->u.AssignStmt.lhs, lhs_aref, &alpha);
        int i = 0;
//            for (i = 0; i < vector_count(&lhs_aref); i++) {
//                Exp e = vector_get(&lhs_aref, i);
//                printf("%s ", e->u.Array.name);
//            }
        collectArrayRefs(s->u.AssignStmt.rhs, rhs_allref, &alpha);

//        int ignore_first_ref = 0;
//        if (strcmp(s->u.AssignStmt.astype, "+=") == 0 || strcmp(s->u.AssignStmt.astype, "-=") == 0)
//          ignore_first_ref = 1;

//        tce_string_array lhs_indices = (tce_string_array)collectExpIndices(s->u.AssignStmt.lhs, &ignore_first_ref);
//        tce_string_array rhs_indices = (tce_string_array)collectExpIndices(s->u.AssignStmt.rhs, &ignore_first_ref);

//            print_index_list(lhs_indices);
//            printf("=");
//            print_index_list(rhs_indices);

//            int tc_ids[MAX_TENSOR_DIMS];
//            getIndexIDs(eqn, vector_get(&lhs_aref,0), tc_ids);

//            for (i=0;i<MAX_TENSOR_DIMS;i++)
//                if(tc_ids[i]!=-1) printf("%d, ",tc_ids[i]);


        int rhs_aref_count = 0;
        for (i = 0; i < rhs_allref.size(); i++) {
          Exp e = (Exp) rhs_allref.at(i);
          if (e->kind == Exp_::is_NumConst) {
            Exp e1 = (Exp)rhs_allref.at(i + 1);
            if (e1->kind == Exp_::is_ArrayRef) {
              rhs_aref_count++;
              rhs_aref.push_back(make_ArrayRefAlpha(e->u.NumConst.value * e1->coef, e1));
              i++;
            }
          } else {
            rhs_aref_count++;
            Exp e1 = (Exp)rhs_allref.at(i);
            rhs_aref.push_back(make_ArrayRefAlpha(1.0 * e1->coef, e1));
          }
        }

        tamm_bool rhs_first_ref = false;
//        tce_string_array rhs_first_ref_indices = (tce_string_array)collectExpIndices(
//            ((ArrayRefAlpha) vector_get(&rhs_aref, 0))->aref, &ignore_first_ref);

        if (rhs_aref.size() > 1) {
          Exp tc_exp = (Exp)lhs_aref.at(0);
          Exp ta_exp = ((ArrayRefAlpha) rhs_aref.at(0))->aref;
          if (strcmp(tc_exp->u.Array.name, ta_exp->u.Array.name) == 0) rhs_first_ref = true;
        }

        //Exp tcp = vector_get(&lhs_aref, 0);
        //printf("name = %s\n",tcp->u.Array.name);

//            tamm_bool isAMOp = (exact_compare_index_lists(lhs_indices, rhs_indices));
//            //a1121[p3,h1,p2,h2] = t_vo[p3,h1] * t_vo[p2,h2];
//            tamm_bool firstRefInd = (lhs_indices->length > rhs_first_ref_indices->length);
//            tamm_bool isEqInd = (lhs_indices->length == rhs_indices->length);
//            tamm_bool isAddOp = isEqInd && isAMOp;
//            tamm_bool isMultOp =
//                    (lhs_indices->length < rhs_indices->length) || (isEqInd && !isAMOp) || (isEqInd && firstRefInd);

        tamm_bool isAddOp = false;
        tamm_bool isMultOp = false;

        if (rhs_first_ref) rhs_aref_count -= 1;

        if (rhs_aref_count == 2) isMultOp = true;
        else if (rhs_aref_count == 1 || rhs_aref_count > 2) isAddOp = true;

        if (isMultOp) {
          //printf(" == MULT OP\n");

          Exp tc_exp = (Exp)lhs_aref.at(0);
          int ta_ind = 0;
          if (rhs_first_ref) ta_ind++;

          MultOp mop = make_MultOp(0, 0, 0, ((ArrayRefAlpha) rhs_aref.at(ta_ind))->alpha);

          Exp ta_exp = ((ArrayRefAlpha) rhs_aref.at(ta_ind))->aref;
          Exp tb_exp = ((ArrayRefAlpha) rhs_aref.at(ta_ind + 1))->aref;

          getIndexIDs(eqn, tc_exp, mop->tc_ids);
          getIndexIDs(eqn, ta_exp, mop->ta_ids);
          getIndexIDs(eqn, tb_exp, mop->tb_ids);

          getTensorIDs(eqn, tc_exp, &mop->tc);
          getTensorIDs(eqn, ta_exp, &mop->ta);
          getTensorIDs(eqn, tb_exp, &mop->tb);

          eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeMult, nullptr, mop));
          op_id++;

        } else if (isAddOp) {

          //printf(" == ADD OP\n");

          Exp tc_exp = (Exp) lhs_aref.at(0);

          int ta_ind = 0;
          if (rhs_first_ref) ta_ind++;

          AddOp mop = make_AddOp(0, 0, ((ArrayRefAlpha) rhs_aref.at(ta_ind))->alpha);

          Exp ta_exp = ((ArrayRefAlpha) rhs_aref.at(ta_ind))->aref;

          getIndexIDs(eqn, tc_exp, mop->tc_ids);
          getIndexIDs(eqn, ta_exp, mop->ta_ids);

          getTensorIDs(eqn, tc_exp, &mop->tc);
          getTensorIDs(eqn, ta_exp, &mop->ta);

          eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeAdd, mop, nullptr));
          op_id++;

          if (rhs_aref.size() > ta_ind + 1) {
            int k;
            for (k = ta_ind + 1; k < rhs_aref.size(); k++) {
              AddOp aop = make_AddOp(0, 0, ((ArrayRefAlpha) rhs_aref.at(k))->alpha);

              Exp ta_exp = ((ArrayRefAlpha) rhs_aref.at(k))->aref;

              getIndexIDs(eqn, tc_exp, aop->tc_ids);
              getIndexIDs(eqn, ta_exp, aop->ta_ids);

              getTensorIDs(eqn, tc_exp, &aop->tc);
              getTensorIDs(eqn, ta_exp, &aop->ta);

              eqn->op_entries.push_back(make_OpEntry(op_id, OpTypeAdd, aop, nullptr));
              op_id++;
            }
          }

        } else {
          fprintf(stderr, "NEITHER ADD OR MULT OP.. THIS SHOULD NOT HAPPEN!\n");
          exit(0);
        }
      }
            break;
        default: {
          fprintf(stderr, "Not an Assignment Statement!\n");
          exit(0);
        }
    }
}


void getTensorIDs(Equations *eqn, Exp exp, int *tid) {
    if (exp->kind == Exp_::is_ArrayRef) {
        tamm_string aname = exp->u.Array.name;
        int j;
        TensorEntry ient;
        for (j = 0; j < eqn->tensor_entries.size(); j++) {
            ient = (TensorEntry)eqn->tensor_entries.at(j);
            if (strcmp(aname, ient->name) == 0) {
                *tid = j;
                break;
            }
        }
    }
}


void getIndexIDs(Equations *eqn, Exp exp, int *tc_ids) {

    int i;
    for (i = 0; i < MAX_TENSOR_DIMS; i++) tc_ids[i] = -1;
    if (exp->kind == Exp_::is_ArrayRef) {
        tamm_string *aind = exp->u.Array.indices;
        int len = exp->u.Array.length;
        int j;
        int ipos = 0;
        IndexEntry ient;
        for (i = 0; i < len; i++) {
            for (j = 0; j < eqn->index_entries.size(); j++) {
                ient = (IndexEntry)eqn->index_entries.at(j);
                if (strcmp(aind[i], ient->name) == 0) {
                    tc_ids[ipos] = j;
                    ipos++;
                    break;
                }
            }
        }

    }
}


void generate_intermediate_ExpList(Equations *eqn, ExpList* expList, tamm_string am) {
    ExpList* elist = expList;
    while (elist != nullptr) {
        generate_intermediate_Exp(eqn, elist->head);
        elist = elist->tail;
        //if (elist != nullptr) fprintf(eqn, "%s ", am);
    }
    elist = nullptr;
}

void generate_intermediate_Exp(Equations *eqn, Exp exp) {
    switch (exp->kind) {
      case Exp_::is_Parenth: {
        generate_intermediate_Exp(eqn, exp->u.Parenth.exp);
      }
            break;
        case Exp_::is_NumConst: {}
            //fprintf(eqn, "%f ", exp->u.NumConst.value);
            break;
        case Exp_::is_ArrayRef: {}
            //fprintf(eqn, "%s[%s] ", exp->u.Array.name, combine_indices(exp->u.Array.indices, exp->u.Array.length));
            break;
        case Exp_::is_Addition: {
          generate_intermediate_ExpList(eqn, exp->u.Addition.subexps, "+");
        }
            break;
        case Exp_::is_Multiplication: {
          generate_intermediate_ExpList(eqn, exp->u.Multiplication.subexps, "*");
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Expression!\n");
          exit(0);
        }
    }
}


void collectArrayRefs(Exp exp, std::vector<Exp> &arefs, double *alpha) {
    ExpList* el = nullptr;
    switch (exp->kind) {
        case Exp_::is_Parenth: {
          collectArrayRefs(exp->u.Parenth.exp, arefs, alpha);
        }
            break;
        case Exp_::is_NumConst: {
          *alpha = *alpha * exp->u.NumConst.value;
          arefs.push_back(exp);
        }
            break;
        case Exp_::is_ArrayRef: {
            *alpha = *alpha * exp->coef;
            arefs.push_back(exp);
          }
            break;
        case Exp_::is_Addition: {
          el = (exp->u.Addition.subexps);
          *alpha = *alpha * exp->coef;
          while (el != nullptr) {
            collectArrayRefs(el->head, arefs, alpha);
            el = el->tail;
          }
        }
            break;
        case Exp_::is_Multiplication: {
          el = (exp->u.Multiplication.subexps);
          *alpha = *alpha * exp->coef;
          while (el != nullptr) {
            collectArrayRefs(el->head, arefs, alpha);
            el = el->tail;
          }
        }
            break;
        default: {
          fprintf(stderr, "Not a valid Expression!\n");
          exit(0);
        }
    }
}


tce_string_array collectExpIndices(Exp exp, int *firstref) {
    ExpList* el = nullptr;
    int i = 0, ui = 0, tot_len = 0;
    tce_string_array p = nullptr;
    tamm_string *uind = nullptr;
    tamm_string *uniq_ind = nullptr;
    tamm_string *all_ind = nullptr;
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
          p = (tce_string_array)tce_malloc(sizeof(*p));
          p->list = replicate_indices(exp->u.Array.indices, exp->u.Array.length);
          p->length = exp->u.Array.length;
          return p;
        }
            break;
        case Exp_::is_Addition: {
          el = exp->u.Addition.subexps;
          tot_len = 0;
          if (*firstref == 1) el = el->tail;
          while (el != nullptr) {
            //print_Exp(el->head);
            tce_string_array se = getUniqIndices(el->head);
            if (se != nullptr) tot_len += se->length;
            se = nullptr;
            el = el->tail;
          }

          el = exp->u.Addition.subexps;
          all_ind = (tamm_string *)tce_malloc(sizeof(tamm_string) * tot_len);

          i = 0, ui = 0;
          if (*firstref == 1) {
            el = el->tail;
            *firstref = 0;
          }
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
          uind = (tamm_string *)tce_malloc(sizeof(tamm_string) * tot_len);

          i = 0, ui = 0;

          for (i = 0; i < tot_len; i++) {
            if (!exists_index(uind, ui, all_ind[i])) {
              uind[ui] = all_ind[i];
              ui++;
            }
          }

          uniq_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * ui);
          for (i = 0; i < ui; i++) uniq_ind[i] = strdup(uind[i]);

          p = (tce_string_array) tce_malloc(sizeof(*p));
          p->list = uniq_ind;
          p->length = ui;

          return p;
        }

            break;
      case Exp_::is_Multiplication: {
        el = exp->u.Multiplication.subexps;
        tot_len = 0;
        if (*firstref == 1) el = el->tail;
        while (el != nullptr) {
          //print_Exp(el->head);
          tce_string_array se = (tce_string_array) getUniqIndices(el->head);
          if (se != nullptr) tot_len += se->length;
          se = nullptr;
          el = el->tail;
        }

        el = exp->u.Multiplication.subexps;
        all_ind = (tamm_string *)tce_malloc(sizeof(tamm_string) * tot_len);

        i = 0, ui = 0;
        if (*firstref == 1) {
          el = el->tail;
          *firstref = 0;
        }
        while (el != nullptr) {
          tce_string_array se = (tce_string_array)getUniqIndices(el->head);
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
        uind = (tamm_string *)tce_malloc(sizeof(tamm_string) * tot_len);

        i = 0, ui = 0;

        for (i = 0; i < tot_len; i++) {
          if (!exists_index(uind, ui, all_ind[i])) {
            uind[ui] = all_ind[i];
            ui++;
          }
        }

        uniq_ind = (tamm_string *) tce_malloc(sizeof(tamm_string) * ui);
        for (i = 0; i < ui; i++) uniq_ind[i] = strdup(uind[i]);

        p = (tce_string_array)tce_malloc(sizeof(*p));
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


