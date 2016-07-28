#include "intermediate.h"
#include "semant.h"

//void make_Equations(Equations p) {
//    p = tce_malloc(sizeof(*p));
//    vector_init(&p->index_entries);
//    vector_init(&p->range_entries);
//    vector_init(&p->op_entries);
//    vector_init(&p->tensor_entries);
//    //return p;
//}

RangeEntry make_RangeEntry(string name) {
    RangeEntry p = tce_malloc(sizeof(*p));
    p->name = name;
    return p;
}

IndexEntry make_IndexEntry(string name, int range_id) {
    IndexEntry p = tce_malloc(sizeof(*p));
    p->name = name;
    p->range_id = range_id;
    return p;
}

TensorEntry make_TensorEntry(string name, int ndim, int nupper) {
    TensorEntry p = tce_malloc(sizeof(*p));
    p->name = name;
    p->ndim = ndim;
    p->nupper = nupper;
    return p;
}


AddOp make_AddOp(int tc, int ta, double alpha) {
    AddOp p = tce_malloc(sizeof(*p));
    p->ta = ta;
    p->tc = tc;
    p->alpha = alpha;
    return p;
}

MultOp make_MultOp(int tc, int ta, int tb, double alpha) {
    MultOp p = tce_malloc(sizeof(*p));
    p->ta = ta;
    p->tc = tc;
    p->tb = tb;
    p->alpha = alpha;
    return p;
}

OpEntry make_OpEntry(OpType ot, AddOp ao, MultOp mo) {
    OpEntry p = tce_malloc(sizeof(*p));
    p->optype = ot;
    p->add = ao;
    p->mult = mo;
    return p;
}

void generate_intermediate_ast(Equations *eqn, TranslationUnit root) {
    vector_init(&eqn->index_entries);
    vector_init(&eqn->range_entries);
    vector_init(&eqn->op_entries);
    vector_init(&eqn->tensor_entries);

    vector *re = &eqn->range_entries;
    vector_add(re, make_RangeEntry("O"));
    vector_add(re, make_RangeEntry("V"));
    vector_add(re, make_RangeEntry("N"));

    CompoundElemList celist = root->celist;
    while (celist != NULL) {
        generate_intermediate_CompoundElem(eqn, celist->head);
        celist = celist->tail;
    }
    celist = NULL;
}

void generate_intermediate_CompoundElem(Equations *eqn, CompoundElem celem) {
    ElemList elist = celem->elist;
    while (elist != NULL) {
        generate_intermediate_Elem(eqn, elist->head);
        elist = elist->tail;
    }
    elist = NULL;
}

void generate_intermediate_Elem(Equations *eqn, Elem elem) {
    Elem e = elem;
    if (e == NULL) return;

    switch (e->kind) {
        case is_DeclList:
            generate_intermediate_DeclList(eqn, elem->u.d);
            break;
        case is_Statement:
            generate_intermediate_Stmt(eqn, e->u.s);
            break;
        default:
            fprintf(stderr, "Not a Declaration or Statement!\n");
            exit(0);
    }
}

void generate_intermediate_DeclList(Equations *eqn, DeclList decllist) {
    DeclList dl = decllist;
    while (dl != NULL) {
        generate_intermediate_Decl(eqn, dl->head);
        dl = dl->tail;
    }
}

void generate_intermediate_Decl(Equations *eqn, Decl d) {
    switch (d->kind) {
        int rid = 0;
        TensorEntry te = NULL;
        case is_RangeDecl:
            //fprintf(eqn, "range %s : %d;\n", d->u.RangeDecl.name, d->u.RangeDecl.value);
            break;
        case is_IndexDecl:

            if (strcmp(d->u.IndexDecl.rangeID, "V") == 0) rid = 1;
            else if (strcmp(d->u.IndexDecl.rangeID, "N") == 0) rid = 2;
            vector_add(&eqn->index_entries, make_IndexEntry(d->u.IndexDecl.name, rid));
            //fprintf(eqn, "index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
            break;
        case is_ArrayDecl:
            te = make_TensorEntry(d->u.ArrayDecl.name, d->u.ArrayDecl.ulen + d->u.ArrayDecl.llen, d->u.ArrayDecl.ulen);
            for (rid = 0; rid < d->u.ArrayDecl.ulen; rid++) {
                string range = d->u.ArrayDecl.upperIndices[rid];
                te->range_ids[rid] = 0;
                if (strcmp(range, "V") == 0) te->range_ids[rid] = 1;
                else if (strcmp(range, "N") == 0) te->range_ids[rid] = 2;
            }

            int lid = rid;
            for (rid = 0; rid < d->u.ArrayDecl.llen; rid++) {
                string range = d->u.ArrayDecl.lowerIndices[rid];
                te->range_ids[lid] = 0;
                if (strcmp(range, "V") == 0) te->range_ids[lid] = 1;
                else if (strcmp(range, "N") == 0) te->range_ids[lid] = 2;
                lid++;
            }

            vector_add(&eqn->tensor_entries, te);

            break;
        default:
            fprintf(stderr, "Not a valid Declaration!\n");
            exit(0);
    }
}

void generate_intermediate_Stmt(Equations *eqn, Stmt s) {
    vector lhs_aref, rhs_aref;
    double alpha = 1;
    switch (s->kind) {
        case is_AssignStmt:

            vector_init(&lhs_aref);
            collectArrayRefs(s->u.AssignStmt.lhs, &lhs_aref, &alpha);
//            int i = 0;
//            for (i = 0; i < vector_count(&lhs_aref); i++) {
//                Exp e = vector_get(&lhs_aref, i);
//                printf("%s ", e->u.Array.name);
//            }
            vector_init(&rhs_aref);
            collectArrayRefs(s->u.AssignStmt.rhs, &rhs_aref, &alpha);

            tce_string_array lhs_indices = collectExpIndices(s->u.AssignStmt.lhs);
            tce_string_array rhs_indices = collectExpIndices(s->u.AssignStmt.rhs);


//            print_index_list(lhs_indices);
//            printf("=");
//            print_index_list(rhs_indices);

//            int tc_ids[MAX_TENSOR_DIMS];
//            getIndexIDs(eqn, vector_get(&lhs_aref,0), tc_ids);

//            for (i=0;i<MAX_TENSOR_DIMS;i++)
//                if(tc_ids[i]!=-1) printf("%d, ",tc_ids[i]);

            bool rhs_first_ref = false;
            tce_string_array rhs_first_ref_indices = collectExpIndices(vector_get(&rhs_aref,0));

            if(vector_count(&rhs_aref) > 1) {
                Exp tc_exp = vector_get(&lhs_aref, 0);
                Exp ta_exp = vector_get(&rhs_aref, 0);
                if (strcmp(tc_exp->u.Array.name,ta_exp->u.Array.name) == 0) rhs_first_ref = true;
            }

            bool isAMOp = (exact_compare_index_lists(lhs_indices, rhs_indices));
            //a1121[p3,h1,p2,h2] = t_vo[p3,h1] * t_vo[p2,h2];
            bool firstRefInd = (lhs_indices->length > rhs_first_ref_indices->length);
            bool isEqInd = (lhs_indices->length == rhs_indices->length);
            bool isAddOp = isEqInd && isAMOp;
            bool isMultOp = (lhs_indices->length < rhs_indices->length) || (isEqInd && !isAMOp) || (isEqInd && firstRefInd);

            if (isMultOp) {
                //printf(" == MULT OP\n");

                MultOp mop = make_MultOp(0, 0, 0, alpha);

                Exp tc_exp = vector_get(&lhs_aref, 0);
                Exp ta_exp = vector_get(&rhs_aref, 0);
                Exp tb_exp = vector_get(&rhs_aref, 1);

                getIndexIDs(eqn, tc_exp, mop->tc_ids);
                getIndexIDs(eqn, ta_exp, mop->ta_ids);
                getIndexIDs(eqn, tb_exp, mop->tb_ids);

                getTensorIDs(eqn, tc_exp, &mop->tc);
                getTensorIDs(eqn, ta_exp, &mop->ta);
                getTensorIDs(eqn, tb_exp, &mop->tb);

                vector_add(&eqn->op_entries, make_OpEntry(OpTypeMult, NULL, mop));

            }
            else if (isAddOp) {

                //printf(" == ADD OP\n");

                AddOp mop = make_AddOp(0, 0, alpha);

                Exp tc_exp = vector_get(&lhs_aref, 0);

                int ta_ind = 0;
                if(rhs_first_ref) ta_ind++;

                Exp ta_exp = vector_get(&rhs_aref, ta_ind);

                getIndexIDs(eqn, tc_exp, mop->tc_ids);
                getIndexIDs(eqn, ta_exp, mop->ta_ids);

                getTensorIDs(eqn, tc_exp, &mop->tc);
                getTensorIDs(eqn, ta_exp, &mop->ta);

                vector_add(&eqn->op_entries, make_OpEntry(OpTypeAdd, mop, NULL));

                //FIXME: Negative sign indices -1 coef not handled, coef for one sub-exp is used for all subexps
                if (vector_count(&rhs_aref) > ta_ind+1){
                    int k;
                    for (k = ta_ind + 1;k<vector_count(&rhs_aref);k++){
                        AddOp aop = make_AddOp(0, 0, alpha);

                        Exp ta_exp = vector_get(&rhs_aref, k);

                        getIndexIDs(eqn, tc_exp, aop->tc_ids);
                        getIndexIDs(eqn, ta_exp, aop->ta_ids);

                        getTensorIDs(eqn, tc_exp, &aop->tc);
                        getTensorIDs(eqn, ta_exp, &aop->ta);

                        vector_add(&eqn->op_entries, make_OpEntry(OpTypeAdd, aop, NULL));
                    }
                }

            }
            else {
                fprintf(stderr, "NEITHER ADD OR MULT OP.. THIS SHOULD NOT HAPPEN!\n");
                exit(0);
            }
            break;
        default:
            fprintf(stderr, "Not an Assignment Statement!\n");
            exit(0);
    }
}


void getTensorIDs(Equations *eqn, Exp exp, int *tid) {
    if (exp->kind == is_ArrayRef) {
        string aname = exp->u.Array.name;
        int j;
        TensorEntry ient;
        for (j = 0; j < vector_count(&eqn->tensor_entries); j++) {
            ient = vector_get(&eqn->tensor_entries, j);
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
    if (exp->kind == is_ArrayRef) {
        string *aind = exp->u.Array.indices;
        int len = exp->u.Array.length;
        int j;
        int ipos = 0;
        IndexEntry ient;
        for (i = 0; i < len; i++) {
            for (j = 0; j < vector_count(&eqn->index_entries); j++) {
                ient = vector_get(&eqn->index_entries, j);
                if (strcmp(aind[i], ient->name) == 0) {
                    tc_ids[ipos] = j;
                    ipos++;
                    break;
                }
            }
        }

    }
}


void generate_intermediate_ExpList(Equations *eqn, ExpList expList, string am) {
    ExpList elist = expList;
    while (elist != NULL) {
        generate_intermediate_Exp(eqn, elist->head);
        elist = elist->tail;
        //if (elist != NULL) fprintf(eqn, "%s ", am);
    }
    elist = NULL;
}

void generate_intermediate_Exp(Equations *eqn, Exp exp) {
    switch (exp->kind) {
        case is_Parenth:
            generate_intermediate_Exp(eqn, exp->u.Parenth.exp);
            break;
        case is_NumConst:
            //fprintf(eqn, "%f ", exp->u.NumConst.value);
            break;
        case is_ArrayRef:
            //fprintf(eqn, "%s[%s] ", exp->u.Array.name, combine_indices(exp->u.Array.indices, exp->u.Array.length));
            break;
        case is_Addition:
            generate_intermediate_ExpList(eqn, exp->u.Addition.subexps, "+");
            break;
        case is_Multiplication:
            generate_intermediate_ExpList(eqn, exp->u.Multiplication.subexps, "*");
            break;
        default:
            fprintf(stderr, "Not a valid Expression!\n");
            exit(0);
    }
}


void collectArrayRefs(Exp exp, vector *arefs, double *alpha) {
    ExpList el = NULL;
    switch (exp->kind) {
        case is_Parenth:
            collectArrayRefs(exp->u.Parenth.exp, arefs, alpha);
            break;
        case is_NumConst:
            *alpha = exp->u.NumConst.value;
            break;
        case is_ArrayRef:
            vector_add(arefs, exp);
            break;
        case is_Addition:
            el = (exp->u.Addition.subexps);
            *alpha = exp->coef;
            while (el != NULL) {
                collectArrayRefs(el->head, arefs, alpha);
                el = el->tail;
            }
            break;
        case is_Multiplication:
            el = (exp->u.Multiplication.subexps);
            *alpha = exp->coef;
            while (el != NULL) {
                collectArrayRefs(el->head, arefs, alpha);
                el = el->tail;
            }
            break;
        default:
            fprintf(stderr, "Not a valid Expression!\n");
            exit(0);
    }
}


tce_string_array collectExpIndices(Exp exp) {
    ExpList el = NULL;
    tce_string_array p = NULL;
    switch (exp->kind) {
        case is_Parenth:
            return getIndices(exp->u.Parenth.exp);
            break;
        case is_NumConst:
            return NULL;
            break;
        case is_ArrayRef:
            p = tce_malloc(sizeof(*p));
            p->list = replicate_indices(exp->u.Array.indices, exp->u.Array.length);
            p->length = exp->u.Array.length;
            return p;
            break;
        case is_Addition:
            return getIndices(exp->u.Addition.subexps->head);
            break;
        case is_Multiplication:
            el = exp->u.Multiplication.subexps;
            int tot_len = 0;
            while (el != NULL) {
                //print_Exp(el->head);
                tce_string_array se = getIndices(el->head);
                if (se != NULL) tot_len += se->length;
                se = NULL;
                el = el->tail;
            }

            el = exp->u.Multiplication.subexps;
            string *all_ind = tce_malloc(sizeof(string) * tot_len);

            int i = 0, ui = 0;
            while (el != NULL) {
                tce_string_array se = getIndices(el->head);
                i = 0;
                if (se != NULL) {
                    for (i = 0; i < se->length; i++) {
                        all_ind[ui] = se->list[i];
                        ui++;
                    }
                }
                se = NULL;
                el = el->tail;
            }
            assert(ui == tot_len);
            string *uind = tce_malloc(sizeof(string) * tot_len);

            i = 0, ui = 0;


            for (i = 0; i < tot_len; i++) {
                if (!exists_index(uind, ui, all_ind[i])) {
                    uind[ui] = all_ind[i];
                    ui++;
                }
            }

            string *uniq_ind = tce_malloc(sizeof(string) * ui);
            for (i = 0; i < ui; i++) uniq_ind[i] = strdup(uind[i]);


            p = tce_malloc(sizeof(*p));
            p->list = uniq_ind;
            p->length = ui;

            return p;
            break;
        default:
            fprintf(stderr, "Not a valid Expression!\n");
            exit(0);
    }
}


