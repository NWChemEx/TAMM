#include "intermediate.h"


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


AddOp make_AddOp(string name, int ndim, int nupper) {
    AddOp p = tce_malloc(sizeof(*p));
    p->name = name;
    p->ndim = ndim;
    p->nupper = nupper;
    return p;
}

void generate_intermediate_ast(Equations *eqn, TranslationUnit root) {
    vector_init(&eqn->index_entries);
    vector_init(&eqn->range_entries);
    vector_init(&eqn->op_entries);
    vector_init(&eqn->tensor_entries);

    vector *re = &eqn->range_entries;
    vector_add(re,make_RangeEntry("O"));
    vector_add(re,make_RangeEntry("V"));
    vector_add(re,make_RangeEntry("N"));

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

            if (strcmp(d->u.IndexDecl.rangeID,"V")==0) rid = 1;
            else if (strcmp(d->u.IndexDecl.rangeID,"N")==0) rid = 2;
            vector_add(&eqn->index_entries,make_IndexEntry(d->u.IndexDecl.name, rid));
            //fprintf(eqn, "index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
            break;
        case is_ArrayDecl:
              te = make_TensorEntry(d->u.ArrayDecl.name,d->u.ArrayDecl.ulen+d->u.ArrayDecl.llen, d->u.ArrayDecl.ulen);
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

            vector_add(&eqn->tensor_entries,te);

//            if (d->u.ArrayDecl.irrep == NULL)
//                fprintf(eqn, "array %s[%s][%s];\n", d->u.ArrayDecl.name,
//                        combine_indices(d->u.ArrayDecl.upperIndices, d->u.ArrayDecl.ulen),
//                        combine_indices(d->u.ArrayDecl.lowerIndices, d->u.ArrayDecl.llen));
//
//            else
//                fprintf(eqn, "array %s[%s][%s] : %s;\n", d->u.ArrayDecl.name,
//                        combine_indices(d->u.ArrayDecl.upperIndices, d->u.ArrayDecl.ulen),
//                        combine_indices(d->u.ArrayDecl.lowerIndices, d->u.ArrayDecl.llen), d->u.ArrayDecl.irrep);

            break;
        default:
            fprintf(stderr, "Not a valid Declaration!\n");
            exit(0);
    }
}

void generate_intermediate_Stmt(Equations *eqn, Stmt s) {
    switch (s->kind) {
        case is_AssignStmt:
            //if (s->u.AssignStmt.label != NULL)
              //  fprintf(eqn, "%s: ", s->u.AssignStmt.label);
            generate_intermediate_Exp(eqn, s->u.AssignStmt.lhs);
            //fprintf(eqn, " %s ",                   "="); //s->u.AssignStmt.astype); //astype not needed after we flatten. keep it for now.
            generate_intermediate_Exp(eqn, s->u.AssignStmt.rhs);
            //fprintf(eqn, ";\n");
            break;
        default:
            fprintf(stderr, "Not an Assignment Statement!\n");
            exit(0);
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


