#include "visitor.h"

void visit_ast(FILE *outFile, TranslationUnit* root) {
    CompoundElemList* celist = root->celist;
    while (celist != nullptr) {
        visit_CompoundElem(outFile, celist->head);
        celist = celist->tail;
    }
    celist = nullptr;
}

void visit_CompoundElem(FILE *outFile, CompoundElem* celem) {
    ElemList *elist = celem->elist;
    while (elist != nullptr) {
        visit_Elem(outFile, elist->head);
        elist = elist->tail;
    }
    elist = nullptr;
}

void visit_Elem(FILE *outFile, Elem elem) {
    Elem e = elem;
    if (e == nullptr) return;

    switch (e->kind) {
        case Elem_::is_DeclList:
            visit_DeclList(outFile, e->u.d);
            break;
        case Elem_::is_Statement:
            visit_Stmt(outFile, e->u.s);
            break;
        default:
            fprintf(stderr, "Not a Declaration or Statement!\n");
            exit(0);
    }
}

void visit_DeclList(FILE *outFile, DeclList* decllist) {
    DeclList* dl = decllist;
    while (dl != nullptr) {
        visit_Decl(outFile, dl->head);
        dl = dl->tail;
    }
}

void visit_Decl(FILE *outFile, Decl d) {
    switch (d->kind) {
        case Decl_::is_RangeDecl:
            fprintf(outFile, "range %s : %d;\n", d->u.RangeDecl.name, d->u.RangeDecl.value);
            break;
        case Decl_::is_IndexDecl:
            fprintf(outFile, "index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
            break;
        case Decl_::is_ArrayDecl:
            if (d->u.ArrayDecl.irrep == nullptr)
                fprintf(outFile, "array %s[%s][%s];\n", d->u.ArrayDecl.name,
                        combine_indices(d->u.ArrayDecl.upperIndices, d->u.ArrayDecl.ulen),
                        combine_indices(d->u.ArrayDecl.lowerIndices, d->u.ArrayDecl.llen));

            else
                fprintf(outFile, "array %s[%s][%s] : %s;\n", d->u.ArrayDecl.name,
                        combine_indices(d->u.ArrayDecl.upperIndices, d->u.ArrayDecl.ulen),
                        combine_indices(d->u.ArrayDecl.lowerIndices, d->u.ArrayDecl.llen), d->u.ArrayDecl.irrep);

            break;
        default:
            fprintf(stderr, "Not a valid Declaration!\n");
            exit(0);
    }
}

void visit_Stmt(FILE *outFile, Stmt s) {
    switch (s->kind) {
        case Stmt_::is_AssignStmt:
            if (s->u.AssignStmt.label != nullptr)
                fprintf(outFile, "%s: ", s->u.AssignStmt.label);
            visit_Exp(outFile, s->u.AssignStmt.lhs);
            fprintf(outFile, " %s ",
                    "="); //s->u.AssignStmt.astype); //astype not needed after we flatten. keep it for now.
            visit_Exp(outFile, s->u.AssignStmt.rhs);
            fprintf(outFile, ";\n");
            break;
        default:
            fprintf(stderr, "Not an Assignment Statement!\n");
            exit(0);
    }
}

void visit_ExpList(FILE *outFile, ExpList *expList, tamm_string am) {
    ExpList *elist = expList;
    while (elist != nullptr) {
        visit_Exp(outFile, elist->head);
        elist = elist->tail;
        if (elist != nullptr) fprintf(outFile, "%s ", am);
    }
    elist = nullptr;
}

void visit_Exp(FILE *outFile, Exp exp) {
    switch (exp->kind) {
        case Exp_::is_Parenth:
            visit_Exp(outFile, exp->u.Parenth.exp);
            break;
        case Exp_::is_NumConst:
            fprintf(outFile, "%f ", exp->u.NumConst.value);
            break;
        case Exp_::is_ArrayRef:
            fprintf(outFile, "%s[%s] ", exp->u.Array.name, combine_indices(exp->u.Array.indices, exp->u.Array.length));
            break;
        case Exp_::is_Addition:
            visit_ExpList(outFile, exp->u.Addition.subexps, "+");
            break;
        case Exp_::is_Multiplication:
            visit_ExpList(outFile, exp->u.Multiplication.subexps, "*");
            break;
        default:
            fprintf(stderr, "Not a valid Expression!\n");
            exit(0);
    }
}


