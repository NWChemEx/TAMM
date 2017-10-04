#include "visitor.h"
#include "absyn.h"

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

void visit_Elem(FILE *outFile, Elem* elem) {
    Elem* e = elem;
    if (e == nullptr) return;

    switch (e->kind) {
        case Elem::is_DeclList:
            visit_DeclList(outFile, e->u.d);
            break;
        case Elem::is_Statement:
            visit_Stmt(outFile, e->u.s);
            break;
        default:
            std::cerr <<  "Not a Declaration or Statement!\n";
            std::exit(EXIT_FAILURE);
    }
}

void visit_DeclList(FILE *outFile, DeclList* decllist) {
    DeclList* dl = decllist;
    while (dl != nullptr) {
        visit_Decl(outFile, dl->head);
        dl = dl->tail;
    }
}

void visit_Decl(FILE *outFile, Decl* d) {
    switch (d->kind) {
        case Decl::is_RangeDecl:
            fprintf(outFile, "range %s : %d;\n", d->u.RangeDecl.name, d->u.RangeDecl.value);
            break;
        case Decl::is_IndexDecl:
            fprintf(outFile, "index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
            break;
        case Decl::is_ArrayDecl: {
            tamm_string_array up_ind(d->u.ArrayDecl.ulen);
            for (int i = 0; i < d->u.ArrayDecl.ulen; i++)
                up_ind[i] = d->u.ArrayDecl.upperIndices[i];

            tamm_string_array lo_ind(d->u.ArrayDecl.llen);
            for (int i = 0; i < d->u.ArrayDecl.llen; i++)
                lo_ind[i] = d->u.ArrayDecl.lowerIndices[i];

            if (d->u.ArrayDecl.irrep == nullptr)
                fprintf(outFile, "array %s[%s][%s];\n", d->u.ArrayDecl.name,
                        combine_indices(up_ind),
                        combine_indices(lo_ind));

            else
                fprintf(outFile, "array %s[%s][%s] : %s;\n", d->u.ArrayDecl.name,
                        combine_indices(up_ind),
                        combine_indices(lo_ind), d->u.ArrayDecl.irrep);

            break;
        }
        default:
            std::cerr <<  "Not a valid Declaration!\n";
            std::exit(EXIT_FAILURE);
    }
}

void visit_Stmt(FILE *outFile, Stmt* s) {
    switch (s->kind) {
        case Stmt::is_AssignStmt:
            if (s->u.AssignStmt.label != nullptr)
                fprintf(outFile, "%s: ", s->u.AssignStmt.label);
            visit_Exp(outFile, s->u.AssignStmt.lhs);
            fprintf(outFile, " %s ",
                    "="); //s->u.AssignStmt.astype); //astype not needed after we flatten. keep it for now.
            visit_Exp(outFile, s->u.AssignStmt.rhs);
            fprintf(outFile, ";\n");
            break;
        default:
            std::cerr <<  "Not an Assignment Statement!\n";
            std::exit(EXIT_FAILURE);
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

void visit_Exp(FILE *outFile, Exp* exp) {
    switch (exp->kind) {
        case Exp::is_Parenth:
            visit_Exp(outFile, exp->u.Parenth.exp);
            break;
        case Exp::is_NumConst:
            fprintf(outFile, "%f ", exp->u.NumConst.value);
            break;
        case Exp::is_ArrayRef: {
            tamm_string_array up_ind(exp->u.Array.length);
            for (int i = 0; i < exp->u.Array.length; i++)
                up_ind[i] = exp->u.Array.indices[i];
            fprintf(outFile, "%s[%s] ", exp->u.Array.name, combine_indices(up_ind));
            break;
        }
        case Exp::is_Addition:
            visit_ExpList(outFile, exp->u.Addition.subexps, "+");
            break;
        case Exp::is_Multiplication:
            visit_ExpList(outFile, exp->u.Multiplication.subexps, "*");
            break;
        default:
            std::cerr <<  "Not a valid Expression!\n";
            std::exit(EXIT_FAILURE);
    }
}


