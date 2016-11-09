#include "absyn.h"

Exp* make_Parenth(int pos, Exp* e) {
    Exp* p = new Exp();
    p->pos = pos;
    p->coef = 1;
    p->kind = Exp::is_Parenth;
    p->u.Parenth.exp = e;
    return p;
}


Exp* make_NumConst(int pos, float value) {
    Exp* p = new Exp();
    p->pos = pos;
    p->coef = 1;
    p->kind = Exp::is_NumConst;
    p->u.NumConst.value = value;
    return p;
}

Exp* make_Addition(int pos, ExpList* subexps) {
    Exp* p = new Exp();
    p->pos = pos;
    p->coef = 1;
    p->kind = Exp::is_Addition;
    p->u.Addition.subexps = subexps;
    return p;
}

Exp* make_Multiplication(int pos, ExpList* subexps) {
    Exp* p = new Exp();
    p->pos = pos;
    p->coef = 1;
    p->kind = Exp::is_Multiplication;
    p->u.Multiplication.subexps = subexps;
    return p;
}

Exp* make_Array(int pos, tamm_string name, tamm_string *indices) {
    Exp* p = new Exp();
    p->pos = pos;
    p->coef = 1;
    p->kind = Exp::is_ArrayRef;
    p->u.Array.name = name;
    p->u.Array.indices = indices;
    return p;
}

Stmt* make_AssignStmt(int pos, Exp* lhs, Exp* rhs) {
    Stmt* p = new Stmt();
    p->pos = pos;
    p->kind = Stmt::is_AssignStmt;
    p->u.AssignStmt.lhs = lhs;
    p->u.AssignStmt.rhs = rhs;
    return p;
}

Decl* make_RangeDecl(int pos, tamm_string name, int value) {
    Decl* p = new Decl();
    p->pos = pos;
    p->kind = Decl::is_RangeDecl;
    p->u.RangeDecl.name = strdup(name);
    p->u.RangeDecl.value = value;
    return p;
}

Decl* make_IndexDecl(int pos, tamm_string name, tamm_string rangeID) {
    Decl* p = new Decl();
    p->pos = pos;
    p->kind = Decl::is_IndexDecl;
    p->u.IndexDecl.name = name;
    p->u.IndexDecl.rangeID = rangeID;
    return p;
}

Decl* make_ArrayDecl(int pos, tamm_string name, tamm_string *upperIndices, tamm_string *lowerIndices) {
    Decl* p = new Decl();
    p->pos = pos;
    p->kind = Decl::is_ArrayDecl;
    p->u.ArrayDecl.name = name;
    p->u.ArrayDecl.upperIndices = upperIndices;
    p->u.ArrayDecl.lowerIndices = lowerIndices;
    return p;
}

Elem* make_Elem_DeclList(DeclList *d) {
    Elem* p = new Elem();
    //p->pos = pos;
    p->kind = Elem::is_DeclList;
    p->u.d = d;
    return p;
}

Elem* make_Elem_Stmt(Stmt* s) {
    Elem* p = new Elem();
    //p->pos = pos;
    p->kind = Elem::is_Statement;
    p->u.s = s;
    return p;
}


int count_IDList(IDList* idl) {
    IDList *p = idl;
    int count = 0;
    while (p != nullptr) {
        count++;
        p = p->tail;
    }
    return count;
}

/* TODO: Remove duplicated code for addTail_() - okay for now */

void addTail_ElemList(Elem* newtail, ElemList *origList) {
    ElemList *p = origList;
    ElemList *newList = new ElemList(newtail, nullptr);

    if (p == nullptr) {
        origList = newList;
    } else if (p->head == nullptr) {
        p->head = newtail;
    } else if (p->tail == nullptr) {
        p->tail = newList;
    } else {
        while (p->tail != nullptr)
            p = p->tail;
        p->tail = newList;
    }
    p = nullptr;
}

void addTail_DeclList(Decl* newtail, DeclList *origList) {
    DeclList *p = origList;
    DeclList *newList = new DeclList(newtail, nullptr);

    if (p == nullptr) {
        origList = newList;
    } else if (p->head == nullptr) {
        p->head = newtail;
    } else if (p->tail == nullptr) {
        p->tail = newList;
    } else {
        while (p->tail != nullptr)
            p = p->tail;
        p->tail = newList;
    }
    p = nullptr;
}

void addTail_IDList(Identifier* newtail, IDList* origList) {
    IDList *p = origList;
    IDList *newList = new IDList(newtail, nullptr);

    if (p == nullptr) {
        origList = newList;
    } else if (p->head == nullptr) {
        p->head = newtail;
    } else if (p->tail == nullptr) {
        p->tail = newList;
    } else {
        while (p->tail != nullptr)
            p = p->tail;
        p->tail = newList;
    }
    p = nullptr;
}

void addTail_ExpList(Exp* newtail, ExpList *origList) {
    ExpList *p = origList;
    ExpList *newList = new ExpList(newtail, nullptr);

    if (p == nullptr) {
        origList = newList;
    } else if (p->head == nullptr) {
        p->head = newtail;
    } else if (p->tail == nullptr) {
        p->tail = newList;
    } else {
        while (p->tail != nullptr)
            p = p->tail;
        p->tail = newList;
    }
    p = nullptr;
}

void addTail_CompoundElemList(CompoundElem* newtail, CompoundElemList* origList) {
    CompoundElemList *p = origList;
    CompoundElemList *newList = new CompoundElemList(newtail, nullptr);

    if (p == nullptr) {
        origList = newList;
    } else if (p->head == nullptr) {
        p->head = newtail;
    } else if (p->tail == nullptr) {
        p->tail = newList;
    } else {
        while (p->tail != nullptr)
            p = p->tail;
        p->tail = newList;
    }
    p = nullptr;
}

