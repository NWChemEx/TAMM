//
// Absyn
//  |
//  +-- TranslationUnit
//  |
//  +-- CompoundElem
//  |
//  +-- Elem
//  |    |
//  |    +-- Decl
//  |    |    |
//  |    |    +-- RangeDecl
//  |    |    +-- IndexDecl
//  |    |    +-- ArrayDecl
//  |    |    +-- ExpandDecl
//  |    |    +-- VolatileDecl
//  |    |    +-- IterationDecl
//  |    |
//  |    +-- Stmt
//  |         |
//  |         +-- AssignStmt
//  |
//  +-- Identifier
//  |
//  +-- Exp
//       |
//       +-- Parenth
//       +-- NumConst
//       +-- Array
//       +-- Multiplication
//       +-- Addition
//
//-----------------------------------------

#ifndef ABSYN_H_
#define ABSYN_H_

#include "util.h"

/* Forward Declarations */

//typedef struct Absyn_ *Absyn;
typedef struct Exp_ *Exp;
typedef struct Stmt_ *Stmt;
typedef struct Decl_ *Decl;
typedef struct Elem_ *Elem;
typedef struct Identifier_ *Identifier;
typedef struct CompoundElem_ *CompoundElem;

//typedef struct TranslationUnit_ *TranslationUnit;
//typedef struct IDList_ *IDList;
//typedef struct CompoundElemList_ *CompoundElemList;

/* The Absyn Hierarchy */

class Absyn //Root of the AST
{
    enum class kind {
        is_TranslationUnit, is_CompoundElem, is_Elem, is_Identifier, is_Exp
    };

};

class ElemList //group of declarations and statements corresponding to a single input
{
public:
    Elem head;
    ElemList* tail;

    ElemList(Elem h, ElemList* t){
        head = h;
        tail = t;
    }
};

class IDList {
public:
    Identifier head;
    IDList* tail;

    IDList(Identifier h, IDList* t){
        head = h;
        tail = t;
    }
};

class ExpList {
public:
    Exp head;
    ExpList *tail;

    ExpList(Exp h, ExpList* t){
        head = h;
        tail = t;
    }

};

class DeclList {
public:
    Decl head;
    DeclList *tail;

    DeclList(Decl h, DeclList *t){
        head = h;
        tail = t;
    }
};

struct CompoundElem_  //represents a single input enclosed in { .. }
{
    ElemList *elist;
};

class CompoundElemList //multiple input equations in a single file
{
public:
    CompoundElem head;
    CompoundElemList* tail;

    CompoundElemList(CompoundElem h, CompoundElemList *t){
        head = h;
        tail = t;
    }
};

class TranslationUnit {
public:
    CompoundElemList* celist;

    TranslationUnit(CompoundElemList *cle){
        celist = cle;
    }
};

struct Identifier_ {
    int pos;
    int lineno;
    tamm_string name;
};


struct Elem_ {
    enum {
        is_DeclList, is_Statement
    } kind;
    //int pos;
    union {
        DeclList *d;
        Stmt s;
    } u;
};

struct Decl_ {
    enum {
        is_RangeDecl, is_IndexDecl, is_ArrayDecl, is_ExpandDecl, is_VolatileDecl, is_IterationDecl
    } kind;
    int lineno;
    int pos;
    union {
        struct {
            int value;
            tamm_string name;
        } RangeDecl;
        struct {
            tamm_string name;
            tamm_string rangeID;
        } IndexDecl;
        struct {
            tamm_string name;
            int ulen, llen;
            tamm_string *upperIndices;
            tamm_string *lowerIndices;
            tamm_string irrep;
        } ArrayDecl;
        //TODO: ExpandDecl, IterationDecl, VolatileDecl
    } u;
};


struct Stmt_ {
    enum {
        is_AssignStmt
    } kind;
    int pos;
    union {
        struct {
            tamm_string label;
            Exp lhs;
            Exp rhs;
            tamm_string astype;
        } AssignStmt;
    } u;
};


struct Exp_ {
    enum {
        is_Parenth, is_NumConst, is_ArrayRef, is_Addition, is_Multiplication
    } kind;
    int pos;
    int lineno;
    float coef;
    union {
        struct {
            Exp exp;
        } Parenth;

        struct {
            float value;
        } NumConst;

        struct {
            tamm_string name;
            int length;
            tamm_string *indices;

        } Array;

        struct {
            ExpList *subexps;
        } Addition;

        struct {
            ExpList *subexps;
        } Multiplication;
    } u;
};

#endif

Exp make_Parenth(int pos, Exp e);

Exp make_NumConst(int pos, float value);

Exp make_Addition(int pos, ExpList *subexps);

Exp make_Multiplication(int pos, ExpList *subexps);

Exp make_Array(int pos, tamm_string name, tamm_string *indices);


Stmt make_AssignStmt(int pos, Exp lhs, Exp rhs);

Decl make_RangeDecl(int pos, tamm_string name, int value);

Decl make_IndexDecl(int pos, tamm_string name, tamm_string rangeID);

Decl
make_ArrayDecl(int pos, tamm_string name, tamm_string *upperIndices, tamm_string *lowerIndices); //TODO: permute and vertex symmetry

Identifier make_Identifier(int pos, tamm_string name);

Elem make_Elem_Stmt(Stmt s);

Elem make_Elem_DeclList(DeclList *d);

int count_IDList(IDList* idl);

CompoundElem make_CompoundElem(ElemList *elist);

void addTail_ElemList(Elem newtail, ElemList *origList);

void addTail_DeclList(Decl newtail, DeclList *origList);

void addTail_IDList(Identifier newtail, IDList* origList);

void addTail_ExpList(Exp newtail, ExpList *origList);

void addTail_CompoundElemList(CompoundElem newtail, CompoundElemList* origList);