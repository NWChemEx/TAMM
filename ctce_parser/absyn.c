#include "absyn.h"

void *tce_malloc(int length)
{
  void *p = malloc(length);
  if (!p) {
    fprintf(stderr,"\n Out of memory!\n");
    exit(1);
  }
  return p;
}

string mkString(char *s)
{
  string p = tce_malloc(strlen(s)+1);
  strcpy(p,s);
  return p;
}

string* mkIndexList(string *indices, int length){
  string *newlist = malloc(length * sizeof(string));
  int i=0;
  for(i=0;i<length;i++) {
    newlist[i] = mkString(indices[i]);
  }
  return newlist;

}

Identifier make_Identifier(int pos, Symbol name){
  Identifier p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->name = name;
  return p;
}

Exp make_Parenth(int pos, Exp e){
  Exp p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->coef = 1;
  p->kind = is_Parenth;
  p->u.Parenth.exp = e;
  return p;
}


Exp make_NumConst(int pos, float value) {
  Exp p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->coef = 1;
  p->kind = is_NumConst;
  p->u.NumConst.value = value;
  return p;
}

Exp make_Addition(int pos, ExpList subexps) {
  Exp p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->coef = 1;
  p->kind = is_Addition;
  p->u.Addition.subexps = subexps;
  return p;
}

Exp make_Multiplication(int pos, ExpList subexps) {
  Exp p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->coef = 1;
  p->kind = is_Multiplication;
  p->u.Multiplication.subexps = subexps;
  return p;
}

Exp make_Array(int pos, Symbol name, string* indices) {
  Exp p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->coef = 1;
  p->kind = is_ArrayRef;
  p->u.Array.name = name;
  p->u.Array.indices = indices;
  return p;
}

Stmt make_AssignStmt(int pos, Exp lhs, Exp rhs) {
  Stmt p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->kind = is_AssignStmt;
  p->u.AssignStmt.lhs = lhs;
  p->u.AssignStmt.rhs = rhs;
  return p;
}

Decl make_RangeDecl(int pos, Symbol name, int value) {
  Decl p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->kind = is_RangeDecl;
  p->u.RangeDecl.name = mkString(name);
  p->u.RangeDecl.value = value;
  return p;
}

Decl make_IndexDecl(int pos, Symbol name, Symbol rangeID) {
  Decl p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->kind = is_IndexDecl;
  p->u.IndexDecl.name = name;
  p->u.IndexDecl.rangeID = rangeID;
  return p;
}

Decl make_ArrayDecl(int pos, Symbol name, string *upperIndices, string *lowerIndices) {
  Decl p = tce_malloc(sizeof(*p));
  p->pos = pos;
  p->kind = is_ArrayDecl;
  p->u.ArrayDecl.name = name;
  p->u.ArrayDecl.upperIndices = upperIndices;
  p->u.ArrayDecl.lowerIndices = lowerIndices;
  return p;
}

Elem make_Elem_DeclList(DeclList d){
  Elem p = tce_malloc(sizeof(*p));
  //p->pos = pos;
  p->kind = is_DeclList;
  p->u.d = d;
  return p;
}

Elem make_Elem_Stmt(Stmt s){
  Elem p = tce_malloc(sizeof(*p));
  //p->pos = pos;
  p->kind = is_Statement;
  p->u.s = s;
  return p;
}

ElemList make_ElemList(Elem head, ElemList tail) {
  ElemList p = tce_malloc(sizeof(*p));
  p->head=head;
  p->tail=tail;
  return p;
}

DeclList make_DeclList(Decl head, DeclList tail) {
  DeclList p = tce_malloc(sizeof(*p));
  p->head=head;
  p->tail=tail;
  return p;
}

IDList make_IDList(Identifier head, IDList tail) {
  IDList p = tce_malloc(sizeof(*p));
  p->head=head;
  p->tail=tail;
  return p;
}

ExpList make_ExpList(Exp head, ExpList tail) {
  ExpList p = tce_malloc(sizeof(*p));
  p->head=head;
  p->tail=tail;
  return p;
}

CompoundElemList make_CompoundElemList(CompoundElem head, CompoundElemList tail) {
  CompoundElemList p = tce_malloc(sizeof(*p));
  p->head = head;
  p->tail = tail;
  return p;
}

CompoundElem make_CompoundElem(ElemList elist) {
  CompoundElem p = tce_malloc(sizeof(*p));
  p->elist = elist;
  return p;
}

TranslationUnit make_TranslationUnit(CompoundElemList celist) {
  TranslationUnit p = tce_malloc(sizeof(*p));
  p->celist = celist;
  return p;
}

int count_IDList(IDList idl){
  IDList p = idl;
  int count = 0;
    while (p!= NULL) {
      count++;
      p = p->tail;
  }
  return count;
}

/* TODO: Remove duplicated code for addTail_() - okay for now */

void addTail_ElemList(Elem newtail, ElemList origList){
  ElemList p = origList;
  ElemList newList = make_ElemList(newtail, NULL);

  if (p == NULL) {
    origList = newList;
  }else if (p->head == NULL) {
    p->head = newtail;
  } else if (p->tail == NULL) {
    p->tail = newList;
  } else {
    while (p->tail != NULL)
      p = p->tail;
    p->tail = newList;
  }
  p = NULL;
}

void addTail_DeclList(Decl newtail, DeclList origList){
  DeclList p = origList;
  DeclList newList = make_DeclList(newtail, NULL);

  if (p == NULL) {
    origList = newList;
  } else if (p->head == NULL) {
    p->head = newtail;
  } else if (p->tail == NULL) {
    p->tail = newList;
  } else {
    while (p->tail != NULL)
      p = p->tail;
    p->tail = newList;
  }
  p = NULL;
}

void addTail_IDList(Identifier newtail, IDList origList){
  IDList p = origList;
  IDList newList = make_IDList(newtail, NULL);

  if (p == NULL) {
    origList = newList;
  } else if (p->head == NULL) {
    p->head = newtail;
  } else if (p->tail == NULL) {
    p->tail = newList;
  } else {
    while (p->tail != NULL)
      p = p->tail;
    p->tail = newList;
  }
  p = NULL;
}

void addTail_ExpList(Exp newtail, ExpList origList){
  ExpList p = origList;
  ExpList newList = make_ExpList(newtail, NULL);

  if (p == NULL) {
    origList = newList;
  } else if (p->head == NULL) {
    p->head = newtail;
  } else if (p->tail == NULL) {
    p->tail = newList;
  } else {
    while (p->tail != NULL)
      p = p->tail;
    p->tail = newList;
  }
  p = NULL;
}

void addTail_CompoundElemList(CompoundElem newtail, CompoundElemList origList){
  CompoundElemList p = origList;
  CompoundElemList newList = make_CompoundElemList(newtail, NULL);

  if (p == NULL) {
    origList = newList;
  } else if (p->head == NULL) {
    p->head = newtail;
  } else if (p->tail == NULL) {
    p->tail = newList;
  } else {
    while (p->tail != NULL)
      p = p->tail;
    p->tail = newList;
  }
  p = NULL;
}

