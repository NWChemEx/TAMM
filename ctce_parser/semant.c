#include "semant.h"

void check_ast(TranslationUnit root){
  CompoundElemList celist = root->celist;
  while(celist != NULL){
    check_CompoundElem( celist->head);
    celist = celist->tail;
  }
  celist = NULL;
}

void check_CompoundElem(CompoundElem celem){
  ElemList elist = celem->elist;
  while(elist != NULL){
    check_Elem( elist->head);
    elist = elist->tail;
  }
  elist = NULL;
}

void check_Elem(Elem elem){
  Elem e = elem;
  if(e==NULL) return;

  switch(e->kind) {
  case is_DeclList:
    check_DeclList(elem->u.d);
    break;
  case is_Statement:
    check_Stmt(e->u.s);
    break;
  default:
    fprintf(stderr,"Not a Declaration or Statement!\n");
    exit(0);
  }
}

void check_DeclList(DeclList decllist){
  DeclList dl = decllist;
  while(dl!=NULL){
    check_Decl(dl->head);
    dl = dl->tail;
  }
}

void check_Decl(Decl d){
  switch(d->kind) {
  case is_RangeDecl:
    if(d->u.RangeDecl.value % 1 != 0 || d->u.RangeDecl.value <= 0)
      fprintf(stderr, "For range declaration %s, the value %d is not a positive integer\n",
          d->u.RangeDecl.name,d->u.RangeDecl.value);
    break;
  case is_IndexDecl:
    //printf("index %s : %s;\n", d->u.IndexDecl.name, d->u.IndexDecl.rangeID);
    break;
  case is_ArrayDecl:
    break;
  default:
    fprintf(stderr,"Not a valid Declaration!\n");
    exit(0);
  }
}

void check_Stmt(Stmt s){
  switch(s->kind) {
  case is_AssignStmt:
    check_Exp( s->u.AssignStmt.lhs);
    //printf(" %s ", s->u.AssignStmt.astype); //astype not needed since we flatten. keep it for now.
    check_Exp( s->u.AssignStmt.rhs);
    //printf("\n");
    break;
  default:
    fprintf(stderr,"Not an Assignment Statement!\n");
    exit(0);
  }
}

void check_ExpList(ExpList expList, string am){
  ExpList elist = expList;
  while(elist != NULL){
    check_Exp( elist->head);
    elist = elist->tail;
    //if(elist!=NULL) //printf("%s ",am);
  }
  elist = NULL;
}

void check_Exp(Exp exp){
  switch(exp->kind) {
  case is_Parenth:
    check_Exp(exp->u.Parenth.exp);
    break;
  case is_NumConst:
    //printf("%f ",exp->u.NumConst.value);
    break;
  case is_ArrayRef:
    ////printf("%s[%s] ",exp->u.Array.name,combine_indices(exp->u.Array.indices,exp->u.Array.length));
    break;
  case is_Addition:
    check_ExpList(exp->u.Addition.subexps,"+");
    break;
  case is_Multiplication:
    check_ExpList(exp->u.Multiplication.subexps,"*");
    break;
  default:
    fprintf(stderr,"Not a valid Expression!\n");
    exit(0);
  }
}

