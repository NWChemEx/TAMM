#include "semant.h"

void check_ast(TranslationUnit root, SymbolTable symtab){
  CompoundElemList celist = root->celist;
  while(celist != NULL){
    check_CompoundElem(celist->head,symtab);
    celist = celist->tail;
  }
  celist = NULL;
}

void check_CompoundElem(CompoundElem celem, SymbolTable symtab){
  ElemList elist = celem->elist;
  while(elist != NULL){
    check_Elem( elist->head,symtab);
    elist = elist->tail;
  }
  elist = NULL;
}

void check_Elem(Elem elem, SymbolTable symtab){
  Elem e = elem;
  if(e==NULL) return;

  switch(e->kind) {
  case is_DeclList:
    check_DeclList(elem->u.d,symtab);
    break;
  case is_Statement:
    check_Stmt(e->u.s,symtab);
    break;
  default:
    fprintf(stderr,"Not a Declaration or Statement!\n");
    exit(0);
  }
}

void check_DeclList(DeclList decllist, SymbolTable symtab){
  DeclList dl = decllist;
  while(dl!=NULL){
    check_Decl(dl->head,symtab);
    dl = dl->tail;
  }
}

void verifyVarDecl(string name, int line_no, SymbolTable symtab){
//    if (ST_contains(symtab,name)){
//        fprintf(stderr,"Error: %s is already defined", name, line_no);
//        exit(2);
//    }
}

void check_Decl(Decl d, SymbolTable symtab){
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

void check_Stmt(Stmt s, SymbolTable symtab){
  switch(s->kind) {
  case is_AssignStmt:
    check_Exp( s->u.AssignStmt.lhs, symtab);
    //printf(" %s ", s->u.AssignStmt.astype); //astype not needed since we flatten. keep it for now.
    check_Exp( s->u.AssignStmt.rhs, symtab);
    //printf("\n");
    break;
  default:
    fprintf(stderr,"Not an Assignment Statement!\n");
    exit(0);
  }
}

void check_ExpList(ExpList expList, SymbolTable symtab, string am){
  ExpList elist = expList;
  while(elist != NULL){
    check_Exp( elist->head, symtab);
    elist = elist->tail;
    //if(elist!=NULL) //printf("%s ",am);
  }
  elist = NULL;
}

void check_Exp(Exp exp, SymbolTable symtab){
  switch(exp->kind) {
  case is_Parenth:
    check_Exp(exp->u.Parenth.exp,symtab);
    break;
  case is_NumConst:
    //printf("%f ",exp->u.NumConst.value);
    break;
  case is_ArrayRef:
    ////printf("%s[%s] ",exp->u.Array.name,combine_indices(exp->u.Array.indices,exp->u.Array.length));
    break;
  case is_Addition:
    check_ExpList(exp->u.Addition.subexps,symtab,"+");
    break;
  case is_Multiplication:
    check_ExpList(exp->u.Multiplication.subexps,symtab,"*");
    break;
  default:
    fprintf(stderr,"Not a valid Expression!\n");
    exit(0);
  }
}

