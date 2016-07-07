#ifndef SEMANT_H_
#define SEMANT_H_

#include "absyn.h"

void check_ast(TranslationUnit root);
void check_CompoundElem(CompoundElem celem);
void check_Elem(Elem el);
void check_Decl(Decl d);
void check_Stmt(Stmt s);

void check_Exp(Exp exp);
void check_ExpList(ExpList expList, string am);
void check_DeclList(DeclList dl);

#endif
