#ifndef SEMANT_H_
#define SEMANT_H_

#include "absyn.h"
#include "symtab.h"

void check_ast(TranslationUnit, SymbolTable);
void check_CompoundElem(CompoundElem, SymbolTable);
void check_Elem(Elem, SymbolTable);
void check_Decl(Decl, SymbolTable);
void check_Stmt(Stmt, SymbolTable);

void check_Exp(Exp, SymbolTable);
void check_ExpList(ExpList, SymbolTable, string);
void check_DeclList(DeclList, SymbolTable);

#endif
