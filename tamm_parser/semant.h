#ifndef SEMANT_H_
#define SEMANT_H_

#include "absyn.h"
//#include "symtab.h"
#include <map>

typedef std::map<std::string, tamm_string> SymbolTable;

void check_ast(TranslationUnit*, SymbolTable&);

void check_CompoundElem(CompoundElem*, SymbolTable&);

void check_Elem(Elem*, SymbolTable&);

void check_Decl(Decl*, SymbolTable&);

void check_Stmt(Stmt*, SymbolTable&);

void check_Exp(Exp*, SymbolTable&);

void check_ExpList(ExpList*, SymbolTable&);

void check_DeclList(DeclList*, SymbolTable&);

tce_string_array getIndices(Exp* exp);

tce_string_array getUniqIndices(Exp* exp);

void print_Exp(Exp*);

void print_ExpList(ExpList*, tamm_string);


#endif
