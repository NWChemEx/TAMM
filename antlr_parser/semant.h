//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------


#ifndef __TAMM_SEMANT_H_
#define __TAMM_SEMANT_H_

#include "absyn.h"
#include <map>
#include <string>

using SymbolTable = std::map<std::string,tamm_string>;

void check_ast(TranslationUnit*, SymbolTable&);

void check_CompoundElem(CompoundElem*, SymbolTable&);

void check_Elem(Elem*, SymbolTable&);

void check_Decl(Decl*, SymbolTable&);

void check_Stmt(Stmt*, SymbolTable&);

void check_Exp(Exp*, SymbolTable&);

void check_ExpList(ExpList*, SymbolTable&);

void check_DeclList(DeclList*, SymbolTable&);

tamm_string_array getIndices(Exp* exp);

tamm_string_array getUniqIndices(Exp* exp);

void print_Exp(Exp*);

void print_ExpList(ExpList*, tamm_string);


#endif
