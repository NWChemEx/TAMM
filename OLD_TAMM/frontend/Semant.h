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

#include <string>
#include "Absyn.h"
#include "SymbolTable.h"

namespace tamm {

namespace frontend {

void type_check(const CompilationUnit* const, SymbolTable* const);

void check_CompoundElement(const CompoundElement* const ce,
                           SymbolTable* const context);

void check_Element(Element* const element, SymbolTable* const context);

void check_DeclarationList(const DeclarationList* const decllist,
                           SymbolTable* const symtab);

void check_Statement(Statement* const statement, SymbolTable* const context);

void check_AssignStatement(const AssignStatement* const statement,
                           SymbolTable* const context);

void check_Declaration(Declaration* const declaration,
                       SymbolTable* const context);

void check_expression(Expression* const exp, SymbolTable* const context);

// tamm_string_array getIndices(Exp* exp);

// tamm_string_array getUniqIndices(Exp* exp);

}  // namespace frontend

}  // namespace tamm

#endif
