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

#ifndef __TAMM_PARSER_H__
#define __TAMM_PARSER_H__

#include <iostream>
// #include "Semant.h"
#include "ASTBuilder.h"
#include "Intermediate.h"
#include "TAMMLexer.h"
using namespace antlr4;

namespace tamm {

namespace frontend {

void tamm_frontend(const std::string input_file, Equations* const equations);
}

}  // namespace tamm

#endif /*__TAMM_PARSER_H__*/
