#ifndef __TAMM_PARSER_H__
#define __TAMM_PARSER_H__

#include <iostream>

#include "TAMMLexer.h"
#include "TAMMBaseVisitor.h"

// #include "semant.h"
// #include "error.h"
#include "intermediate.h"

using namespace antlr4;

void tamm_parser(const char* input_file, Equations &eqn);

#endif /*__TAMM_PARSER_H__*/

