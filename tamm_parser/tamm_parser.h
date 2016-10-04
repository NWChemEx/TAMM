#ifndef __TAMM_PARSER_H__
#define __TAMM_PARSER_H__

#include "parser.h"
#include "visitor.h"
#include "scanner.h"
#include "semant.h"
#include "error.h"
#include "intermediate.h"

void tamm_parser(char const *input_file, Equations *eqn);

#endif /*__TAMM_PARSER_H__*/

