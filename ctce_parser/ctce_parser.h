#ifndef __CTCE_PARSER_H__
#define __CTCE_PARSER_H__

#include "parser.h"
#include "visitor.h"
#include "scanner.h"
#include "semant.h"
#include "error.h"
#include "intermediate.h"

void ctce_parser(char const *input_file, Equations *eqn);

#endif /*__CTCE_PARSER_H__*/

