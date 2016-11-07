#ifndef __TAMM_PARSER_H__
#define __TAMM_PARSER_H__

#include "parser.h"
#include "scanner.h"

#include "visitor.h"
#include "semant.h"
#include "error.h"
#include "intermediate.h"

void tamm_parser(char const *input_file, Equations *eqn);


// Lemon headers
void Parse(void *yyp, int yymajor, void *yyminor, TranslationUnit *extra);
void *ParseAlloc(void *(*mallocProc)(size_t));
void ParseFree(void *p, void (*freeProc)(void *));

#endif /*__TAMM_PARSER_H__*/

