#ifndef VISITOR_H_
#define VISITOR_H_

#include "absyn.h"

void visit_ast(FILE *outFile, TranslationUnit root);

void visit_CompoundElem(FILE *outFile, CompoundElem celem);

void visit_Elem(FILE *outFile, Elem el);

void visit_Decl(FILE *outFile, Decl d);

void visit_Stmt(FILE *outFile, Stmt s);

void visit_Exp(FILE *outFile, Exp exp);

void visit_ExpList(FILE *outFile, ExpList* expList, tamm_string am);

void visit_DeclList(FILE *outFile, DeclList* dl);


#endif
