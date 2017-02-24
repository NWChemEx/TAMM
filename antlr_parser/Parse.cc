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

#include "Parse.h"



namespace tamm {

void tamm_parser(const char* input_file, Equations &genEq) {

  std::ifstream stream;
  stream.open(input_file);
  ANTLRInputStream tamminput(stream);
  TAMMLexer lexer(&tamminput);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);

  tree::ParseTree *tree = parser.translation_unit();

  ASTBuilder *visitor = new ASTBuilder();
  CompilationUnit *ast_root = visitor->visit(tree);


  //visit_ast(outputFile, astRoot);

  // SymbolTable symtab;
  // check_ast(astRoot, symtab);

  // generate_intermediate_ast(genEq, astRoot);

}

}