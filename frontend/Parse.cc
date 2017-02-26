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
#include "Semant.h"
#include "Intermediate.h"

namespace tamm {

namespace frontend {

  void tamm_frontend(const std::string input_file, Equations* const tamm_equations) {

    std::ifstream stream;
    stream.open(input_file.c_str());
    ANTLRInputStream tamminput(stream);
    TAMMLexer lexer(&tamminput);
    CommonTokenStream tokens(&lexer);
    TAMMParser parser(&tokens);

    tree::ParseTree *tree = parser.translation_unit();

    ASTBuilder *visitor = new ASTBuilder();
    CompilationUnit *ast_root = visitor->visit(tree);

    SymbolTable* const context = new SymbolTable();
    type_check(ast_root, context);

    //Equations* equations = new Equations();
    generate_equations(ast_root, tamm_equations);

  }

}

}