
#include "tamm_parser.h"

void tamm_parser(const char* input_file, Equations &genEq) {

  std::ifstream stream;
  stream.open(input_file);
  ANTLRInputStream tamminput(stream);
  TAMMLexer lexer(&tamminput);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);

  tree::ParseTree *tree = parser.translation_unit();

  TAMMBaseVisitor *visitor = new TAMMBaseVisitor();
  visitor->visit(tree);


  //visit_ast(outputFile, astRoot);

  // SymbolTable symtab;
  // check_ast(astRoot, symtab);

  // generate_intermediate_ast(genEq, astRoot);

}