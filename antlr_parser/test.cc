#include <iostream>

#include "antlr4-runtime.h"
#include "TAMMLexer.h"
#include "TAMMBaseVisitor.h"

using namespace antlr4;

int main(int argc, const char* argv[]) {
  std::ifstream stream;
  stream.open(argv[1]);
  ANTLRInputStream input(stream);
  TAMMLexer lexer(&input);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);

  tree::ParseTree *tree = parser.translation_unit();
  //std::cout << tree->toStringTree(&parser) << std::endl;

  TAMMBaseVisitor *visitor = new TAMMBaseVisitor();
  //tree::ParseTree::visit(tree);
  //visitor.visit(tree);
  visitor->visitTranslation_unit(tree);

  return 0;
}