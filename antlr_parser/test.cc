#include <iostream>

#include "antlr4-runtime.h"
#include "TAMMLexer.h"
#include "TAMMParser.h"

using namespace antlr4;

int main(int argc, const char* argv[]) {
  std::ifstream stream;
  stream.open(argv[1]);
  ANTLRInputStream input(stream);
  TAMMLexer lexer(&input);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);

  tree::ParseTree *tree = parser.main();
  std::cout << tree->toStringTree(&parser) << std::endl;

  return 0;
}