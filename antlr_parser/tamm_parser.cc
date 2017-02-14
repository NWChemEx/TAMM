

#include "tamm_parser.h"
#include "antlr4-runtime.h"
#include "TAMMLexer.h"
#include "TAMMBaseVisitor.h"

using namespace antlr4;

void tamm_parser(const char* input_file) {//, Equations *genEq) {

  std::ifstream stream;
  stream.open(input_file);
  ANTLRInputStream tamminput(stream);
  TAMMLexer lexer(&tamminput);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);

  tree::ParseTree *tree = parser.translation_unit();

  //std::cout << tree->toStringTree(&parser) << std::endl;

}