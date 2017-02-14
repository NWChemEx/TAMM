

#include "tamm_parser.h"
#include "antlr4-runtime.h"
#include "TAMMLexer.h"
#include "TAMMParser.h"

using namespace antlr4;

void tamm_parser(const char* input_file) {//, Equations *genEq) {

  std::ifstream stream;
  stream.open(input_file);
  ANTLRInputStream tamminput(stream);
  TAMMLexer lexer(&tamminput);
  CommonTokenStream tokens(&lexer);
  // tokens.fill();
  TAMMParser parser(&tokens);

  //tree::ParseTree *tree = parser.parse();

  //std::cout << tree->toStringTree(&parser) << std::endl;

}