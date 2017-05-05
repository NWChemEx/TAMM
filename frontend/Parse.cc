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
#include <regex>
#include "Intermediate.h"
#include "Semant.h"

namespace tamm {

namespace frontend {

class TAMMErrorListener : public BaseErrorListener {
 public:
  void syntaxError(Recognizer *recognizer, Token *offendingSymbol, size_t line,
                   size_t charPositionInLine, const std::string &msg,
                   std::exception_ptr e) override {
    // std::vector<std::string> stack =
    //    ((Parser *)recognizer)->getRuleInvocationStack();
    // std::reverse(stack.begin(), stack.end());
    // std::cout << "rule stack: ";
    // for (auto &x: stack)  std::cout << x << ",";
    // std::cout << std::endl;

    std::cout << "Syntax error at line " << line << ":" << charPositionInLine 
              //<< " at " << offendingSymbol->getText()
               << ": " << msg << std::endl;

    underlineError(recognizer, offendingSymbol, line, charPositionInLine);
  }

  void underlineError(Recognizer *recognizer, Token *offendingToken,
                      size_t line, size_t charPositionInLine) {
    CommonTokenStream *tokens =
        dynamic_cast<CommonTokenStream *>(recognizer->getInputStream());
    const std::string input = tokens->getTokenSource()->getInputStream()->toString();
    std::regex rnewline = std::regex("\n");
    const std::vector<std::string> lines{
        std::sregex_token_iterator(input.begin(), input.end(), rnewline, -1),
        std::sregex_token_iterator()};
    const std::string errorLine = lines[line - 1];
    std::cout << errorLine << std::endl;
    for (int i = 0; i < charPositionInLine; i++) std::cout << " ";
    const int start = offendingToken->getStartIndex();
    const int stop = offendingToken->getStopIndex();
    if (start >= 0 && stop >= 0) {
      for (int i = start; i <= stop; i++) std::cout << "^";
    }
    std::cout << std::endl;
  }
};

void tamm_frontend(const std::string input_file,
                   Equations *const tamm_equations) {
  std::ifstream stream(input_file.c_str());
  if (!stream.good()) Error("File " + input_file + " not found!");
  ANTLRInputStream tamminput(stream);
  TAMMLexer lexer(&tamminput);
  CommonTokenStream tokens(&lexer);
  TAMMParser parser(&tokens);
  parser.removeErrorListeners();
  parser.addErrorListener(new TAMMErrorListener());

  tree::ParseTree *tree = parser.translation_unit();

  auto* visitor = new ASTBuilder();
  CompilationUnit *ast_root = visitor->visit(tree);

  auto* const context = new SymbolTable();
  type_check(ast_root, context);

  // Equations* equations = new Equations();
  generate_equations(ast_root, tamm_equations);
}

}  // namespace frontend

}  // namespace tamm