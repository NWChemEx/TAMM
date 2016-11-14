#include "tamm_parser.h"

int tce_tokPos = 0;
int tce_lineno = 1;

void tamm_parser(char const *input_file, Equations *genEq) {

  yyscan_t scanner;
  void *parser;
  int yv;

  TranslationUnit *astRoot;
  yylex_init(&scanner);
  parser = ParseAlloc(malloc);

  if (access(input_file, F_OK) == -1) {
    std::cerr << "File " << input_file << " not found!\n";
    std::exit(EXIT_FAILURE);
  }

  FILE *inputFile = fopen(input_file, "r");
  yyset_in(inputFile, scanner);
  while ((yv = yylex(scanner)) != 0) {
    char *tok = yyget_extra(scanner);
    //std::cout << tok << std::endl; //Debug
    Parse(parser, yv, tok, &astRoot);
  }

  Parse(parser, 0, nullptr, &astRoot);
  fclose(inputFile);

  //Call Visitor
  FILE *outputFile = fopen("output.txt", "w");

  if (!outputFile) {
    std::cerr << "Failed to open output file...\n";
    return;
  }

  visit_ast(outputFile, astRoot);
  fclose(outputFile);

  //SymbolTable symtab = ST_create(65535);
  SymbolTable symtab;
  check_ast(astRoot, symtab);

  //Equations genEq;
  generate_intermediate_ast(genEq, astRoot);

  ParseFree(parser, free);
  yylex_destroy(scanner);

}
