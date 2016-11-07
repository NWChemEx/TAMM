#include "tamm_parser.h"

int tce_tokPos = 0;
int tce_lineno = 1;

void tamm_parser(char const *input_file, Equations *genEq) {

  yyscan_t scanner;
  void *parser;
  int yv;

  TranslationUnit astRoot;
  yylex_init(&scanner);
  parser = ParseAlloc(malloc);

  if (access(input_file, F_OK) == -1) {
    fprintf(stderr, "File %s not found!\n", input_file);
    exit(2);
  }

  FILE *inputFile = fopen(input_file, "r");
  yyset_in(inputFile, scanner);
  while ((yv = yylex(scanner)) != 0) {
    char *tok = yyget_extra(scanner);
    //printf("%s = %d,%d\n",tok); //Debug
    Parse(parser, yv, tok, &astRoot);
  }

  Parse(parser, 0, NULL, &astRoot);
  fclose(inputFile);

  //Call Visitor
  FILE *outputFile = fopen("output.txt", "w");

  if (!outputFile) {
    fprintf(stderr, "failed to open output file\n");
    return;
  }

  visit_ast(outputFile, astRoot);
  fclose(outputFile);

  SymbolTable symtab = ST_create(65535);
  check_ast(astRoot, symtab);

  //Equations genEq;
  generate_intermediate_ast(genEq, astRoot);

  ParseFree(parser, free);
  yylex_destroy(scanner);

}
