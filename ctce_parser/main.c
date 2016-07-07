#include "parser.h"
#include "visitor.h"
#include "scanner.h"

int tce_tokPos = 0;
int main(int argc, char **argv){

    // Lemon headers
    void *ParseAlloc(void *(*mallocProc)(size_t));
    void ParseFree(void *p,void (*freeProc)(void*));
    void Parse(void *yyp,int yymajor,char *yyminor, TranslationUnit* extra);

    yyscan_t scanner;
    void *parser;
    int yv;

    TranslationUnit astRoot;
    yylex_init(&scanner);
    parser = ParseAlloc(malloc);

    FILE *inputFile = fopen(argv[1], "r");
    yyset_in(inputFile,scanner);
    while((yv=yylex(scanner)) != 0) {
        char *tok = yyget_extra(scanner);
        //printf("%s = %d,%d\n",tok); //Debug
        Parse(parser,yv,tok,&astRoot);
    }

    Parse(parser,0,NULL,&astRoot);
    fclose(inputFile);

    //Call Visitor
    FILE* outputFile = fopen("output.txt", "w");

    if(!outputFile){
        fprintf(stderr,"failed to open output file\n");
        return 2;
    }

    visit_ast(outputFile, astRoot);

    check_ast(astRoot);

    fclose(outputFile);

    ParseFree(parser,free);
    yylex_destroy(scanner);


    return 0;
}
