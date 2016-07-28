#include "parser.h"
#include "visitor.h"
#include "scanner.h"
#include "semant.h"
#include "error.h"
#include "intermediate.h"

int tce_tokPos = 0;
int tce_lineno = 1;


char *getTensorName(vector *v, int pos){
    TensorEntry  te = vector_get(v,pos);
    return te->name;
}

char *getIndexName(vector *v, int pos){
    IndexEntry te = vector_get(v,pos);
    return te->name;
}

int main(int argc, char **argv) {

    // Lemon headers
    void *ParseAlloc(void *(*mallocProc)(size_t));
    void ParseFree(void *p, void (*freeProc)(void *));
    void Parse(void *yyp, int yymajor, char *yyminor, TranslationUnit *extra);

    yyscan_t scanner;
    void *parser;
    int yv;

    TranslationUnit astRoot;
    yylex_init(&scanner);
    parser = ParseAlloc(malloc);

    if (access(argv[1], F_OK) == -1) {
        fprintf(stderr, "File %s not found!\n", argv[1]);
        exit(2);
    }

    FILE *inputFile = fopen(argv[1], "r");
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
        return 2;
    }

    visit_ast(outputFile, astRoot);
    fclose(outputFile);

    SymbolTable symtab = ST_create(65535);
    check_ast(astRoot, symtab);

    Equations genEq;


    generate_intermediate_ast(&genEq, astRoot);

    int i = 0;
    RangeEntry rent;
    printf("\nRANGE ENTRIES... \n");
    for (i = 0; i < vector_count(&genEq.range_entries); i++) {
        rent = vector_get(&genEq.range_entries, i);
        printf("At position %d -> %s\n", i, rent->name);
    }

    IndexEntry ient;
    printf("\nINDEX ENTRIES... \n");
    for (i = 0; i < vector_count(&genEq.index_entries); i++) {
        ient = vector_get(&genEq.index_entries, i);
        printf("At position %d -> %s %d\n", i, ient->name, ient->range_id);
    }

    printf("\nTENSOR ENTRIES... \n");
    TensorEntry tent;
    int j = 0;
    for (i = 0; i < vector_count(&genEq.tensor_entries); i++) {
        tent = vector_get(&genEq.tensor_entries, i);
        printf("At position %d -> {%s, {", i, tent->name);
        for (j = 0; j < tent->ndim; j++) {
            if (tent->range_ids[j] == 0) printf("O,");
            if (tent->range_ids[j] == 1) printf("V,");
            if (tent->range_ids[j] == 2) printf("N,");
        }
        printf("}, %d, ", tent->ndim);
        printf("%d}\n", tent->nupper);
    }

    printf("\nOP ENTRIES... \n");
    OpEntry oent;
    vector *tensor_entries = &genEq.tensor_entries;
    vector *index_entries = &genEq.index_entries;

    for (i = 0; i < vector_count(&genEq.op_entries); i++) {
        oent = vector_get(&genEq.op_entries, i);
        if (oent->optype==0) printf("OpTypeAdd, ");
        else printf("OpTypeMult, ");
        int j;
        if (oent->add != NULL) {
            printf("%s, %s, %lf, {", getTensorName(tensor_entries,oent->add->tc), getTensorName(tensor_entries,oent->add->ta), oent->add->alpha);
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->add->tc_ids[j] != -1) printf("%s,", getIndexName(index_entries,oent->add->tc_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->add->ta_ids[j] != -1) printf("%s,", getIndexName(index_entries,oent->add->ta_ids[j]));
            printf("}");
        }
        else if (oent->mult != NULL) {
            printf("%s, %s, %s, %lf, {", getTensorName(tensor_entries,oent->mult->tc), getTensorName(tensor_entries,oent->mult->ta), getTensorName(tensor_entries,oent->mult->tb), oent->mult->alpha);
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->tc_ids[j] != -1) printf("%s,", getIndexName(index_entries,oent->mult->tc_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->ta_ids[j] != -1) printf("%s,", getIndexName(index_entries,oent->mult->ta_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->tb_ids[j] != -1) printf("%s,", getIndexName(index_entries,oent->mult->tb_ids[j]));
            printf("}");
        }
        printf("\n");
    }

/* Insert any transformation code here to test */

    ParseFree(parser, free);
    yylex_destroy(scanner);


    return 0;
}
