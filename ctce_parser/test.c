#include "ctce_parser.h"


char *getTensorName(ctce_vector *v, int pos) {
    TensorEntry te = vector_get(v, pos);
    return te->name;
}

char *getIndexName(ctce_vector *v, int pos) {
    IndexEntry te = vector_get(v, pos);
    return te->name;
}

int main(int argc, char **argv) {

    Equations genEq;

    ctce_parser(argv[1], &genEq);

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
    ctce_vector *tensor_entries = &genEq.tensor_entries;
    ctce_vector *index_entries = &genEq.index_entries;

    for (i = 0; i < vector_count(&genEq.op_entries); i++) {
        oent = vector_get(&genEq.op_entries, i);
        if (oent->optype == OpTypeAdd) printf("OpTypeAdd, ");
        else printf("OpTypeMult, ");
        int j;
        if (oent->add != NULL) {
            printf("%s, %s, %lf, {", getTensorName(tensor_entries, oent->add->tc),
                   getTensorName(tensor_entries, oent->add->ta), oent->add->alpha);
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->add->tc_ids[j] != -1) printf("%s,", getIndexName(index_entries, oent->add->tc_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->add->ta_ids[j] != -1) printf("%s,", getIndexName(index_entries, oent->add->ta_ids[j]));
            printf("}");
        }
        else if (oent->mult != NULL) {
            printf("%s, %s, %s, %lf, {", getTensorName(tensor_entries, oent->mult->tc),
                   getTensorName(tensor_entries, oent->mult->ta), getTensorName(tensor_entries, oent->mult->tb),
                   oent->mult->alpha);
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->tc_ids[j] != -1) printf("%s,", getIndexName(index_entries, oent->mult->tc_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->ta_ids[j] != -1) printf("%s,", getIndexName(index_entries, oent->mult->ta_ids[j]));
            printf("}, {");
            for (j = 0; j < MAX_TENSOR_DIMS; j++)
                if (oent->mult->tb_ids[j] != -1) printf("%s,", getIndexName(index_entries, oent->mult->tb_ids[j]));
            printf("}");
        }
        printf("\n");
    }

    return 0;
}
