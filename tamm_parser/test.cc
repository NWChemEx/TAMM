#include "tamm_parser.h"


char *getTensorName(std::vector<TensorEntry> v, int pos) {
    TensorEntry te = (TensorEntry) v.at(pos);
    return te->name;
}

char *getIndexName(std::vector<IndexEntry> v, int pos) {
    IndexEntry te = (IndexEntry) v.at(pos);
    return te->name;
}

int main(int argc, char **argv) {

    Equations genEq;

    tamm_parser(argv[1], &genEq);

    unsigned int i = 0;
    RangeEntry rent;
    printf("\nRANGE ENTRIES... \n");
    for (i = 0; i < genEq.range_entries.size(); i++) {
        rent = (RangeEntry) genEq.range_entries.at(i);
        printf("At position %d -> %s\n", i, rent->name);
    }

    IndexEntry ient;
    printf("\nINDEX ENTRIES... \n");
    for (i = 0; i < genEq.index_entries.size(); i++) {
        ient = (IndexEntry) genEq.index_entries.at(i);
        printf("At position %d -> %s %d\n", i, ient->name, ient->range_id);
    }

    printf("\nTENSOR ENTRIES... \n");
    TensorEntry tent;
    unsigned int j = 0;
    for (i = 0; i < genEq.tensor_entries.size(); i++) {
        tent = (TensorEntry) genEq.tensor_entries.at(i);
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
    std::vector<TensorEntry> &tensor_entries = genEq.tensor_entries;
    std::vector<IndexEntry> &index_entries = genEq.index_entries;

    for (i = 0; i < genEq.op_entries.size(); i++) {
        oent = (OpEntry) genEq.op_entries.at(i);
        if (oent->optype == OpTypeAdd) printf("op%d: OpTypeAdd, ", oent->op_id);
        else printf("op%d: OpTypeMult, ", oent->op_id);
        unsigned int j;
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
