#include "util.h"

void *tce_malloc(int length) {
    void *p = malloc(length);
    if (!p) {
        std::cerr << "\n Out of memory!\n";
        std::exit(EXIT_FAILURE);
    }
    return p;
}

tamm_string mkString(char *s) {
    tamm_string p = (tamm_string) tce_malloc(strlen(s) + 1);
    strcpy(p, s);
    return p;
}

tamm_string *mkIndexList(tamm_string *indices, int length) {
    tamm_string *newlist = (tamm_string *) malloc(length * sizeof(tamm_string));
    int i = 0;
    for (i = 0; i < length; i++) {
        newlist[i] = strdup(indices[i]);
    }
    return newlist;
}

tamm_string int_str(int a) {
    int size = snprintf(nullptr, 0, "%d", a);
    tamm_string val = (tamm_string) malloc(size + 1);
    sprintf(val, "%d", a);
    return val;
}



tamm_string combine_indexLists(tamm_string *indices1, int count1, tamm_string *indices2, int count2) {
    tamm_string upper = combine_indices(indices1, count1);
    tamm_string lower = combine_indices(indices2, count2);

    tamm_string result = (tamm_string) tce_malloc(strlen(upper) + strlen(lower) + 1);
    strcpy(result, upper);
    strcat(result, ":");
    strcat(result, lower);
    //std::cout << result << std::endl;
    return result;

}


tamm_string *replicate_indices(tamm_string *indices, int len) {
    tamm_string *newind = (tamm_string *)tce_malloc(len * sizeof(tamm_string));
    int i = 0;
    for (i = 0; i < len; i++) newind[i] = strdup(indices[i]);

    return newind;
}

tamm_bool exists_index(tamm_string *list, int len, tamm_string x) {
    int i = 0;
    for (i = 0; i < len; i++) if (strcmp(list[i], x) == 0) return true;
    return false;
}

tamm_bool compare_index_lists(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    tamm_string *alist1 = list1->list;
    tamm_string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (!exists_index(alist2, len2, alist1[i])) return false;
    }
    return true;
}

tamm_bool check_array_usage(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    tamm_string *alist1 = list1->list;
    tamm_string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (strcmp(alist1[i], "N") != 0) if (strcmp(alist1[i], alist2[i]) != 0) return false;
    }
    return true;
}


tamm_bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    tamm_string *alist1 = list1->list;
    tamm_string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (strcmp(alist1[i], alist2[i]) != 0) return false;
    }
    return true;
}

void print_index_list(tce_string_array list1) {
    int i = 0;
    for (i = 0; i < list1->length; i++) std::cout << list1->list[i] << ",";

}


int count_index(tamm_string *list, int len, tamm_string x) {
    int count = 0, i = 0;
    for (i = 0; i < len; i++) {
        if (strcmp(list[i], x) == 0) count++;
    }
    return count;
}


tce_string_array stringToList(const tamm_string s) {
    tamm_string str = strdup(s);
    int len = strlen(str);
    tamm_string *list = (tamm_string *) tce_malloc(sizeof(tamm_string) * (len+1));
    int i = 0;

    char *c = strtok(str, " ,:");
    while (c != nullptr) {
        list[i] = c;
        i++;
        c = strtok(nullptr, " ,:");
    }

    free(str);
    str = strdup(s);
    free(list);
    len = i;
    i = 0;
    list = (tamm_string*) tce_malloc(sizeof(tamm_string) * (len+1));

    c = strtok(str, " ,:");
    while (c != nullptr) {
        list[i] = c;
        i++;
        c = strtok(nullptr, " ,:");
    }
    tce_string_array p = (tce_string_array) tce_malloc(sizeof(*p));
    p->list = list;
    p->length = len;

    return p;
}

//Convert string array of indices to comma seperated string
tamm_string combine_indices(tamm_string *indices, int count) {
    if (indices == nullptr) return "\0";

    char *str = nullptr;             /* Pointer to the combined string  */
    int total_length = 0;      /* Total length of combined string */
    int length = 0;            /* Length of a string             */
    int i = 0;                    /* Loop counter                   */

    /* Find total length of the combined string */
    for (i = 0; i < count; i++) total_length += strlen(indices[i]);
    ++total_length;     /* For combined string terminator */

    total_length += count-1; // For commas

    str = (char *) malloc(total_length);  /* Allocate memory for joined strings + commas */
    str[0] = '\0';                      /* Empty string we can append to      */

    for (i = 0; i < count; i++) {
        strcat(str, indices[i]);
        strcat(str, ",");
        length = strlen(str);
    }
    str[length-1] = '\0';           /* followed by terminator */

    return  str;

}


