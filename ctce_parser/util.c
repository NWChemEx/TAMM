#include "util.h"

void *tce_malloc(int length) {
    void *p = malloc(length);
    if (!p) {
        fprintf(stderr, "\n Out of memory!\n");
        exit(1);
    }
    return p;
}

string mkString(char *s) {
    string p = tce_malloc(strlen(s) + 1);
    strcpy(p, s);
    return p;
}

string *mkIndexList(string *indices, int length) {
    string *newlist = malloc(length * sizeof(string));
    int i = 0;
    for (i = 0; i < length; i++) {
        newlist[i] = mkString(indices[i]);
    }
    return newlist;
}

string int_str(int a) {
    int size = snprintf(NULL, 0, "%d", a);
    string val = malloc(size + 1);
    sprintf(val, "%d", a);
    return val;
}

//Convert string array of indices to comma seperated string
string combine_indices(string *indices, int count) {
    if (indices == NULL) return "";
    char *str = NULL;             /* Pointer to the combined string  */
    int total_length = 0;      /* Total length of combined string */
    int length = 0;            /* Length of a string             */
    int i = 0;                    /* Loop counter                   */

    /* Find total length of the combined string */
    for (i = 0; i < count; i++) {
        total_length += strlen(indices[i]);
        if (indices[i][strlen(indices[i]) - 1] != '\n')
            ++total_length; /* For newline to be added */
    }
    ++total_length;     /* For combined string terminator */

    str = (char *) malloc(total_length);  /* Allocate memory for joined strings */
    str[0] = '\0';                      /* Empty string we can append to      */

    /* Append all the strings */
    for (i = 0; i < count; i++) {
        strcat(str, indices[i]);
        length = strlen(str);

        /* Check if we need to insert newline */
        if (str[length - 1] != ',') {
            str[length] = ',';             /* Append a comma       */
        }
    }
    str[length] = '\0';           /* followed by terminator */
    //printf ("%s = %d\n",str,length);
    return str;

    //  free(str);                /* Free memory for combined string   */
    //  for(i = 0 ; i<count ; i++)           /* Free memory for original strings */
    //    free(str[i]);

}

string combine_indexLists(string *indices1, int count1, string *indices2, int count2) {
    string upper = combine_indices(indices1, count1);
    string lower = combine_indices(indices2, count2);

    char *result = tce_malloc(strlen(upper) + strlen(lower) + 2);
    strcpy(result, upper);
    strcat(result, ":");
    strcat(result, lower);
    //printf("%s\n",result);
    return result;

}


string *replicate_indices(string *indices, int len) {
    string *newind = tce_malloc(len * sizeof(string));
    int i = 0;
    for (i = 0; i < len; i++) newind[i] = indices[i];

    return newind;
}

bool exists_index(string *list, int len, string x) {
    int i = 0;
    for (i = 0; i < len; i++) if (strcmp(list[i], x) == 0) return true;
    return false;
}

bool compare_index_lists(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    string *alist1 = list1->list;
    string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (!exists_index(alist2, len2, alist1[i])) return false;
    }
    return true;
}

bool check_array_usage(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    string *alist1 = list1->list;
    string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (strcmp(alist1[i], "N") != 0) if (strcmp(alist1[i], alist2[i]) != 0) return false;
    }
    return true;
}


bool exact_compare_index_lists(tce_string_array list1, tce_string_array list2) {
    int len1 = list1->length;
    int len2 = list2->length;
    if (len1 != len2) return false;
    string *alist1 = list1->list;
    string *alist2 = list2->list;
    int i = 0;
    for (i = 0; i < len1; i++) {
        if (strcmp(alist1[i], alist2[i]) != 0) return false;
    }
    return true;
}

void print_index_list(tce_string_array list1) {
    int i = 0;
    for (i = 0; i < list1->length; i++) printf("%s,", list1->list[i]);

}


int count_index(string *list, int len, string x) {
    int count = 0, i = 0;
    for (i = 0; i < len; i++) {
        if (strcmp(list[i], x) == 0) count++;
    }
    return count;
}


tce_string_array stringToList(const string s) {
    string str = strdup(s);
    int len = strlen(str);
    string *list = tce_malloc(sizeof(string) * len);
    int i = 0;

    char *c = strtok(str, " ,:");
    while (c != NULL) {
        list[i] = c;
        i++;
        c = strtok(NULL, " ,:");
    }

    free(str);
    str = strdup(s);
    free(list);
    len = i;
    i = 0;
    list = tce_malloc(sizeof(string) * len);

    c = strtok(str, " ,:");
    while (c != NULL) {
        list[i] = c;
        i++;
        c = strtok(NULL, " ,:");
    }
    tce_string_array p = tce_malloc(sizeof(*p));
    p->list = list;
    p->length = len;

    return p;
}

