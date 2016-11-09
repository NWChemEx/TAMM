#include "symtab.h"


struct STEntry_ {
    tamm_string key;
    tamm_string value;
    STEntry next;
};

//static unsigned int hash(string s0)
//{unsigned int h=0; string s;
//for(s=s0; *s; s++)
//	h = h*65599 + *s;
//return h;
//}

struct hashtable_ {
    int size;
    STEntry *table;
};

/* Hash a string for a particular hash table. */
int ST_hash(SymbolTable hashtable, tamm_string key) {

    unsigned long int hashval = 0;
    int i = 0;
    int key_len = strlen(key);

    /* Convert our string to an integer */
    while (hashval < 4294967295 && i < key_len) {
        hashval = hashval << 8;
        hashval += key[i];
        i++;
    }

    return hashval % hashtable->size;
}

/* Create a new SymbolTable. */
SymbolTable ST_create(int size) {

    SymbolTable hashtable = nullptr;
    int i;

    if (size < 1)
        return nullptr;

    /* Allocate the table itself. */
    hashtable = (SymbolTable) malloc(sizeof(SymbolTable));
    if (hashtable == nullptr) return nullptr;

    /* Allocate pointers to the head nodes. */
    if ((hashtable->table = (STEntry*) malloc(sizeof(STEntry) * size)) == nullptr) {
        return nullptr;
    }
    for (i = 0; i < size; i++) {
        hashtable->table[i] = nullptr;
    }

    hashtable->size = size;

    return hashtable;
}

/* Create a key-value pair. */
STEntry ST_newpair(tamm_string key, tamm_string value) {
    STEntry newpair;

    if ((newpair = (STEntry) malloc(sizeof(STEntry))) == nullptr) {
        return nullptr;
    }

    if ((newpair->key = strdup(key)) == nullptr) {
        return nullptr;
    }

    if ((newpair->value = strdup(value)) == nullptr) {
        return nullptr;
    }

    newpair->next = nullptr;

    return newpair;
}

/* Insert a key-value pair into a hash table. */
void ST_insert(SymbolTable SymbolTable, tamm_string key, tamm_string value) {
    int bin = 0;
    STEntry newpair = nullptr;
    STEntry next = nullptr;
    STEntry last = nullptr;

    bin = ST_hash(SymbolTable, key);

    next = SymbolTable->table[bin];

    while (next != nullptr && next->key != nullptr && strcmp(key, next->key) > 0) {
        last = next;
        next = next->next;
    }

    /* There's already a pair.  Let's replace that string. */
    if (next != nullptr && next->key != nullptr && strcmp(key, next->key) == 0) {

        free(next->value);
        next->value = strdup(value);

        /* Could not find an existing entry. Create a new one */
    } else {
        newpair = ST_newpair(key, value);

        /* We're at the start of the linked list in this bin. */
        if (next == SymbolTable->table[bin]) {
            newpair->next = next;
            SymbolTable->table[bin] = newpair;

            /* We're at the end of the linked list in this bin. */
        } else if (next == nullptr) {
            last->next = newpair;

            /* We're in the middle of the list. */
        } else {
            newpair->next = next;
            last->next = newpair;
        }
    }
}

/* Retrieve a key-value pair from a hash table. */
tamm_string ST_get(SymbolTable hashtable, tamm_string key) {
    int bin = 0;
    STEntry pair;

    bin = ST_hash(hashtable, key);

    /* Step through the bin, looking for our value. */
    pair = hashtable->table[bin];
    while (pair != nullptr && pair->key != nullptr && strcmp(key, pair->key) > 0) {
        pair = pair->next;
    }

    /* Did we actually find anything? */
    if (pair == nullptr || pair->key == nullptr || strcmp(key, pair->key) != 0) {
        return nullptr;

    } else {
        return pair->value;
    }

}

/* Check existance of a key-value pair from a hash table. Same code as ST_get */
tamm_bool ST_contains(SymbolTable hashtable, tamm_string key) {
    int bin = 0;
    STEntry pair;

    bin = ST_hash(hashtable, key);

    /* Step through the bin, looking for our value. */
    pair = hashtable->table[bin];
    while (pair != nullptr && pair->key != nullptr && strcmp(key, pair->key) > 0) {
        pair = pair->next;
    }

    /* Did we actually find anything? */
    if (pair == nullptr || pair->key == nullptr || strcmp(key, pair->key) != 0) {
        return false;
    } else {
        return true;
    }
}


