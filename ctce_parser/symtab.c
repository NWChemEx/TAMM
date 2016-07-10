#include "symtab.h"


struct STEntry_ {
	string key;
	string value;
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
int ST_hash(SymbolTable hashtable, string key) {

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

	SymbolTable hashtable = NULL;
	int i;

	if (size < 1)
		return NULL;

	/* Allocate the table itself. */
	hashtable = malloc(sizeof(SymbolTable));
	if (hashtable == NULL) return NULL;

	/* Allocate pointers to the head nodes. */
	if ((hashtable->table = malloc(sizeof(STEntry) * size)) == NULL) {
		return NULL;
	}
	for (i = 0; i < size; i++) {
		hashtable->table[i] = NULL;
	}

	hashtable->size = size;

	return hashtable;
}

/* Create a key-value pair. */
STEntry ST_newpair(string key, string value) {
	STEntry newpair;

	if ((newpair = malloc(sizeof(STEntry))) == NULL) {
		return NULL;
	}

	if ((newpair->key = mkString(key)) == NULL) {
		return NULL;
	}

	if ((newpair->value = mkString(value)) == NULL) {
		return NULL;
	}

	newpair->next = NULL;

	return newpair;
}

/* Insert a key-value pair into a hash table. */
void ST_insert(SymbolTable SymbolTable, string key, string value) {
	int bin = 0;
	STEntry newpair = NULL;
	STEntry next = NULL;
	STEntry last = NULL;

	bin = ST_hash(SymbolTable, key);

	next = SymbolTable->table[bin];

	while (next != NULL && next->key != NULL && strcmp(key, next->key) > 0) {
		last = next;
		next = next->next;
	}

	/* There's already a pair.  Let's replace that string. */
	if (next != NULL && next->key != NULL && strcmp(key, next->key) == 0) {

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
		} else if (next == NULL) {
			last->next = newpair;

			/* We're in the middle of the list. */
		} else {
			newpair->next = next;
			last->next = newpair;
		}
	}
}

/* Retrieve a key-value pair from a hash table. */
string ST_get(SymbolTable hashtable, string key) {
	int bin = 0;
	STEntry pair;

	bin = ST_hash(hashtable, key);

	/* Step through the bin, looking for our value. */
	pair = hashtable->table[bin];
	while (pair != NULL && pair->key != NULL && strcmp(key, pair->key) > 0) {
		pair = pair->next;
	}

	/* Did we actually find anything? */
	if (pair == NULL || pair->key == NULL || strcmp(key, pair->key) != 0) {
		return NULL;

	} else {
		return pair->value;
	}

}

/* Check existance of a key-value pair from a hash table. Same code as ST_get */
bool ST_contains(SymbolTable hashtable, string key) {
	int bin = 0;
	STEntry pair;

	bin = ST_hash(hashtable, key);

	/* Step through the bin, looking for our value. */
	pair = hashtable->table[bin];
	while (pair != NULL && pair->key != NULL && strcmp(key, pair->key) > 0) {
		pair = pair->next;
	}

	/* Did we actually find anything? */
	if (pair == NULL || pair->key == NULL || strcmp(key, pair->key) != 0) {
		return false;
	} else {
		return true;
	}
}


