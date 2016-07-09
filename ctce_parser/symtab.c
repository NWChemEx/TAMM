#include <limits.h>
#include "symtab.h"
#include "absyn.h"

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
int ST_hash(hashtable hashtab, string key) {

	unsigned long int hashval = 0;
	int i = 0;
	int key_len = strlen(key);

	/* Convert our string to an integer */
	while (hashval < ULONG_MAX && i < key_len) {
		hashval = hashval << 8;
		hashval += key[i];
		i++;
	}

	return hashval % hashtab->size;
}

/* Create a new hashtable. */
hashtable ST_create(int size) {

	hashtable hashtab = NULL;
	int i;

	if (size < 1)
		return NULL;

	/* Allocate the table itself. */
	hashtab = malloc(sizeof(hashtable));
	if (hashtab == NULL) return NULL;

	/* Allocate pointers to the head nodes. */
	if ((hashtab->table = malloc(sizeof(STEntry) * size)) == NULL) {
		return NULL;
	}
	for (i = 0; i < size; i++) {
		hashtab->table[i] = NULL;
	}

	hashtab->size = size;

	return hashtab;
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
void ST_insert(hashtable hashtable, string key, string value) {
	int bin = 0;
	STEntry newpair = NULL;
	STEntry next = NULL;
	STEntry last = NULL;

	bin = ST_hash(hashtable, key);

	next = hashtable->table[bin];

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
		if (next == hashtable->table[bin]) {
			newpair->next = next;
			hashtable->table[bin] = newpair;

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
string ST_get(hashtable hashtab, string key) {
	int bin = 0;
	STEntry pair;

	bin = ST_hash(hashtab, key);

	/* Step through the bin, looking for our value. */
	pair = hashtab->table[bin];
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

