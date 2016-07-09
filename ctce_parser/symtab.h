/*
 * symbol.h - STEntrys and symbol-tables
 *
 */

typedef struct STEntry_ *STEntry;

typedef struct hashtable_ *hashtable;

typedef char* string;

/* Make a unique symbol from a given string.  
 *  Different calls to S_STEntry("foo") will yield the same S_symbol
 *  value, even if the "foo" strings are at different locations. */
STEntry insertSTEntry(string);

/* Extract the underlying string from a symbol */
string S_name(STEntry);

/* Make a new table */
hashtable ST_create( int size );

/* Enter a binding "sym->value" into "t", shadowing but not deleting
 *    any previous binding of "sym". */
void ST_insert( hashtable hashtable, string key, string value );

/* Look up the most recent binding of "sym" in "t", or return NULL
 *    if sym is unbound. */
string ST_get( hashtable hashtab, string key);


