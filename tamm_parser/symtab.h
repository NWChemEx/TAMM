/*
 * symtab.h - Symbol table and symbol table Entries.
 *
 */

#include <limits.h>
#include "util.h"

typedef struct STEntry_ *STEntry;

typedef struct hashtable_ *SymbolTable;

//typedef char *ctce_string;

/* Make a unique symbol from a given string.  
 *  Different calls to S_STEntry("foo") will yield the same S_symbol
 *  value, even if the "foo" strings are at different locations. */

/* Make a new table */
SymbolTable ST_create(int size);

/* Enter a binding "sym->value" into "t", shadowing but not deleting
 *    any previous binding of "sym". */
void ST_insert(SymbolTable SymbolTable, ctce_string key, ctce_string value);

/* Look up the most recent binding of "sym" in "t", or return NULL
 *    if sym is unbound. */
ctce_string ST_get(SymbolTable hashtab, ctce_string key);

ctce_bool ST_contains(SymbolTable hashtab, ctce_string key);


