/*
 * symtab.h - Symbol table and symbol table Entries.
 *
 */

#include <limits.h>
#include "util.h"

typedef struct STEntry_ *STEntry;

typedef struct hashtable_ *SymbolTable;

typedef char *string;

/* Make a unique symbol from a given string.  
 *  Different calls to S_STEntry("foo") will yield the same S_symbol
 *  value, even if the "foo" strings are at different locations. */

/* Make a new table */
SymbolTable ST_create(int size);

/* Enter a binding "sym->value" into "t", shadowing but not deleting
 *    any previous binding of "sym". */
void ST_insert(SymbolTable SymbolTable, string key, string value);

/* Look up the most recent binding of "sym" in "t", or return NULL
 *    if sym is unbound. */
string ST_get(SymbolTable hashtab, string key);

bool ST_contains(SymbolTable hashtab, string key);


