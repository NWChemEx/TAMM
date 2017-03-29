//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

/// Types stored in Entry object.
#ifndef __TAMM_UTIL_H__
#define __TAMM_UTIL_H__

#include "Absyn.h"
#include <iostream>
#include <algorithm>

namespace tamm {

    namespace frontend {

using index_list = std::vector<std::string>;
using identifier_list = std::vector<Identifier*>;

bool is_positive_integer(const std::string& s);
bool exists_index(const index_list& indices, const std::string x);
bool compare_index_lists(const index_list& alist1, const index_list& alist2);
void get_array_refs_from_expression(Expression* const exp, std::vector<Array*>& arefs);
void get_all_refs_from_expression(Expression* const exp, std::vector<Array*>& arefs, std::vector<NumConst*>& all_consts);
index_list get_indices_from_identifiers(const identifier_list& id_list);
index_list get_non_summation_indices_from_expression(std::vector<Array*>& arefs);
index_list get_unique_indices_from_expression(std::vector<Array*>& arefs);
const std::string get_index_list_as_string(const index_list& ilist);

}

}
#endif
