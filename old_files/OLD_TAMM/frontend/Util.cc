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

#include "Util.h"
#include <set>

namespace tamm {

namespace frontend {

bool exists_index(const index_list& indices, const std::string x) {
  if (std::find(indices.begin(), indices.end(), x) == indices.end())
    return false;
  return true;
}

bool is_positive_integer(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

bool compare_index_lists(const index_list& alist1, const index_list& alist2) {
  const int len1 = alist1.size();
  const int len2 = alist2.size();
  if (len1 != len2) return false;
  for (auto& index : alist1) {
    if (!exists_index(alist2, index)) return false;
  }
  return true;
}

void get_array_refs_from_expression(Expression* const exp,
                                    std::vector<Array*>& arefs) {
  if (Array* const a = dynamic_cast<Array*>(exp))
    arefs.push_back(a);

  else if (Addition* const add = dynamic_cast<Addition*>(exp))
    for (auto& e : add->subexps) get_array_refs_from_expression(e, arefs);

  else if (Multiplication* const mult = dynamic_cast<Multiplication*>(exp))
    for (auto& m : mult->subexps) get_array_refs_from_expression(m, arefs);
}

void get_all_refs_from_expression(Expression* const exp,
                                  std::vector<Array*>& arefs,
                                  std::vector<NumConst*>& all_consts) {
  if (Array* const a = dynamic_cast<Array*>(exp))
    arefs.push_back(a);

  else if (NumConst* const nc = dynamic_cast<NumConst*>(exp))
    all_consts.push_back(nc);

  else if (Addition* const add = dynamic_cast<Addition*>(exp))
    for (auto& e : add->subexps)
      get_all_refs_from_expression(e, arefs, all_consts);

  else if (Multiplication* const mult = dynamic_cast<Multiplication*>(exp))
    for (auto& m : mult->subexps)
      get_all_refs_from_expression(m, arefs, all_consts);
}

index_list get_indices_from_identifiers(const identifier_list& id_list) {
  index_list indices;
  for (auto& identifier : id_list) indices.push_back(identifier->name);
  return indices;
}

/// Return non-summation indices in the rhs of a contraction
index_list get_non_summation_indices_from_expression(
    std::vector<Array*>& arefs) {
  std::vector<std::string> indices;
  for (auto& arr : arefs) {
    identifier_list a_indices = arr->indices;
    for (auto& index : a_indices) {
      if (!exists_index(indices, index->name))
        indices.push_back(index->name);
      else
        indices.erase(std::remove(indices.begin(), indices.end(), index->name),
                      indices.end());
    }
  }
  return indices;
}

index_list get_unique_indices_from_expression(std::vector<Array*>& arefs) {
  std::set<std::string> indices;
  for (auto& arr : arefs) {
    identifier_list a_indices = arr->indices;
    for (auto& index : a_indices) {
      indices.insert(index->name);
    }
  }
  std::vector<std::string> unique_indices;
  unique_indices.assign(indices.begin(), indices.end());
  return unique_indices;
}

const std::string get_index_list_as_string(const index_list& ilist) {
  std::string list_string = "[";
  for (auto& x : ilist) list_string += x + ",";
  list_string.pop_back();
  list_string += "]";
  return list_string;
}

}  // namespace frontend

}  // namespace tamm