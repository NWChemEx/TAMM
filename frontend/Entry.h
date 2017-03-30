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

/// Object stored in symbol table. Contains type information.
#ifndef __TAMM_ENTRY_H__
#define __TAMM_ENTRY_H__

#include "Types.h"

/// Might extend this class at some point in the future if scopes are needed in
/// the language.
namespace tamm {

namespace frontend {

class Entry {
 public:
  Type* const type;
  Entry(Type* const type) : type(type) {}
  ~Entry() {}
};

}  // namespace frontend
}  // namespace tamm

#endif
