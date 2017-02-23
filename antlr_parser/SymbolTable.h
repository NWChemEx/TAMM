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

#ifndef __TAMM_SYMBOLTABLE_H__
#define __TAMM_SYMBOLTABLE_H__

#include<string>
#include "Entry.h"

namespace tamm{

class SymbolTable {
    public:
        SymbolTable() {};
        ~SymbolTable() {};
        const Entry* get(std::string key);
        void put(std::string key, const Entry* const value);
};

}

#endif