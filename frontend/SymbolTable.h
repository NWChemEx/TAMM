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

#include<map>
#include<string>
#include "Entry.h"

namespace tamm{

namespace frontend {

class SymbolTable {
    public:
        // Plain C++ map for now
        std::map<const std::string, const Entry* const> context;

        SymbolTable() {};
        ~SymbolTable() {};
        
        void put(std::string key, const Entry* const value){
            context.insert(std::map<const std::string, const Entry* const>::value_type(key, value));
        }
        const Entry* const get(const std::string key) {
            auto st_entry = context.find(key);
            if (st_entry == context.end()) {
                    return nullptr; /// key not defined
            }
            return st_entry->second;

            
        }
};

}
}

#endif