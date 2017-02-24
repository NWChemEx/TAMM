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
#ifndef __TAMM_TYPES_H__
#define __TAMM_TYPES_H__


#include <vector>

namespace tamm {
class Identifier;

class Type {
    public:
        enum kType {
            kRangeType,
            kIndexType,
            kTensorType
        };

        virtual int getType() = 0;

        virtual ~Type() {
        }
};

class RangeType: public Type {
    public:
        const int range;
        RangeType(const int range): range(range) {}

        ~RangeType() {}

        int getType() {
            return Type::kRangeType;
        }
};

class IndexType: public Type {
    public:
        const Identifier* const range_name; //  Look up RangeType from name if needed later

        IndexType(const Identifier* const range_name): range_name(range_name) {}

        int getType() {
            return Type::kIndexType;
        }
};

class TensorType: public Type {
    public:
          const std::vector<Identifier*> upper_indices;
          const std::vector<Identifier*> lower_indices;

    public:
        TensorType(const std::vector<Identifier*> upper_indices, const std::vector<Identifier*> lower_indices) 
                  : upper_indices(upper_indices), lower_indices(lower_indices) {}

    int getType() {
        return Type::kTensorType;
    }

};

}

#endif