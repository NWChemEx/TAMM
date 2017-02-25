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

#include "Parse.h"
#include "Intermediate.h"

using namespace tamm;

const std::string getTensorName(std::vector<TensorEntry*> v, int pos) {
    TensorEntry* te = static_cast<TensorEntry*>(v.at(pos));
    return te->tensor_name;
}

const std::string getIndexName(std::vector<IndexEntry*> v, int pos) {
    IndexEntry* ie = static_cast<IndexEntry*>(v.at(pos));
    return ie->index_name;
}

int main(int argc, const char* argv[]) {

  tamm::Equations* const equations = new tamm::Equations();

  tamm::tamm_parser(argv[1], equations);

   unsigned int i = 0;
   RangeEntry* rent = nullptr;
   std::cout << "\nRANGE ENTRIES... \n";
    for (auto &re: equations->range_entries) {
        rent = static_cast<RangeEntry*>(re);
        std::cout << "At position " << i << " -> " <<  rent->range_name << std::endl;
        i++;
    }

    IndexEntry* ient = nullptr;
    i = 0;
    std::cout << "\nINDEX ENTRIES... \n";
     for (auto &ie: equations->index_entries)  {
        ient = static_cast<IndexEntry*>(ie);
        std::cout << "At position " << i << " -> " <<  ient->index_name << " " << ient->range_id << std::endl;
        i++;
    }

    std::cout << "\nTENSOR ENTRIES... \n";
    TensorEntry* tent = nullptr;
    i = 0;
    unsigned int j = 0;
    for (auto &te: equations->tensor_entries) {
        tent = static_cast<TensorEntry*>(te);
        std::cout << "At position " << i << " -> {" << tent->tensor_name << ", {";
        for (j = 0; j < tent->ndim; j++) {
            if (tent->range_ids[j] == 0) std::cout << "O,";
            if (tent->range_ids[j] == 1) std::cout << "V,";
            if (tent->range_ids[j] == 2) std::cout << "N,";
        }
        std::cout << "}, " << tent->ndim << ", ";
        std::cout << tent->nupper << "}\n";
        i++;
    }

    std::cout << "\nOP ENTRIES... \n";
    OpEntry* oent = nullptr;
    std::vector<TensorEntry*> tensor_entries = equations->tensor_entries;
    std::vector<IndexEntry*> index_entries = equations->index_entries;

    i = 0;
    for (i = 0; i < equations->op_entries.size(); i++) {
        oent = (OpEntry*) equations->op_entries.at(i);
        if (oent->optype == OpTypeAdd) std::cout << "op" << oent->op_id << ": OpTypeAdd, ";
        else std::cout << "op" << oent->op_id << ": OpTypeMult, ";
        unsigned int j;
        std::cout << std::fixed;
        if (oent->add != nullptr) {
            std::cout << getTensorName(tensor_entries, oent->add->tc) << ", " << getTensorName(tensor_entries, oent->add->ta)
                                                                              << ", " << oent->add->alpha << ", {";
            for (j = 0; j < oent->add->tc_ids.size(); j++)
                if (oent->add->tc_ids[j] != -1) std::cout << getIndexName(index_entries, oent->add->tc_ids[j]) << ",";
            std::cout << "}, {";
            for (j = 0; j < oent->add->ta_ids.size(); j++)
                if (oent->add->ta_ids[j] != -1) std::cout << getIndexName(index_entries, oent->add->ta_ids[j]) << ",";
            std::cout << "}";
        }
         else if (oent->mult != nullptr) {
             std::cout << getTensorName(tensor_entries, oent->mult->tc) << ", " << getTensorName(tensor_entries, oent->mult->ta)
                     << ", " << getTensorName(tensor_entries, oent->mult->tb) << ", " << oent->mult->alpha << ", {";
            for (j = 0; j < oent->mult->tc_ids.size(); j++)
                if (oent->mult->tc_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->tc_ids[j]) << ",";
            std::cout << "}, {";
            for (j = 0; j < oent->mult->ta_ids.size(); j++)
                if (oent->mult->ta_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->ta_ids[j]) << ",";
            std::cout << "}, {";
            for (j = 0; j < oent->mult->tb_ids.size(); j++)
                if (oent->mult->tb_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->tb_ids[j]) << ",";
            std::cout << "}";
        }
        std::cout << std::endl;
    }

  return 0;
}