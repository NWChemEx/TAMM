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


int main(int argc, const char* argv[]) {

  tamm::Equations* const equations = new tamm::Equations();

  tamm::tamm_parser(argv[1], equations);

// unsigned int i = 0;
//     RangeEntry* rent;
//     std::cout << "\nRANGE ENTRIES... \n";
//     for (i = 0; i < genEq.range_entries.size(); i++) {
//         rent = (RangeEntry*) genEq.range_entries.at(i);
//         std::cout << "At position " << i << " -> " <<  rent->name << std::endl;
//     }

//     IndexEntry* ient;
//     std::cout << "\nINDEX ENTRIES... \n";
//     for (i = 0; i < genEq.index_entries.size(); i++) {
//         ient = (IndexEntry*) genEq.index_entries.at(i);
//         std::cout << "At position " << i << " -> " <<  ient->name << " " << ient->range_id << std::endl;
//     }

//     std::cout << "\nTENSOR ENTRIES... \n";
//     TensorEntry* tent;
//     unsigned int j = 0;
//     for (i = 0; i < genEq.tensor_entries.size(); i++) {
//         tent = (TensorEntry*) genEq.tensor_entries.at(i);
//         std::cout << "At position " << i << " -> {" << tent->name << ", {";
//         for (j = 0; j < tent->ndim; j++) {
//             if (tent->range_ids[j] == 0) std::cout << "O,";
//             if (tent->range_ids[j] == 1) std::cout << "V,";
//             if (tent->range_ids[j] == 2) std::cout << "N,";
//         }
//         std::cout << "}, " << tent->ndim << ", ";
//         std::cout << tent->nupper << "}\n";
//     }

//     std::cout << "\nOP ENTRIES... \n";
//     OpEntry* oent;
//     std::vector<TensorEntry*> &tensor_entries = genEq.tensor_entries;
//     std::vector<IndexEntry*> &index_entries = genEq.index_entries;

//     for (i = 0; i < genEq.op_entries.size(); i++) {
//         oent = (OpEntry*) genEq.op_entries.at(i);
//         if (oent->optype == OpTypeAdd) std::cout << "op" << oent->op_id << ": OpTypeAdd, ";
//         else std::cout << "op" << oent->op_id << ": OpTypeMult, ";
//         unsigned int j;
//         std::cout << std::fixed;
//         if (oent->add != nullptr) {
//             std::cout << getTensorName(tensor_entries, oent->add->tc) << ", " << getTensorName(tensor_entries, oent->add->ta)
//                                                                               << ", " << oent->add->alpha << ", {";
//             for (j = 0; j < MAX_TENSOR_DIMS; j++)
//                 if (oent->add->tc_ids[j] != -1) std::cout << getIndexName(index_entries, oent->add->tc_ids[j]) << ",";
//             std::cout << "}, {";
//             for (j = 0; j < MAX_TENSOR_DIMS; j++)
//                 if (oent->add->ta_ids[j] != -1) std::cout << getIndexName(index_entries, oent->add->ta_ids[j]) << ",";
//             std::cout << "}";
//         }
//         else if (oent->mult != nullptr) {
//             std::cout << getTensorName(tensor_entries, oent->mult->tc) << ", " << getTensorName(tensor_entries, oent->mult->ta)
//                     << ", " << getTensorName(tensor_entries, oent->mult->tb) << ", " << oent->mult->alpha << ", {";
//             for (j = 0; j < MAX_TENSOR_DIMS; j++)
//                 if (oent->mult->tc_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->tc_ids[j]) << ",";
//             std::cout << "}, {";
//             for (j = 0; j < MAX_TENSOR_DIMS; j++)
//                 if (oent->mult->ta_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->ta_ids[j]) << ",";
//             std::cout << "}, {";
//             for (j = 0; j < MAX_TENSOR_DIMS; j++)
//                 if (oent->mult->tb_ids[j] != -1) std::cout << getIndexName(index_entries, oent->mult->tb_ids[j]) << ",";
//             std::cout << "}";
//         }
//         std::cout << std::endl;
//     }

  return 0;
}