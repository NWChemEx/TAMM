#include <numeric>
#include <algorithm>
#include <vector>
#include <cassert>
//#include "tammx.h"

using namespace tammx;

// @todo better name
struct Value {
  double val;
  size_t block, pos;
};

inline bool
operator < (const Value& lhs, const Value& rhs) {
  return lhs.val < rhs.val;
}


void tce_eom_ipxguess(double *p_evl_sorted,
                      double &maxdiff,
                      int nroots,
                      double *rtdb_maxeorb,
                      int maxtrials,
                      int *offsets,
                      Irrep irrep_x;
                      std::vector<Tensor*> &x1_tensors,
                      std::vector<Tensor*> &x2_tensors) {
  //
  //     Determine threshold
  //
  auto group_o = TensorVec<SymmGroup>{{DimType::o}};
  Tensor ttemp{group_o, Distribution::tce_nwma, 0, irrep_x, false};
  int ivec = 0;
  tensor_map(ttemp(), [&](const TensorIndex& blockid) {
      ivec +=  tc.block_size(blockid);
    });
  std::vector<Value> p_diff(ivec);
  //std::fill_n(p_diff.begin(), ivec, 1.0e99);
  //std::iota(p_diff.begin(), p_diff.end(), 0);
  int pos = 0; 
  tensor_map(ttemp(), [&](const TensorIndex& blockid) {
      auto size = ttemp.block_size(blockid);
      for(int i=0; i<size; i++) {
        p_diff[pos+i].block = blockid.value();
        p_diff[pos+i].pos = i;
        p_diff[pos+i].val = - p_evl_sorted[offsets[blockid[0].value()] + i];
      }
      pos += size;
    });

  std::sort(p_diff.begin(), p_diff.end());
  auto nroots_reduced = std::min(ivec, nroots);
  auto nxtrials = nroots_reduced;
  
  x1_tensors.clear();
  x2_tensors.clear();
  for(int i=0; i<nxtrials; i++) {
    x1_tensors.push_back(new Tensor{group_o, Distribution::tce_nwma, 0, irrep_x, false});
    // @todo Check where size_2 and k_x2_offset are computed to
    // setup arguments to tensor creation
    x2_tensors.push_back(new Tensor());
  }
  
  //Make trial X1/X2 based on single excitations
  for(int i=0; i<nroots_reduced; i++) {
    Block block = x1_tensors[i].alloc({BlockDim{p_diff[i].block}});
    block.init(0);
    reinterpret_cast<double*>(block.buf())[p_diff[i].pos] = 1;
    x1_tensors[i].put(block);
  }
  
  // int x1pos = 0;
  // int diffpos=0;
  // tensor_map(, [&] (const TensorIndex& blockid) {
  //     int size = ttemp.block_size(blockid);
  //     for(int h3 = 0; h3<size; h3++) {
  //       if (p_diff[diffpos++] <= maxdiff) {
  //         Block block = x1_tensors[pos].alloc(blockid);
  //         block.init(0);
  //         reinterpret_cast<double*>(block.buf())[h3] = 1;        
  //         x1_tensors[pos].put(block);
  //         pos += 1;
  //       }
  //   }
  // }

  
//   if(rtdb_maxeorb != nullptr) {
//     maxdiff = *rtdb_maxeorb;
//   } else {
//     auto nroots_reduced = std::min(ivec, nroots);
//     assert(ivec >= nroots_reduced);
//     std::nth_element(p_diff.begin(), p_diff.begin()+nroots_reduced,
//                      p_diff.end());
//     maxdiff = p_diff[nroots_reduced];
//   }

  
  
// #if 0
//   // @todo Ask KK: where is maxdiff declared? where is it used? How
//   // does it relate to maxeorb (see below)
//   maxdiff = 0;
//   while (true) {
//     double nextmaxdiff = 1.0e99;
//     int jvec = 0;
//     for(double v : p_diff) {
//       if(v < maxdiff) {
//         jvec += 1;
//       }
//       if(v >= maxdiff && v < nextmaxdiff) {
//         nextmaxdiff = v;
//       }
//     }    
//     if (jvec >= nroots_reduced) {
//       break;
//     }
//     maxdiff = nextmaxdiff + 0.001e0;
//   }
// #else
//   assert(ivec >= nroots_reduced);
//   std::nth_element(p_diff.begin(), p_diff.begin()+nroots_reduced, p_diff.end());
//   maxdiff = p_diff[nroots_reduced];
// #endif
//   }
  
//   // @todo Where is this used?
//   // double maxeorb = 0;
//   // bool defmeo = false;
//   // if (rtdb_maxeorb != nullptr) {
//   //   maxeorb = *rtdb_maxeorb;
//   //   defmeo = false;
//   // }
//   // else {
//   //   defmeo = true;
//   //   maxdiff = maxeorb;
//   // }

//   int nxtrials = 0;
//   for (auto v : p_diff) {
//     nxtrials += (v <= maxdiff) ? 1 : 0;
//   }
//   if (nxtrials < nroots_reduced) {
//     errquit("there is a bug in the program");
//   }
//   if (nxtrials > maxtrials) {
//     errquit("tce_eom_xguess: problem too large",nxtrials);
//   }        

//   x1_tensors.clear();
//   x2_tensors.clear();
//   for(int i=0; i<nxtrials; i++) {
//     // @todo Check where size_1 and k_x1_offset are computed to
//     // setup arguments to tensor creation
//     x1_tensors.push_back(new Tensor());
//     // @todo Check where size_2 and k_x2_offset are computed to
//     // setup arguments to tensor creation
//     x2_tensors.push_back(new Tensor());
//   }
  
//   //Make trial X1/X2 based on single excitations
//   int x1pos = 0;
//   int diffpos=0;
//   tensor_map(, [&] (const TensorIndex& blockid) {
//       int size = ttemp.block_size(blockid);
//       for(int h3 = 0; h3<size; h3++) {
//         if (p_diff[diffpos++] <= maxdiff) {
//           Block block = x1_tensors[pos].alloc(blockid);
//           block.init(0);
//           reinterpret_cast<double*>(block.buf())[h3] = 1;        
//           x1_tensors[pos].put(block);
//           pos += 1;
//         }
//     }
//   }
//     //p_diff.clear(); // delete p_diff vector

  nodezero_print("No. of initial right vectors %4d", nxtrials);
  if(nroots > nroots_reduced) {
    nodezero_print("No. of roots reduced from    %4d to %4d", nroots, nroots_reduced);
  }
}
