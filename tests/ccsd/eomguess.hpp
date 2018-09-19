#ifndef TAMM_TESTS_EOMGUESS_HPP_
#define TAMM_TESTS_EOMGUESS_HPP_


#include "tamm/tamm.hpp"
#include <algorithm>
#include <complex>
#include "tamm/eigen_utils.hpp"

template<typename T>
void eom_guess(int nroots, const TAMM_SIZE& noab, std::vector<T>& p_evl_sorted, std::vector<Tensor<T>>& x1){
//PASS nroots and the set of x1 vectors (really only need the first nroots # of vectors)
//PASS p_evl_sorted 

//Allocate minlist(nroots)
std::vector<T> minlist(nroots);
  for(auto root = 0; root < nroots; root++){
     minlist[root]=1000+root; //large positive value
  }

//Allocate DIFF(a,i) a=alpha occ, i=alpha-virtual
const TAMM_SIZE nvab= p_evl_sorted.size()-noab;
        std::vector<T> p_evl_sorted_occ(noab);
        std::vector<T> p_evl_sorted_virt(nvab);
        std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + noab,
                  p_evl_sorted_occ.begin());
        std::copy(p_evl_sorted.begin() + noab, p_evl_sorted.end(),
                  p_evl_sorted_virt.begin());
Matrix eom_diff(noab,nvab);
eom_diff.setZero();


//for(int i = 0, i < occ; i++){                            //i is all occupied spin orbitals
//  for(int a = occ, a < virtual; a++){                    //a is all unoccupied spin orbtals
//    if(spin i == spin a){                                //i and a must be the same spin
//    DIFF(a,i)=p_evl_sorted_virt(a)-p_evl_sorted_occ(i)
//
//      if DIFF(a,i) < max val in [minlist(nroots)]
//	Then max val in [minlist(nroots)] = DIFF(a,i)
//    }
//  }
//}

for(auto x = 0; x < noab / 2; x++) {
    for(auto y = 0; y < nvab / 2; y++) {
        eom_diff(x, y) = p_evl_sorted_virt[y] - p_evl_sorted_occ[x];
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) < *max_ml)
            minlist[std::distance(minlist.begin(),max_ml)] = eom_diff(x,y); 
    }
}

for(auto x = noab / 2; x < noab; x++) {
    for(auto y = nvab / 2; y < nvab; y++) {
        eom_diff(x, y) = p_evl_sorted_virt[y] - p_evl_sorted_occ[x];
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) < *max_ml)
            minlist[std::distance(minlist.begin(),max_ml)] = eom_diff(x,y);         
    }
}

//int root-number=0
//for(int i = 0, i < occ; i++){                            //i is all occupied spin orbitals
//  for(int a = occ, a < virtual; a++){                    //a is all unoccupied spin orbtals
//    if(spin i == spin a){                                //i and a must be the same spin
//      if(DIFF(a,i) <= max[minlist(root)]){ 
//        then x1({},{},root-number)=0    ! Zero all elements first
//             x1(a,1,root-number)=1
//             root-number=root-number+1
//        else nothing
//
//      if root-number=nroots-1 exit this whole loop (max # of roots have been reached)
//      }
//    }
//  }
//}

auto root = 0;
for(auto x = 0; x < noab / 2; x++) {
    for(auto y = 0; y < nvab / 2; y++) {
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) <= *max_ml){
           Tensor<T>& t = x1.at(root);
           Tensor2D et = tamm_to_eigen_tensor<T,2>(t);
           et.setZero();
           et(x,y) = 1;
           eigen_to_tamm_tensor(t,et);
           root++;
        }
    }
}

for(auto x = noab / 2; x < noab; x++) {
    for(auto y = nvab / 2; y < nvab; y++) {
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) <= *max_ml){
           Tensor<T>& t = x1.at(root);
           Tensor2D et = tamm_to_eigen_tensor<T,2>(t);
           et.setZero();
           et(x,y) = 1;
           eigen_to_tamm_tensor(t,et);
           root++;
        }
    }
}
//NOTE: All that is important is which pairs of indices {a,i} give
//      the 'nroots'-number of lowest energy differences. The differences don't matter, 
//      only what pair of indices {a,i} give the lowest energy differences.
//
//      The above algorith creates the DIFF matrix and a list of the lowest energy differences
//      in the first loop and then goes through the DIFF array and check if a given value is 
//      below or equal to the largest values in minlist. If it is then the pair of {a,i} indices
//      is used to create an initial guess x1 vector.
//
//      If in the first loop minlist could not only store the lowest values, but also which pair
//      of indices {a,i} give the lowest energy differences, then there would be no need for the 
//      second loop to search all values of DIFF. Instead, for each pair {a,i} in the minlist,
//      create the initial guess vector x1.
}

#endif //TAMM_TESTS_EOMGUESS_HPP_