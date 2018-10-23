#ifndef TAMM_TESTS_EOMGUESS_HPP_
#define TAMM_TESTS_EOMGUESS_HPP_


#include "tamm/tamm.hpp"
#include <algorithm>
#include <complex>
#include "tamm/eigen_utils.hpp"


template<typename T>
void print_tensor(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
      if (buf[i]>0.0000000000001||buf[i]<-0.0000000000001) {
//        std::cout << buf[i] << " ";
         std::cout << buf[i] << endl;
//        std::cout << std::endl;
      }
    }
}

template<typename T>
void print_tensor_all(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << buf[i] << endl;
    }
}

template<typename T>
void eom_guess(int nroots, const TAMM_SIZE& noab, std::vector<T>& p_evl_sorted, std::vector<Tensor<T>>& x1){



std::vector<T> minlist(nroots);
  for(auto root = 0; root < nroots; root++){
     minlist[root]=1000+root; //large positive value
  }

const TAMM_SIZE nvab= p_evl_sorted.size()-noab;
        std::vector<T> p_evl_sorted_occ(noab);
        std::vector<T> p_evl_sorted_virt(nvab);
        std::copy(p_evl_sorted.begin(), p_evl_sorted.begin() + noab,
                  p_evl_sorted_occ.begin());
        std::copy(p_evl_sorted.begin() + noab, p_evl_sorted.end(),
                  p_evl_sorted_virt.begin());
Matrix eom_diff(noab,nvab);
eom_diff.setZero();

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

auto root = 0;
for(auto x = 0; x < noab / 2; x++) {
    for(auto y = 0; y < nvab / 2; y++) {
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) <= *max_ml){
std::cout << "root = " << root << " for " << x <<" " << y<< std::endl;
           Tensor<T>& t = x1.at(root);
           Tensor2D et = tamm_to_eigen_tensor<T,2>(t);
           et.setZero();
           et(y,x) = 1;
           eigen_to_tamm_tensor(t,et);
           root++;
//           std::cout << "//////" << std::endl;
//           std::cout << et(y,x) << std::endl;
//           std::cout << "^^^^^^" << std::endl;
//           std::cout << et << std::endl;
//           std::cout << "******" << std::endl;
//           print_tensor(t);
        }
    }
}

for(auto x = noab / 2; x < noab; x++) {
    for(auto y = nvab / 2; y < nvab; y++) {
        auto max_ml = std::max_element(minlist.begin(),minlist.end());
        if(eom_diff(x, y) <= *max_ml){
std::cout << "root = " << root << " for " << x <<" " << y<< std::endl;
           Tensor<T>& t = x1.at(root);
           Tensor2D et = tamm_to_eigen_tensor<T,2>(t);
           et.setZero();
           et(y,x) = 1;
           eigen_to_tamm_tensor(t,et);
           root++;
//           std::cout << "//////" << std::endl;
//           std::cout << et(y,x) << std::endl;
//           std::cout << "^^^^^^" << std::endl;
//           std::cout << et << std::endl;
//           std::cout << "******" << std::endl;
//           print_tensor(t);
        }
    }
}


std::cout << "root = " << root << std::endl;
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
