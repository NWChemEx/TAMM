#ifndef TAMM_DIIS_HPP_
#define TAMM_DIIS_HPP_

#include "ga.h"
#include "tamm/tamm.hpp"

//PASS nroots and the set of x1 vectors (really only need the first nroots # of vectors)
//PASS p_evl_sorted 

//Allocate minlist(nroots)
//Allocate DIFF(a,i) a=alpha occ, i=alpha-virtual

//                                                         //Initialize minlist(nroots)
//   for(root = 0, root < nroots; root++){
//      minlist(root)=1000+root                            //large positive value
//   }

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
