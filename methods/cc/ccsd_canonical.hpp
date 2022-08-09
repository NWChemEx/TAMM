#pragma once

#include "diis.hpp"
#include "ccsd_util.hpp"
#include "ga/macdecls.h"
#include "ga/ga-mpi.h"

#ifdef USE_UPCXX
#include <upcxx/upcxx.hpp>
#endif

using namespace tamm;

template<typename T>
void ccsd_e(Scheduler& sch,
            const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");


    // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
    //                                        SpinPosition::lower};

    Tensor<T> i1{{O, V},{1,1}};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
    std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

    sch.allocate(i1)
        (i1(h6, p5) = f1(h6, p5))
        (i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5))
        (de() = 0)
        (de() += t1(p5, h6) * i1(h6, p5))
        (de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2))
        .deallocate(i1);
}

template<typename T>
void ccsd_t1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");


    // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
    //                                        SpinPosition::lower};

    // std::vector<SpinPosition> {2,2}{SpinPosition::upper,SpinPosition::upper,
    //                                        SpinPosition::lower,SpinPosition::lower};

    Tensor<T> t1_2_1{{O, O},{1,1}};
    Tensor<T> t1_2_2_1{{O, V},{1,1}};
    Tensor<T> t1_3_1{{V, V},{1,1}};
    Tensor<T> t1_5_1{{O, V},{1,1}};
    Tensor<T> t1_6_1{{O, O, O, V},{2,2}};

    TiledIndexLabel p2, p3, p4, p5, p6, p7;
    TiledIndexLabel h1, h4, h5, h6, h7, h8;

    std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
    std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");

    sch
      .allocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
      (t1_2_1(h7, h1) = 0)
      (t1_3_1(p2, p3)  = 0)
      ( i0(p2,h1)            =        f1(p2,h1))
      ( t1_2_1(h7,h1)        =        f1(h7,h1))
      ( t1_2_2_1(h7,p3)      =        f1(h7,p3))
      ( t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
      ( t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
      ( t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
      ( t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
      ( i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
      ( t1_3_1(p2,p3)        =        f1(p2,p3))
      ( t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
      ( i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
      ( i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
      ( t1_5_1(h8,p7)        =        f1(h8,p7))
      ( t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
      ( i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
      ( t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
      ( t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
      ( i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
      ( i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
    .deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);

}

template<typename T>
void ccsd_t2(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");


    // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
    //                                        SpinPosition::lower};

    // std::vector<SpinPosition> {2,2}{SpinPosition::upper,SpinPosition::upper,
    //                                        SpinPosition::lower,SpinPosition::lower};

    Tensor<T> i0_temp{{V, V, O, O},{2,2}};
    Tensor<T> t2_temp{{V, V, O, O},{2,2}};
    Tensor<T> t2_2_1{{O, V, O, O},{2,2}};
    Tensor<T> t2_2_1_temp{{O, V, O, O},{2,2}};
    Tensor<T> t2_2_2_1{{O, O, O, O},{2,2}};
    Tensor<T> t2_2_2_1_temp{{O, O, O, O},{2,2}};
    Tensor<T> t2_2_2_2_1{{O, O, O, V},{2,2}};
    Tensor<T> t2_2_4_1{{O, V},{1,1}};
    Tensor<T> t2_2_5_1{{O, O, O, V},{2,2}};
    Tensor<T> t2_4_1{{O, O},{1,1}};
    Tensor<T> t2_4_2_1{{O, V},{1,1}};
    Tensor<T> t2_5_1{{V, V},{1,1}};
    Tensor<T> t2_6_1{{O, O, O, O},{2,2}};
    Tensor<T> t2_6_1_temp{{O, O, O, O},{2,2}};
    Tensor<T> t2_6_2_1{{O, O, O, V},{2,2}};
    Tensor<T> t2_7_1{{O, V, O, V},{2,2}};
    Tensor<T> vt1t1_1{{O, V, O, O},{2,2}};
    Tensor<T> vt1t1_1_temp{{O, V, O, O},{2,2}};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9) = MO.labels<9>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

    sch.allocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
             t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
             t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp)
    (i0(p3, p4, h1, h2) = v2(p3, p4, h1, h2))
    (t2_4_1(h9, h1) = 0)
    (t2_5_1(p3, p5) = 0)
    (t2_2_1(h10, p3, h1, h2) = v2(h10, p3, h1, h2))

    (t2_2_2_1(h10, h11, h1, h2) = -1 * v2(h10, h11, h1, h2))
    (t2_2_2_2_1(h10, h11, h1, p5) = v2(h10, h11, h1, p5))
    (t2_2_2_2_1(h10, h11, h1, p5) += -0.5 * t1(p6, h1) * v2(h10, h11, p5, p6))

//    (t2_2_2_1(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
//    (t2_2_2_1(h10, h11, h2, h1) += -1 * t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5)) //perm symm
    (t2_2_2_1_temp(h10, h11, h1, h2) = 0)
    (t2_2_2_1_temp(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
    (t2_2_2_1(h10, h11, h1, h2) += t2_2_2_1_temp(h10, h11, h1, h2))
    (t2_2_2_1(h10, h11, h2, h1) += -1 * t2_2_2_1_temp(h10, h11, h1, h2)) //perm symm

    (t2_2_2_1(h10, h11, h1, h2) += -0.5 * t2(p7, p8, h1, h2) * v2(h10, h11, p7, p8))
    (t2_2_1(h10, p3, h1, h2) += 0.5 * t1(p3, h11) * t2_2_2_1(h10, h11, h1, h2))
    
    (t2_2_4_1(h10, p5) = f1(h10, p5))
    (t2_2_4_1(h10, p5) += -1 * t1(p6, h7) * v2(h7, h10, p5, p6))
    (t2_2_1(h10, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_2_4_1(h10, p5))
    (t2_2_5_1(h7, h10, h1, p9) = v2(h7, h10, h1, p9))
    (t2_2_5_1(h7, h10, h1, p9) += t1(p5, h1) * v2(h7, h10, p5, p9))

    // (t2_2_1(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
    // (t2_2_1(h10, p3, h2, h1) += -1 * t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9)) //perm symm
    (t2_2_1_temp(h10, p3, h1, h2) = 0)
    (t2_2_1_temp(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
    (t2_2_1(h10, p3, h1, h2) += t2_2_1_temp(h10, p3, h1, h2))
    (t2_2_1(h10, p3, h2, h1) += -1 * t2_2_1_temp(h10, p3, h1, h2)) //perm symm

    // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2_temp(p1, p2, h3, h4) = 0)
    (t2_temp(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

    (t2_2_1(h10, p3, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(h10, p3, p5, p6))
    // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    

//    (i0(p3, p4, h1, h2) += -1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
//    (i0(p4, p3, h1, h2) += 1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += i0_temp(p3, p4, h1, h2)) //perm sym


    //  (i0(p3, p4, h1, h2) += -1 * t1(p5, h1) * v2(p3, p4, h2, p5))
    //  (i0(p3, p4, h2, h1) += 1 * t1(p5, h1) * v2(p3, p4, h2, p5)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t1(p5, h1) * v2(p3, p4, h2, p5))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym

    (t2_4_1(h9, h1) = f1(h9, h1))
    (t2_4_2_1(h9, p8) = f1(h9, p8))
    (t2_4_2_1(h9, p8) += t1(p6, h7) * v2(h7, h9, p6, p8))
    (t2_4_1(h9, h1) += t1(p8, h1) * t2_4_2_1(h9, p8))
    (t2_4_1(h9, h1) += -1 * t1(p6, h7) * v2(h7, h9, h1, p6))
    (t2_4_1(h9, h1) += -0.5 * t2(p6, p7, h1, h8) * v2(h8, h9, p6, p7))

    // (i0(p3, p4, h1, h2) += -1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
    // (i0(p3, p4, h2, h1) += 1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym


    (t2_5_1(p3, p5) = f1(p3, p5))
    (t2_5_1(p3, p5) += -1 * t1(p6, h7) * v2(h7, p3, p5, p6))
    (t2_5_1(p3, p5) += -0.5 * t2(p3, p6, h7, h8) * v2(h7, h8, p5, p6))

//  (i0(p3, p4, h1, h2) += 1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
//  (i0(p4, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5)) //perm sym
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
    (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm sym

    (t2_6_1(h9, h11, h1, h2) = -1 * v2(h9, h11, h1, h2))
    (t2_6_2_1(h9, h11, h1, p8) = v2(h9, h11, h1, p8))
    (t2_6_2_1(h9, h11, h1, p8) += 0.5 * t1(p6, h1) * v2(h9, h11, p6, p8))
    
//    (t2_6_1(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
//    (t2_6_1(h9, h11, h2, h1) += -1 * t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8)) //perm symm
    (t2_6_1_temp(h9, h11, h1, h2) = 0)
    (t2_6_1_temp(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
    (t2_6_1(h9, h11, h1, h2) += t2_6_1_temp(h9, h11, h1, h2))
    (t2_6_1(h9, h11, h2, h1) += -1 * t2_6_1_temp(h9, h11, h1, h2)) //perm symm

    (t2_6_1(h9, h11, h1, h2) += -0.5 * t2(p5, p6, h1, h2) * v2(h9, h11, p5, p6))
    (i0(p3, p4, h1, h2) += -0.5 * t2(p3, p4, h9, h11) * t2_6_1(h9, h11, h1, h2))

    (t2_7_1(h6, p3, h1, p5) = v2(h6, p3, h1, p5))
    (t2_7_1(h6, p3, h1, p5) += -1 * t1(p7, h1) * v2(h6, p3, p5, p7))
    (t2_7_1(h6, p3, h1, p5) += -0.5 * t2(p3, p7, h1, h8) * v2(h6, h8, p5, p7))

    // (i0(p3, p4, h1, h2) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
    // (i0(p3, p4, h2, h1) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //4 perms
    // (i0(p4, p3, h1, h2) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm
    // (i0(p4, p3, h2, h1) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm

    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
    (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
    (i0(p3, p4, h2, h1) +=  1 * i0_temp(p3, p4, h1, h2)) //4 perms
    (i0(p4, p3, h1, h2) +=  1 * i0_temp(p3, p4, h1, h2)) //perm
    (i0(p4, p3, h2, h1) += -1 * i0_temp(p3, p4, h1, h2)) //perm

    //(vt1t1_1(h5, p3, h1, h2) = 0)
    //(vt1t1_1(h5, p3, h1, h2) += -2 * t1(p6, h1) * v2(h5, p3, h2, p6))
    //(vt1t1_1(h5, p3, h2, h1) += 2 * t1(p6, h1) * v2(h5, p3, h2, p6)) //perm symm
    (vt1t1_1_temp()=0)
    (vt1t1_1_temp(h5, p3, h1, h2) += t1(p6, h1) * v2(h5, p3, h2, p6))
    (vt1t1_1(h5, p3, h1, h2) = -2 * vt1t1_1_temp(h5, p3, h1, h2))
    (vt1t1_1(h5, p3, h2, h1) += 2 * vt1t1_1_temp(h5, p3, h1, h2)) //perm symm

    // (i0(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
    // (i0(p4, p3, h1, h2) += 0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2)) //perm symm
    (i0_temp(p3, p4, h1, h2) = 0)
    (i0_temp(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
    (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
    (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm symm

    // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
    (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

    (i0(p3, p4, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(p3, p4, p5, p6))
    
    // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
    // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perms
    // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perms
    (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
    (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
    (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perms
    (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perms

    .deallocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
              t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
              t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp);
    // sch.execute();

}

template<typename T>
std::tuple<double,double> ccsd_spin_driver(SystemData sys_data, ExecutionContext& ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s, 
                   std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
                   std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted,
                   bool ccsd_restart=false, std::string out_fp="") {

    int    maxiter     = sys_data.options_map.ccsd_options.ccsd_maxiter;
    int    ndiis       = sys_data.options_map.ccsd_options.ndiis;
    double thresh      = sys_data.options_map.ccsd_options.threshold;
    bool   writet      = sys_data.options_map.ccsd_options.writet;
    int    writet_iter = sys_data.options_map.ccsd_options.writet_iter;
    double zshiftl     = sys_data.options_map.ccsd_options.lshift;
    bool   profile     = sys_data.options_map.ccsd_options.profile_ccsd;    
    double residual    = 0.0;
    double energy      = 0.0;
    int    niter       = 0;

    const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
    const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);
    
    std::string t1file = out_fp+".t1amp";
    std::string t2file = out_fp+".t2amp";                       

    std::cout.precision(15);

    Tensor<T> d_e{};
    Tensor<T>::allocate(&ec, d_e);
    Scheduler sch{ec};

    
    if(!ccsd_restart) {

    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {

        const auto timer_start = std::chrono::high_resolution_clock::now();
        
        niter   = iter;
        int off = iter - titer;

        sch
            ((d_t1s[off])()  = d_t1())
            ((d_t2s[off])()  = d_t2())
            .execute();

        ccsd_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);
        ccsd_t1(sch, MO, d_r1, d_t1, d_t2, d_f1, d_v2);
        ccsd_t2(sch, MO, d_r2, d_t1, d_t2, d_f1, d_v2);

        sch.execute(ec.exhw(), profile);

        std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2,
                                d_e, d_r1_residual, d_r2_residual, 
                                p_evl_sorted, zshiftl, n_occ_alpha, n_occ_beta);

        update_r2(ec, d_r2());

        sch
            ((d_r1s[off])() = d_r1())
            ((d_r2s[off])() = d_r2())
            .execute();

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

        iteration_print(sys_data, ec.pg(), iter, residual, energy, iter_time);

        if(writet && ( ((iter+1)%writet_iter == 0) || (residual < thresh) ) ) {
            write_to_disk(d_t1,t1file);
            write_to_disk(d_t2,t2file);
        }   

        if(residual < thresh) { 
            break; 
        }

      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec.pg().rank() == 0) {
          std::cout << " MICROCYCLE DIIS UPDATE:";
          std::cout.width(21);
          std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
          std::cout.width(21);
          std::cout << std::right << "5" << std::endl;
      }

      std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
      std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
      std::vector<Tensor<T>> next_t{d_t1, d_t2};
      diis<T>(ec, rs, ts, next_t);
    }
    Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  } //no restart
  else {
    ccsd_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);

    sch.execute(ec.exhw(), profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

    sys_data.ccsd_corr_energy  = energy;

    if(ec.pg().rank() == 0) {
      sys_data.results["output"]["CCSD"]["n_iterations"] =   niter+1;
      sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] =  energy;
      sys_data.results["output"]["CCSD"]["final_energy"]["total"] =  sys_data.scf_energy+energy;

      write_json_data(sys_data,"CCSD");
    }

    return std::make_tuple(residual,energy);
}
