#define CATCH_CONFIG_RUNNER

#include "HF/hartree_fock_eigen.hpp"
#include "diis.hpp"
#include "4index_transform.hpp"
#include "catch/catch.hpp"
#include "eomguess.hpp"
#include "macdecls.h"
#include "ga-mpi.h"

using namespace tamm;

template<typename T>
std::ostream& operator << (std::ostream &os, std::vector<T>& vec){
    os << "[";
    for(auto &x: vec)
        os << x << ",";
    os << "]\n";
    return os;
}

// This will go away when the CCSD routine is replaced by a call to the seperated code.
template<typename T>
void ccsd_e(ExecutionContext &ec,
            const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{O, V};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
    std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

    Scheduler{&ec}.allocate(i1)
        (i1(h6, p5) = f1(h6, p5))
        (i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5))
        (de() = 0)
        (de() += t1(p5, h6) * i1(h6, p5))
        (de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2))
        .deallocate(i1)
        .execute();
}

// This will go away when the CCSD routine is replaced by a call to the seperated code.
template<typename T>
void ccsd_t1(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{O, O};
    Tensor<T> t1_2_2_1{O, V};
    Tensor<T> t1_3_1{V, V};
    Tensor<T> t1_5_1{O, V};
    Tensor<T> t1_6_1{O, O, O, V};

    TiledIndexLabel p2, p3, p4, p5, p6, p7;
    TiledIndexLabel h1, h4, h5, h6, h7, h8;

    std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
    std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");

    Scheduler sch{&ec};
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
    .deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
    .execute();
}

// This will go away when the CCSD routine is replaced by a call to the seperated code.
template<typename T>
void ccsd_t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2) {
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    Tensor<T> i0_temp{V, V, O, O};
    Tensor<T> t2_temp{V, V, O, O};
    Tensor<T> t2_2_1{O, V, O, O};
    Tensor<T> t2_2_1_temp{O, V, O, O};
    Tensor<T> t2_2_2_1{O, O, O, O};
    Tensor<T> t2_2_2_1_temp{O, O, O, O};
    Tensor<T> t2_2_2_2_1{O, O, O, V};
    Tensor<T> t2_2_4_1{O, V};
    Tensor<T> t2_2_5_1{O, O, O, V};
    Tensor<T> t2_4_1{O, O};
    Tensor<T> t2_4_2_1{O, V};
    Tensor<T> t2_5_1{V, V};
    Tensor<T> t2_6_1{O, O, O, O};
    Tensor<T> t2_6_1_temp{O, O, O, O};
    Tensor<T> t2_6_2_1{O, O, O, V};
    Tensor<T> t2_7_1{O, V, O, V};
    Tensor<T> vt1t1_1{O, V, O, O};
    Tensor<T> vt1t1_1_temp{O, V, O, O};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9) = MO.labels<9>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

    Scheduler sch{&ec};
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
    sch.execute();
}

template<typename T>
std::pair<double,double> rest(ExecutionContext& ec,
                              const TiledIndexSpace& MO,
                               Tensor<T>& d_r1,
                               Tensor<T>& d_r2,
                               Tensor<T>& d_t1,
                               Tensor<T>& d_t2,
                              const Tensor<T>& de,
                              std::vector<T>& p_evl_sorted, T zshiftl, 
                              const TAMM_SIZE& noab, bool transpose=false) {

    T residual, energy;
    Scheduler sch{&ec};
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
    sch
      (d_r1_residual() = 0)
      (d_r2_residual() = 0)
      (d_r1_residual() += d_r1()  * d_r1())
      (d_r2_residual() += d_r2()  * d_r2())
      .execute();

      auto l0 = [&]() {
        T r1, r2;
        d_r1_residual.get({}, {&r1, 1});
        d_r2_residual.get({}, {&r2, 1});
        r1 = 0.5*std::sqrt(r1);
        r2 = 0.5*std::sqrt(r2);
        de.get({}, {&energy, 1});
        residual = std::max(r1,r2);
      };

      auto l1 =  [&]() {
        jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted,noab);
      };
      auto l2 = [&]() {
        jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted,noab);
      };

      l0();
      l1();
      l2();

      Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
      
    return {residual, energy};
}


void iteration_print(const ProcGroup& pg, int iter, double residual, double energy) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}

void iteration_print_lambda(const ProcGroup& pg, int iter, double residual) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::string(8, ' ') << "0.0";
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}


template<typename T>
void ccsd_driver(ExecutionContext* ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   int maxiter, double thresh,
                   double zshiftl,
                   int ndiis, double hf_energy,
                   long int total_orbitals, const TAMM_SIZE& noab) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    std::cout.precision(15);

    Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  std::cout << "Total orbitals = " << total_orbitals << std::endl;

  std::vector<double> p_evl_sorted = d_f1.diagonal();

  if(ec->pg().rank() == 0) {
    std::cout << "p_evl_sorted:" << '\n';
    for(size_t p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
  }

  if(ec->pg().rank() == 0) {
    std::cout << "\n\n";
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }
   
  std::vector<Tensor<T>*> d_r1s, d_r2s, d_t1s, d_t2s;

  for(int i=0; i<ndiis; i++) {
    d_r1s.push_back(new Tensor<T>{V,O});
    d_r2s.push_back(new Tensor<T>{V,V,O,O});
    d_t1s.push_back(new Tensor<T>{V,O});
    d_t2s.push_back(new Tensor<T>{V,V,O,O});
    Tensor<T>::allocate(ec,*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
 
  Tensor<T> d_r1{V,O};
  Tensor<T> d_r2{V,V,O,O};
  Tensor<T>::allocate(ec,d_r1, d_r2);

  Scheduler{ec}   
  (d_r1() = 0)
  (d_r2() = 0)
  .execute();

  double corr = 0;
  double residual = 0.0;
  double energy = 0.0;

  {
      auto lambda2 = [&](const IndexVector& blockid) {
          if(blockid[0] != blockid[1]) {
              Tensor<T> tensor     = d_f1;
              const TAMM_SIZE size = tensor.block_size(blockid);

              std::vector<T> buf(size);
              tensor.get(blockid, buf);

              auto block_dims   = tensor.block_dims(blockid);
              auto block_offset = tensor.block_offsets(blockid);

              TAMM_SIZE c = 0;
              for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
                  i++) {
                  for(auto j = block_offset[1];
                      j < block_offset[1] + block_dims[1]; j++, c++) {
                      buf[c] = 0;
                  }
              }
              d_f1.put(blockid, buf);
          }
      };
      block_for(ec->pg(), d_f1(), lambda2);
  }

  for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(ec, d_e, d_r1_residual, d_r2_residual);

          Scheduler{ec}(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
            .execute();

          Scheduler{ec}((*d_t1s[off])() = d_t1())((*d_t2s[off])() = d_t2())
            .execute();

          ccsd_e(*ec, MO, d_e, d_t1, d_t2, d_f1, d_v2);
          ccsd_t1(*ec, MO, d_r1, d_t1, d_t2, d_f1, d_v2);
          ccsd_t2(*ec, MO, d_r2, d_t1, d_t2, d_f1, d_v2);

          std::tie(residual, energy) = rest(*ec, MO, d_r1, d_r2, d_t1, d_t2,
                                            d_e, p_evl_sorted, zshiftl, noab);

          {
              auto lambdar2 = [&](const IndexVector& blockid) {
                  if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
                      Tensor<T> tensor     = d_r2;
                      const TAMM_SIZE size = tensor.block_size(blockid);

                      std::vector<T> buf(size);
                      tensor.get(blockid, buf);

                      auto block_dims   = tensor.block_dims(blockid);
                      auto block_offset = tensor.block_offsets(blockid);

                      TAMM_SIZE c = 0;
                      for(auto i = block_offset[0];
                          i < block_offset[0] + block_dims[0]; i++) {
                          for(auto j = block_offset[1];
                              j < block_offset[1] + block_dims[1]; j++) {
                              for(auto k = block_offset[2];
                                  k < block_offset[2] + block_dims[2]; k++) {
                                  for(auto l = block_offset[3];
                                      l < block_offset[3] + block_dims[3];
                                      l++, c++) {
                                      buf[c] = 0;
                                  }
                              }
                          }
                      }
                      d_r2.put(blockid, buf);
                  }
              };
              block_for(ec->pg(), d_r2(), lambdar2);
          }

          Scheduler{ec}((*d_r1s[off])() = d_r1())((*d_r2s[off])() = d_r2())
            .execute();

          iteration_print(ec->pg(), iter, residual, energy);
          Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

          if(residual < thresh) { break; }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec->pg().rank() == 0) {
          std::cout << " MICROCYCLE DIIS UPDATE:";
          std::cout.width(21);
          std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
          std::cout.width(21);
          std::cout << std::right << "5" << std::endl;
      }

      std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
      std::vector<std::vector<Tensor<T>*>*> ts{&d_t1s, &d_t2s};
      std::vector<Tensor<T>*> next_t{&d_t1, &d_t2};
      diis<T>(*ec, rs, ts, next_t);
  }

  if(ec->pg().rank() == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < thresh) {
        std::cout << " Iterations converged" << std::endl;
        std::cout.precision(15);
        std::cout << " CCSD correlation energy / hartree ="
                  << std::setw(26) << std::right << energy
                  << std::endl;
        std::cout << " CCSD total energy / hartree       ="
                  << std::setw(26) << std::right
                  << energy + hf_energy << std::endl;
    }
  }

  for(size_t i=0; i<ndiis; i++) {
    Tensor<T>::deallocate(*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
  d_r1s.clear();
  d_r2s.clear();
  Tensor<T>::deallocate(d_r1, d_r2);
}

template<typename T>
void eomccsd_x1(ExecutionContext& ec, const TiledIndexSpace& MO,
               Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2,
               const Tensor<T>& x1, const Tensor<T>& x2,
               const Tensor<T>& f1, const Tensor<T>& v2) {

   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");

   auto [h1, h3, h4, h5, h6, h8] = MO.labels<6>("occ");
   auto [p2, p3, p4, p5, p6, p7] = MO.labels<6>("virt");

   Tensor<T> i_1   {O, O};
   Tensor<T> i_1_1 {O, V};
   Tensor<T> i_2   {V, V};
   Tensor<T> i_3   {O, V};
   Tensor<T> i_4   {O, O, O, V};
   Tensor<T> i_5_1 {O, V};
   Tensor<T> i_5   {O, O};
   Tensor<T> i_5_2 {O, V};
   Tensor<T> i_6   {V, V};
   Tensor<T> i_7   {O, V};
   Tensor<T> i_8   {O, O, O, V};

   Scheduler sch{&ec};

   sch
     .allocate(i_1,i_1_1,i_2,i_3,i_4,i_5_1,i_5_2,i_5,i_6,i_7,i_8)
     ( i0(p2,h1)           =  0                                        )
     (   i_1(h6,h1)       +=        f1(h6,h1)                          )
     (     i_1_1(h6,p7)   +=        f1(h6,p7)                          )
     (     i_1_1(h6,p7)   +=        t1(p4,h5)       * v2(h5,h6,p4,p7)  )
     (   i_1(h6,h1)       +=        t1(p7,h1)       * i_1_1(h6,p7)     )
     (   i_1(h6,h1)       += -1   * t1(p3,h4)       * v2(h4,h6,h1,p3)  )
     (   i_1(h6,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h6,p3,p4)  )
     ( i0(p2,h1)          += -1   * x1(p2,h6)       * i_1(h6,h1)       )
     (   i_2(p2,p6)       +=        f1(p2,p6)                          )
     (   i_2(p2,p6)       +=        t1(p3,h4)       * v2(h4,p2,p3,p6)  )
     ( i0(p2,h1)          +=        x1(p6,h1)       * i_2(p2,p6)       )
     ( i0(p2,h1)          += -1   * x1(p4,h3)       * v2(h3,p2,h1,p4)  )
     (   i_3(h6,p7)       +=        f1(h6,p7)                          )
     (   i_3(h6,p7)       +=        t1(p3,h4)       * v2(h4,h6,p3,p7)  )
     ( i0(p2,h1)          +=        x2(p2,p7,h1,h6) * i_3(h6,p7)       )
     (   i_4(h6,h8,h1,p7) +=        v2(h6,h8,h1,p7)                    )
     (   i_4(h6,h8,h1,p7) +=        t1(p3,h1)       * v2(h6,h8,p3,p7)  )
     ( i0(p2,h1)          += -0.5 * x2(p2,p7,h6,h8) * i_4(h6,h8,h1,p7) )
     ( i0(p2,h1)          += -0.5 * x2(p4,p5,h1,h3) * v2(h3,p2,p4,p5)  )
     (     i_5_1(h8,p3)   +=        f1(h8,p3)                          )
     (     i_5_1(h8,p3)   += -1   * t1(p4,h5)       * v2(h5,h8,p3,p4)  )
     (   i_5(h8,h1)       +=        x1(p3,h1)       * i_5_1(h8,p3)     )
     (   i_5(h8,h1)       += -1   * x1(p5,h4)       * v2(h4,h8,h1,p5)  )
     (   i_5(h8,h1)       += -0.5 * x2(p5,p6,h1,h4) * v2(h4,h8,p5,p6)  )
     (     i_5_2(h8,p3)   += -1   * x1(p6,h5)       * v2(h5,h8,p3,p6)  )
     (   i_5(h8,h1)       +=        t1(p3,h1)       * i_5_2(h8,p3)     )
     ( i0(p2,h1)          += -1   * t1(p2,h8)       * i_5(h8,h1)       )
     (   i_6(p2,p3)       +=        x1(p5,h4)       * v2(h4,p2,p3,p5)  )
     ( i0(p2,h1)          += -1   * t1(p3,h1)       * i_6(p2,p3)       )
     (   i_7(h4,p3)       +=        x1(p6,h5)       * v2(h4,h5,p3,p6)  )
     ( i0(p2,h1)          +=        t2(p2,p3,h1,h4) * i_7(h4,p3)       )
     (   i_8(h4,h5,h1,p3) +=        x1(p6,h1)       * v2(h4,h5,p3,p6)  )
     ( i0(p2,h1)          +=  0.5 * t2(p2,p3,h4,h5) * i_8(h4,h5,h1,p3) )
     .deallocate(i_1,i_1_1,i_2,i_3,i_4,i_5_1,i_5,i_5_2,
                 i_6,i_7,i_8).execute();
}

template<typename T>
void eomccsd_x2(ExecutionContext& ec, const TiledIndexSpace& MO,
               Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2,
               const Tensor<T>& x1, const Tensor<T>& x2,
               const Tensor<T>& f1, const Tensor<T>& v2) {

   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");

   TiledIndexLabel h1, h2, h5, h6, h7, h8, h9, h10;
   TiledIndexLabel p3, p4, p5, p6, p7, p8, p9;

   std::tie(h1, h2, h5, h6, h7, h8, h9, h10) = MO.labels<8>("occ");
   std::tie(p3, p4, p5, p6, p7, p8, p9) = MO.labels<7>("virt");

   Tensor<T> i_1     {O, V, O, O};
   Tensor<T> i_1_1   {O, V, O, V};
   Tensor<T> i_1_2   {O, V};
   Tensor<T> i_1_3   {O, O, O, V};
   Tensor<T> i_2     {O, O};
   Tensor<T> i_2_1   {O, V};
   Tensor<T> i_3     {V, V};
   Tensor<T> i_4     {O, O, O, O};
   Tensor<T> i_4_1   {O, O, O, V};
   Tensor<T> i_5     {O, V, O, V};
   Tensor<T> i_6_1   {O, O, O, O};
   Tensor<T> i_6_1_1 {O, O, O, V};
   Tensor<T> i_6     {O, V, O, O};
   Tensor<T> i_6_2   {O, V};
   Tensor<T> i_6_3   {O, O, O, V};
   Tensor<T> i_6_4   {O, O, O, O};
   Tensor<T> i_6_4_1 {O, O, O, V};
   Tensor<T> i_6_5   {O, V, O, V};
   Tensor<T> i_6_6   {O, V};
   Tensor<T> i_6_7   {O, O, O, V};
   Tensor<T> i_7     {V, V, O, V};
   Tensor<T> i_8_1   {O, V};
   Tensor<T> i_8     {O, O};
   Tensor<T> i_8_2   {O, V};
   Tensor<T> i_9     {O, O, O, O};
   Tensor<T> i_9_1   {O, O, O, V};
   Tensor<T> i_10    {V, V};
   Tensor<T> i_11    {O, V, O, V};

   Scheduler sch{&ec};

   sch
     .allocate(i_1,i_1_1,i_1_2,i_1_3,i_2,i_2_1,i_3,i_4,i_4_1,i_5,
               i_6_1,i_6_1_1,i_6,i_6_2,i_6_3,i_6_4,i_6_4_1,i_6_5,
               i_6_6,i_6_7,i_7,i_8_1,i_8,i_8_2,i_9,i_9_1,i_10,i_11)
     ( i0(p4,p3,h1,h2)              =  0                                               )
     (   i_1(h9,p3,h1,h2)          +=         v2(h9,p3,h1,h2)                          )
     (     i_1_1(h9,p3,h1,p5)      +=         v2(h9,p3,h1,p5)                          )
     (     i_1_1(h9,p3,h1,p5)      += -0.5  * t1(p6,h1)        * v2(h9,p3,p5,p6)       )
     (   i_1(h9,p3,h1,h2)          += -1    * t1(p5,h1)        * i_1_1(h9,p3,h2,p5)    )
     (   i_1(h9,p3,h2,h1)          +=         t1(p5,h1)        * i_1_1(h9,p3,h2,p5)    ) //P(h1/h2)
     (     i_1_2(h9,p8)            +=         f1(h9,p8)                                )
     (     i_1_2(h9,p8)            +=         t1(p6,h7)        * v2(h7,h9,p6,p8)       )
     (   i_1(h9,p3,h1,h2)          += -1    * t2(p3,p8,h1,h2)  * i_1_2(h9,p8)          )
     (     i_1_3(h6,h9,h1,p5)      +=         v2(h6,h9,h1,p5)                          )
     (     i_1_3(h6,h9,h1,p5)      += -1    * t1(p7,h1)        * v2(h6,h9,p5,p7)       )
     (   i_1(h9,p3,h1,h2)          +=         t2(p3,p5,h1,h6)  * i_1_3(h6,h9,h2,p5)    )
     (   i_1(h9,p3,h2,h1)          += -1    * t2(p3,p5,h1,h6)  * i_1_3(h6,h9,h2,p5)    ) //P(h1/h2)
     (   i_1(h9,p3,h1,h2)          +=  0.5  * t2(p5,p6,h1,h2)  * v2(h9,p3,p5,p6)       )
     ( i0(p3,p4,h1,h2)             += -1    * x1(p3,h9)        * i_1(h9,p4,h1,h2)      )
     ( i0(p4,p3,h1,h2)             +=         x1(p3,h9)        * i_1(h9,p4,h1,h2)      ) //P(p3/p4)
     ( i0(p3,p4,h1,h2)             += -1    * x1(p5,h1)        * v2(p3,p4,h2,p5)       )
     ( i0(p3,p4,h2,h1)             +=         x1(p5,h1)        * v2(p3,p4,h2,p5)       ) //P(h1/h2)
     (   i_2(h8,h1)                +=         f1(h8,h1)                                )
     (     i_2_1(h8,p9)            +=         f1(h8,p9)                                )
     (     i_2_1(h8,p9)            +=         t1(p6,h7)        * v2(h7,h8,p6,p9)       )
     (   i_2(h8,h1)                +=         t1(p9,h1)        * i_2_1(h8,p9)          )
     (   i_2(h8,h1)                += -1    * t1(p5,h6)        * v2(h6,h8,h1,p5)       )
     (   i_2(h8,h1)                += -0.5  * t2(p5,p6,h1,h7)  * v2(h7,h8,p5,p6)       )
     ( i0(p3,p4,h1,h2)             += -1    * x2(p3,p4,h1,h8)  * i_2(h8,h2)            )
     ( i0(p3,p4,h2,h1)             +=         x2(p3,p4,h1,h8)  * i_2(h8,h2)            ) //P(h1/h2)
     (   i_3(p3,p8)                +=         f1(p3,p8)                                )
     (   i_3(p3,p8)                +=         t1(p5,h6)        * v2(h6,p3,p5,p8)       )
     (   i_3(p3,p8)                +=  0.5  * t2(p3,p5,h6,h7)  * v2(h6,h7,p5,p8)       )
     ( i0(p3,p4,h1,h2)             +=         x2(p3,p8,h1,h2)  * i_3(p4,p8)            )
     ( i0(p4,p3,h1,h2)             += -1    * x2(p3,p8,h1,h2)  * i_3(p4,p8)            ) //P(p3/p4)
     (   i_4(h9,h10,h1,h2)         +=         v2(h9,h10,h1,h2)                         )
     (     i_4_1(h9,h10,h1,p5)     +=         v2(h9,h10,h1,p5)                         )
     (     i_4_1(h9,h10,h1,p5)     += -0.5  * t1(p6,h1)        * v2(h9,h10,p5,p6)      )
     (   i_4(h9,h10,h1,h2)         += -1    * t1(p5,h1)        * i_4_1(h9,h10,h2,p5)   )
     (   i_4(h9,h10,h2,h1)         +=         t1(p5,h1)        * i_4_1(h9,h10,h2,p5)   ) //P(h1/h2)
     (   i_4(h9,h10,h1,h2)         +=  0.5  * t2(p5,p6,h1,h2)  * v2(h9,h10,p5,p6)      )
     ( i0(p3,p4,h1,h2)             +=  0.5  * x2(p3,p4,h9,h10) * i_4(h9,h10,h1,h2)     )
     (   i_5(h7,p3,h1,p8)          +=         v2(h7,p3,h1,p8)                          )
     (   i_5(h7,p3,h1,p8)          +=         t1(p5,h1)        * v2(h7,p3,p5,p8)       )
     ( i0(p3,p4,h1,h2)             += -1    * x2(p3,p8,h1,h7)  * i_5(h7,p4,h2,p8)      )
     ( i0(p3,p4,h2,h1)             +=         x2(p3,p8,h1,h7)  * i_5(h7,p4,h2,p8)      ) //P(h1/h2)
     ( i0(p4,p3,h1,h2)             +=         x2(p3,p8,h1,h7)  * i_5(h7,p4,h2,p8)      ) //P(p3/p4)
     ( i0(p4,p3,h2,h1)             += -1    * x2(p3,p8,h1,h7)  * i_5(h7,p4,h2,p8)      ) //P(h1/h2,p3/p4)
     ( i0(p3,p4,h1,h2)             +=  0.5  * x2(p5,p6,h1,h2)  * v2(p3,p4,p5,p6)       )
     (     i_6_1(h8,h10,h1,h2)     +=         v2(h8,h10,h1,h2)                         )
     (       i_6_1_1(h8,h10,h1,p5) +=         v2(h8,h10,h1,p5)                         )
     (       i_6_1_1(h8,h10,h1,p5) += -0.5  * t1(p6,h1)        * v2(h8,h10,p5,p6)      )
     (     i_6_1(h8,h10,h1,h2)     += -1    * t1(p5,h1)        * i_6_1_1(h8,h10,h2,p5) )
     (     i_6_1(h8,h10,h2,h1)     +=         t1(p5,h1)        * i_6_1_1(h8,h10,h2,p5) ) //P(h1/h2)
     (     i_6_1(h8,h10,h1,h2)     +=  0.5  * t2(p5,p6,h1,h2)  * v2(h8,h10,p5,p6)      )
     (   i_6(h10,p3,h1,h2)         += -1    * x1(p3,h8)        * i_6_1(h8,h10,h1,h2)   )
     (   i_6(h10,p3,h1,h2)         +=         x1(p6,h1)        * v2(h10,p3,h2,p6)      )
     (   i_6(h10,p3,h2,h1)         += -1    * x1(p6,h1)        * v2(h10,p3,h2,p6)      ) //P(h1/h2)
     (     i_6_2(h10,p5)           +=         f1(h10,p5)                               )
     (     i_6_2(h10,p5)           += -1    * t1(p6,h7)        * v2(h7,h10,p5,p6)      )
     (   i_6(h10,p3,h1,h2)         +=         x2(p3,p5,h1,h2)  * i_6_2(h10,p5)         )
     (     i_6_3(h8,h10,h1,p9)     +=         v2(h8,h10,h1,p9)                         )
     (     i_6_3(h8,h10,h1,p9)     +=         t1(p5,h1)        * v2(h8,h10,p5,p9)      )
     (   i_6(h10,p3,h1,h2)         += -1    * x2(p3,p9,h1,h8)  * i_6_3(h8,h10,h2,p9)   )
     (   i_6(h10,p3,h2,h1)         +=         x2(p3,p9,h1,h8)  * i_6_3(h8,h10,h2,p9)   ) //P(h1/h2)
     (   i_6(h10,p3,h1,h2)         += -0.5  * x2(p6,p7,h1,h2)  * v2(h10,p3,p6,p7)      )
     (     i_6_4(h9,h10,h1,h2)     +=  0.5  * x1(p7,h1)        * v2(h9,h10,h2,p7)      )
     (     i_6_4(h9,h10,h2,h1)     += -0.5  * x1(p7,h1)        * v2(h9,h10,h2,p7)      ) //P(h1/h2)
     (     i_6_4(h9,h10,h1,h2)     += -0.25 * x2(p7,p8,h1,h2)  * v2(h9,h10,p7,p8)      )
     (       i_6_4_1(h9,h10,h1,p5) += -1    * x1(p8,h1)        * v2(h9,h10,p5,p8)      )
     (     i_6_4(h9,h10,h1,h2)     +=  0.5  * t1(p5,h1)        * i_6_4_1(h9,h10,h2,p5) )
     (     i_6_4(h9,h10,h2,h1)     += -0.5  * t1(p5,h1)        * i_6_4_1(h9,h10,h2,p5) ) //P(h1/h2)
     (   i_6(h10,p3,h1,h2)         +=         t1(p3,h9)        * i_6_4(h9,h10,h1,h2)   )
     (     i_6_5(h10,p3,h1,p5)     +=         x1(p7,h1)        * v2(h10,p3,p5,p7)      )
     (   i_6(h10,p3,h1,h2)         += -1    * t1(p5,h1)        * i_6_5(h10,p3,h2,p5)   )
     (   i_6(h10,p3,h2,h1)         +=         t1(p5,h1)        * i_6_5(h10,p3,h2,p5)   ) //P(h1/h2)
     (     i_6_6(h10,p5)           += -1    * x1(p8,h7)        * v2(h7,h10,p5,p8)      )
     (   i_6(h10,p3,h1,h2)         +=         t2(p3,p5,h1,h2)  * i_6_6(h10,p5)         )
     (     i_6_7(h6,h10,h1,p5)     +=         x1(p8,h1)        * v2(h6,h10,p5,p8)      )
     (   i_6(h10,p3,h1,h2)         +=         t2(p3,p5,h1,h6)  * i_6_7(h6,h10,h2,p5)   )
     (   i_6(h10,p3,h2,h1)         += -1    * t2(p3,p5,h1,h6)  * i_6_7(h6,h10,h2,p5)   ) //P(h1/h2)
     ( i0(p3,p4,h1,h2)             +=         t1(p3,h10)       * i_6(h10,p4,h1,h2)     )
     ( i0(p4,p3,h1,h2)             += -1    * t1(p3,h10)       * i_6(h10,p4,h1,h2)     ) //P(p3/p4)
     (   i_7(p3,p4,h1,p5)          +=         x1(p6,h1)        * v2(p3,p4,p5,p6)       )
     ( i0(p3,p4,h1,h2)             +=         t1(p5,h1)        * i_7(p3,p4,h2,p5)      )
     ( i0(p3,p4,h2,h1)             += -1    * t1(p5,h1)        * i_7(p3,p4,h2,p5)      ) //P(h1/h2)
     (     i_8_1(h5,p9)            +=         f1(h5,p9)                                )
     (     i_8_1(h5,p9)            += -1    * t1(p6,h7)        * v2(h5,h7,p6,p9)       )
     (   i_8(h5,h1)                +=         x1(p9,h1)        * i_8_1(h5,p9)          )
     (   i_8(h5,h1)                +=         x1(p7,h6)        * v2(h5,h6,h1,p7)       )
     (   i_8(h5,h1)                +=  0.5  * x2(p7,p8,h1,h6)  * v2(h5,h6,p7,p8)       )
     (     i_8_2(h5,p6)            +=         x1(p8,h7)        * v2(h5,h7,p6,p8)       )
     (   i_8(h5,h1)                +=         t1(p6,h1)        * i_8_2(h5,p6)          )
     ( i0(p3,p4,h1,h2)             += -1    * t2(p3,p4,h1,h5)  * i_8(h5,h2)            )
     ( i0(p3,p4,h2,h1)             +=         t2(p3,p4,h1,h5)  * i_8(h5,h2)            ) //P(h1/h2)
     (   i_9(h5,h6,h1,h2)          += -0.5  * x1(p7,h1)        * v2(h5,h6,h2,p7)       )
     (   i_9(h5,h6,h2,h1)          +=  0.5  * x1(p7,h1)        * v2(h5,h6,h2,p7)       ) //P(h1/h2)
     (   i_9(h5,h6,h1,h2)          +=  0.25 * x2(p7,p8,h1,h2)  * v2(h5,h6,p7,p8)       )
     (     i_9_1(h5,h6,h1,p7)      +=         x1(p8,h1)        * v2(h5,h6,p7,p8)       )
     (   i_9(h5,h6,h1,h2)          +=  0.5  * t1(p7,h1)        * i_9_1(h5,h6,h2,p7)    )
     (   i_9(h5,h6,h2,h1)          += -0.5  * t1(p7,h1)        * i_9_1(h5,h6,h2,p7)    ) //P(h1/h2)
     ( i0(p3,p4,h1,h2)             +=         t2(p3,p4,h5,h6)  * i_9(h5,h6,h1,h2)      )
     (   i_10(p3,p5)               +=         x1(p7,h6)        * v2(h6,p3,p5,p7)       )
     (   i_10(p3,p5)               +=  0.5  * x2(p3,p8,h6,h7)  * v2(h6,h7,p5,p8)       )
     ( i0(p3,p4,h1,h2)             += -1    * t2(p3,p5,h1,h2)  * i_10(p4,p5)           )
     ( i0(p4,p3,h1,h2)             +=         t2(p3,p5,h1,h2)  * i_10(p4,p5)           ) //P(p3/p4)
     (   i_11(h6,p3,h1,p5)         +=         x1(p7,h1)        * v2(h6,p3,p5,p7)       )
     (   i_11(h6,p3,h1,p5)         +=         x2(p3,p8,h1,h7)  * v2(h6,h7,p5,p8)       )
     ( i0(p3,p4,h1,h2)             +=         t2(p3,p5,h1,h6)  * i_11(h6,p4,h2,p5)     )
     ( i0(p3,p4,h2,h1)             += -1    * t2(p3,p5,h1,h6)  * i_11(h6,p4,h2,p5)     ) //P(h1/h2)
     ( i0(p4,p3,h1,h2)             += -1    * t2(p3,p5,h1,h6)  * i_11(h6,p4,h2,p5)     ) //P(p3/p4)
     ( i0(p4,p3,h2,h1)             +=         t2(p3,p5,h1,h6)  * i_11(h6,p4,h2,p5)     ) //P(h1/h2,p3/p4)
     .deallocate(i_1,i_1_1,i_1_2,i_1_3,i_2,i_2_1,i_3,i_4,i_4_1,i_5,
                 i_6_1,i_6_1_1,i_6,i_6_2,i_6_3,i_6_4,i_6_4_1,i_6_5,
                 i_6_6,i_6_7,i_7,i_8_1,i_8,i_8_2,i_9,i_9_1,i_10,i_11).execute();
}

template<typename T>
std::vector<size_t> sort_indexes(std::vector<T>& v){
    std::vector<size_t> idx(v.size());
    iota(idx.begin(),idx.end(),0);
    sort(idx.begin(),idx.end(),[&v](size_t x, size_t y) {return v[x] < v[y];});

    return idx;
}

template<typename T>
void eomccsd_driver(ExecutionContext* ec, const TiledIndexSpace& MO,
                   Tensor<T>& t1, Tensor<T>& t2,
                   Tensor<T>& f1, Tensor<T>& v2,
//                   Tensor<T>& d_x1, Tensor<T>& d_x2,
                   int nroots, int maxeomiter,
                   double eomthresh, int microeomiter,
                   long int total_orbitals, const TAMM_SIZE& noab) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    std::cout.precision(15);

    Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  std::cout << "Total orbitals = " << total_orbitals << std::endl;

  std::vector<double> p_evl_sorted = f1.diagonal();

  if(ec->pg().rank() == 0) {
    std::cout << "p_evl_sorted:" << '\n';
    for(size_t p = 0; p < p_evl_sorted.size(); p++)
      std::cout << p_evl_sorted[p] << '\n';
  }

//For jacobi step.
  double zshiftl = 0;
  bool transpose=false;
  const auto hbardim = nroots*microeomiter;
  const auto microdim = nroots;

Matrix hbar = Matrix::Zero(hbardim,hbardim);

auto populate_vector_of_tensors = [&] (std::vector<Tensor<T>> &vec, bool is2D=true){
    for (auto x=0;x<vec.size();x++){
        if(is2D) vec[x] = Tensor<T>{V,O};
        else vec[x] = Tensor<T>{V,V,O,O};

        Tensor<T>::allocate(ec,vec[x]);
    }
};



using std::vector;

  vector<Tensor<T>> x1(hbardim);
  populate_vector_of_tensors(x1);
  vector<Tensor<T>> x2(hbardim);
  populate_vector_of_tensors(x2,false);
  vector<Tensor<T>> xp1(hbardim);
  populate_vector_of_tensors(xp1);
  vector<Tensor<T>> xp2(hbardim);
  populate_vector_of_tensors(xp2,false);
  vector<Tensor<T>> xc1(microdim);
  populate_vector_of_tensors(xc1);
  vector<Tensor<T>> xc2(microdim);
  populate_vector_of_tensors(xc2,false);
  vector<Tensor<T>> r1(microdim);
  populate_vector_of_tensors(r1);
  vector<Tensor<T>> r2(microdim);
  populate_vector_of_tensors(r2,false);
  
//TO DO: NXTRIALS IS SET TO NROOTS BECAUSE WE ONLY ALLOW #NROOTS# INITIAL GUESSES
//       WHEN THE EOM_GUESS ROUTINE IS UPDATED TO BE LIKE THE INITIAL GUESS IN TCE
//       THEN THE FOLLOWING LINE WILL BE "INT NXTRIALS = INITVECS", WHERE INITVEC 
//       IS THE NUMBER OF INITIAL VECTORS. THE {X1,X2} AND {XP1,XP2} TENSORS WILL 
//       BE OF DIMENSION (INITVECS + NROOTS*(MICROEOMITER-1)) FOR THE FIRST 
//       MICROCYCLE AND (NROOTS*MICROEOMITER) FOR THE REMAINING.
  int nxtrials = nroots; 

//################################################################################
//  CALL THE EOM_GUESS ROUTINE (EXTERNAL ROUTINE)
//################################################################################
  eom_guess(nroots,noab,p_evl_sorted,x1);

//################################################################################
//  PRINT THE HEADER FOR THE EOM ITERATIONS
//################################################################################
  if(ec->pg().rank() == 0) {
    std::cout << "\n\n";
//TO DO: THE NUMBER OF INITIAL VECTORS WILL BE DETERMINED IN THE EOM_GUESS ROUTINE.
    std::cout << " No. of initial right vectors " << nxtrials << std::endl;
    std::cout << "\n";
    std::cout << " EOM-CCSD right-hand side iterations" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    std::cout <<
        "     Residuum       Omega / hartree  Omega / eV    Cpu    Wall"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;
  }

  Tensor<T> d_r1{};
  Tensor<T> oscalar{};
  Tensor<T>::allocate(ec, d_r1, oscalar);

  for(int iter = 0; iter < maxeomiter;){
     for(int micro = 0; micro < microeomiter; iter++, micro++){
std::cout<< "######################## 'Iter'ation(+1) " << iter+1 << "#############" << std::endl; 
        nxtrials = (micro+1)*nroots;
        for(int root= 0; root < nroots; root++){

        auto counter = nxtrials-nroots+root;

//std::cout << "@@@@@@@@@@@@@ X1 @@@@@@@@@@@@@" << std::endl;
//        print_tensor_all(x1.at(counter));
//std::cout << "@@@@@@@@@@@@@ X2 @@@@@@@@@@@@@" << std::endl;
//        print_tensor(x2.at(counter));
        eomccsd_x1(*ec, MO, xp1.at(counter), t1, t2, x1.at(counter), x2.at(counter), f1, v2);
        eomccsd_x2(*ec, MO, xp2.at(counter), t1, t2, x1.at(counter), x2.at(counter), f1, v2);
//std::cout << "@@@@@@@@@@@@@ XP1 @@@@@@@@@@@@@" << std::endl;
//        print_tensor_all(xp1.at(counter));
//std::cout << "@@@@@@@@@@@@@ XP2 @@@@@@@@@@@@@" << std::endl;
//        print_tensor(xp2.at(counter));

        std::cout << iter << " " << micro << " COUNTER  " << counter << std::endl;
        }

//***Update hbar which is a matrix of dot products between the x and xp vectors.
           if(micro == 0){
           std::cout << "HERE" << std::endl;
              for(int ivec = 0; ivec < nroots; ivec++){
                 for(int jvec = 0; jvec < nroots; jvec++){
                 std::cout << "PRODUCT " << ivec << " " << jvec << std::endl;
//################################################################################
               sch(d_r1()  = 0) 
                  (d_r1() += xp1.at(ivec)() * x1.at(jvec)()) 
                  (d_r1() += xp2.at(ivec)() * x2.at(jvec)()).execute();
               T r1;
               d_r1.get({}, {&r1, 1});
               hbar(ivec,jvec) = r1;
//           std::cout << hbar(ivec,jvec) << std::endl;
//################################################################################
                 }
              }
           } else {
              for(int ivec = 0; ivec < nxtrials; ivec++){
                 for(int jvec = micro*nroots; jvec < nxtrials; jvec++){
                 std::cout << "PRODUCT " << ivec << " " << jvec << std::endl;
//################################################################################
               sch(d_r1()  = 0) 
                  (d_r1() += xp1.at(ivec)() * x1.at(jvec)()) 
                  (d_r1() += xp2.at(ivec)() * x2.at(jvec)()).execute();
               T r1;
               d_r1.get({}, {&r1, 1});
               hbar(ivec,jvec) = r1;
//################################################################################
                 }
              }
              for(int ivec = micro*nroots; ivec < nxtrials; ivec++){
                 for(int jvec = 0; jvec < micro*nroots; jvec++){
                 std::cout << "PRODUCT " << ivec << " " << jvec << std::endl;
//################################################################################
                sch(d_r1()  = 0) 
                  (d_r1() += xp1.at(ivec)() * x1.at(jvec)()) 
                  (d_r1() += xp2.at(ivec)() * x2.at(jvec)()).execute();
               T r1;
               d_r1.get({}, {&r1, 1});
               hbar(ivec,jvec) = r1;
//################################################################################
                 }
              }
           } 
//           std::cout << hbar << std::endl;

#if 1

Eigen::EigenSolver<Matrix> hbardiag(hbar.block(0,0,nxtrials,nxtrials));
auto omegar1 = hbardiag.eigenvalues();

const auto nev = omegar1.rows();
std::vector<T> omegar(nev);
for (auto x=0; x<nev;x++)
 omegar[x] = real(omegar1(x));

std::cout << omegar << std::endl;

//################################################################################
//Sort the eigenvectors and corresponding eigenvalues 
//################################################################################

std::vector<size_t> omegar_sorted_order = sort_indexes(omegar);
std::sort(omegar.begin(), omegar.end());

std::cout << "Sorted eigenvalues" << std::endl;
std::cout << omegar << std::endl;

// for (auto x: omegar) std::cout << x << " ";
// std::cout << std::endl;
// for (auto x: omegar_sorted_order) std::cout << x << " ";
// std::cout << std::endl;

auto hbar_right1 = hbardiag.eigenvectors();
assert(hbar_right1.rows() == nev && hbar_right1.cols() == nev);
Matrix hbar_right(nev,nev);
hbar_right.setZero();

for (auto x=0;x<nev;x++)
    hbar_right.col(x) = hbar_right1.col(omegar_sorted_order[x]).real();

std::cout << hbar_right1.real() << std::endl;
std::cout << "Sorted eigenvectors" << std::endl;
std::cout << hbar_right << std::endl;

//################################################################################
//--From the lowest nroots number of eigenvalues and vectors, form xc's which 
//  takes each of the vectors and uses them to form a linear combination of x 
//  vectors. xc(k-th)=sum of scalar element i of the k-th vector times x(i)-th vector
//--Then times these by -omega (eigenvalue). Place into r1/2's.
//--Then do the same linear combination trick with xp vectors instead of x's. Add to r1/2's.
//**** This was previously done with daxpy blas routine
//
 for(auto root = 0; root < nroots; root++){
    sch(xc1.at(root)()       = 0)
    (xc2.at(root)() = 0) 
    (r1.at(root)()        = 0)
    (r2.at(root)()        = 0).execute();
    for(int i = 0; i < nxtrials; i++){
//    std::cout << i << hbar_right(i,root) << std::endl;
        T hbr_scalar = hbar_right(i,root);
       sch(xc1.at(root)()       += hbr_scalar * x1.at(i)()) 
          (xc2.at(root)() += hbr_scalar * x2.at(i)()).execute();
    }  
 }
//
 for(int root = 0; root < nroots; root++){
    T omegar_scalar = -1 * omegar[root];
    sch(r1.at(root)()        += omegar_scalar * xc1.at(root)() )
    (r2.at(root)() += omegar_scalar * xc2.at(root)() ).execute();
//    std::cout << "xc1" << std::endl;
//    print_tensor_all(xc1.at(root));
//    std::cout << "omega*xc1" << std::endl;
//    print_tensor_all(r1.at(root));
    for(int i = 0; i < nxtrials; i++){
        T hbr_scalar = hbar_right(i,root);
       sch(r1.at(root)()        += hbr_scalar * xp1.at(i)())
       (r2.at(root)() += hbr_scalar *xp2.at(i)()).execute();
    }  
//                sch(d_r1()  = 0)
//                  (d_r1() += r1.at(root)() * r1.at(root)())
//                  (d_r1() += r2.at(root)() * r2.at(root)()).execute();
// std::cout << "Residual^^2= " << std::endl;
// print_tensor(d_r1);
 }
 #endif
//################################################################################


//################################################################################
//***Call jacobi with the r1/r2's to form the new set of x1/x2's
 std::cout << "nxtrials before jacobi = " << nxtrials << std::endl;
 for(auto root = 0; root < nroots; root++){
// std::cout << "  BEFORE JACOBI   r1.at(root)()= at root "<< root << std::endl;
// print_tensor(r1.at(root));
// std::cout << "  BEFORE JACOBI   r2.at(root)()= at root "<< root << std::endl;
// print_tensor(r2.at(root));
     jacobi(*ec, r1.at(root), x1.at(nxtrials+root), 0.0, false, p_evl_sorted, noab);
     jacobi(*ec, r2.at(root), x2.at(nxtrials+root), 0.0, false, p_evl_sorted, noab);
// std::cout << "  AFTER JACOBI     x1.at(nxtrials+root) at root "<< root << std::endl;
// print_tensor(x1.at(nxtrials+root));
// std::cout << "  AFTER JACOBI     x2.at(nxtrials+root) at root "<< root << std::endl;
// print_tensor(x2.at(nxtrials+root));
 }
//
// FUTURE: Thee will be a specific Jacobi for x's which accounts for symmetry
//################################################################################

// ORTHOGONALIZATION and NORMALIZATION
//Right now u1 and u2 are allocated to be seperate. It is a seperate array to only 
//store and work with the vector that is currently being orthoginalized. 
//To save some memory, you can use any one of the xc vectors as a workspace.  
//***ACTUALLY just have r1 and r2 overwrite themselves in jacobi!!!!!!!!!!!!!!!!!!
   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");

   auto [h1, h2] = MO.labels<2>("occ");
   auto [p3, p4] = MO.labels<2>("virt");

   std::cout << "IN ORTHOGONALIZATION, MICRO = " << micro << std::endl;
   //if(micro > 0){
      for(int ivec = nxtrials; ivec<nxtrials+nroots; ivec++){
          Tensor<T> u1 = x1.at(ivec);
          Tensor<T> u2 = x2.at(ivec);
//          std::cout << "X1 BEFORE REMOVING OVERLAP"<< ivec << std::endl;
//          print_tensor(u1);
//          std::cout << "X2 BEFORE REMOVING OVERLAP"<< ivec << std::endl;
//          print_tensor(u2);
          for(int jvec = 0; jvec<ivec; jvec++){
//             std::cout << ivec << " " << jvec << std::endl;
//          std::cout << "X1 Entering orthogonalization" << std::endl;
//          std::cout << "X1 at ivec"<< ivec << std::endl;
//          print_tensor_all(x1.at(ivec));
//          std::cout << "X1 at jvec"<< jvec << std::endl;
//          print_tensor_all(x1.at(jvec));
             sch(oscalar() = 0)
                (oscalar() += x1.at(ivec)() * x1.at(jvec)())
                (oscalar() += x2.at(ivec)() * x2.at(jvec)()).execute();
              T tmps;
              oscalar.get({}, {&tmps,1});
//         std::cout << "TMPS 1= " << tmps << std::endl;
              sch(u1() += -1 * tmps * x1.at(jvec)())
                 (u2() += -1 * tmps * x2.at(jvec)()).execute();
          }
   //@@@@@@@@@@@@@@@ START FIX ME @@@@@@@@@@@@@@@@@@@@
   std::cout << "THIS IS FOR IVEC = "<< ivec << std::endl;
   std::cout << "X1 (U1) AFTER REMOVING OVERLAP" << std::endl;
   print_tensor(u1);
   std::cout << "X2 (U1) AFTER REMOVING OVERLAP" << std::endl;
   print_tensor(u2);
          sch(oscalar() = 0)
             (oscalar() += u1() * u1())
             (oscalar() += u2() * u2()).execute();
//          T tmps;
//          oscalar.get({}, {&tmps,1});
          T tmps = get_scalar(oscalar);
   std::cout << "TMPS 2= " << tmps << std::endl;
          T newsc = 1/sqrt(tmps);
   std::cout << "SCALING= " << newsc << std::endl;
          sch(x1.at(ivec)(p3,h1) = 0)
             (x2.at(ivec)(p4,p3,h1,h2) = 0)
             (x1.at(ivec)(p3,h1) += newsc * u1(p3,h1))
             (x2.at(ivec)(p4,p3,h1,h2) += newsc * u2(p4,p3,h1,h2)).execute(); 
//             (x1.at(ivec)() += newsc * u1())
//             (x2.at(ivec)() += newsc * u2()).execute(); 
   std::cout << "X1 AFTER NORMALIZATION" << std::endl;
   print_tensor(x1.at(ivec));
   std::cout << "X2 AFTER NORMALIZATION" << std::endl;
   print_tensor(x2.at(ivec));
   //@@@@@@@@@@@@@@@ END FIX ME @@@@@@@@@@@@@@@@@@@@
      }
   //}

 

     } //end micro

//################################################################################
//*** When microeomiter number of iteration is met copy the last xc vectors 
//*** to the first set of x vectors, 
  std::cout << "COLLAPSE" << std::endl;
//THIS NEEDS TO BE TIED IN WITH THE LAST ROUTINE SO THAT THE LATEST ORTHONORMAL SET 
//IS USED AS THE INITIAL GUESS.
//    for(auto root = 0; root < nroots; root++){
//       sch(x1.at(root)() = xc1.at(((microeomiter-1)*nroots)+root)() )
//          (x2.at(root)() = xc2.at(((microeomiter-1)*nroots)+root)() ).execute();
//    }
//################################################################################

  }

  Tensor<T>::deallocate(d_r1);
  Tensor<T>::deallocate(oscalar);

  auto deallocate_vtensors = [&](auto&&... vecx) {
      (std::for_each(vecx.begin(), vecx.end(), [](auto& t) { t.deallocate(); }),
       ...);
  };
  deallocate_vtensors(x1, x2, xp1, xp2, xc1, xc2, r1, r2);

  //   Tensor<T>::deallocate(hbar);

}

std::string filename; //bad, but no choice
int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!\n";
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        return 1;
    }

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run();
    
    GA_Terminate();
    MPI_Finalize();

    return res;
}


TEST_CASE("CCSD Driver") {

    std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    Matrix C;
    Matrix F;
    Tensor4D V2;
    TAMM_SIZE ov_alpha{0};
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    double hf_energy{0.0};
    libint2::BasisSet shells;
    TAMM_SIZE nao{0};

    std::vector<TAMM_SIZE> sizes;

    auto hf_t1 = std::chrono::high_resolution_clock::now();
    std::tie(ov_alpha, nao, hf_energy, shells) = hartree_fock(filename, C, F);
    auto hf_t2 = std::chrono::high_resolution_clock::now();

   double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    std::cout << "\nTime taken for Hartree-Fock: " << hf_time << " secs\n";

    hf_t1        = std::chrono::high_resolution_clock::now();
    std::tie(V2) = four_index_transform(ov_alpha, nao, freeze_core,
                                        freeze_virtual, C, F, shells);
    hf_t2        = std::chrono::high_resolution_clock::now();
    double two_4index_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    std::cout << "\nTime taken for 4-index transform: " << two_4index_time
              << " secs\n";

    TAMM_SIZE ov_beta{nao - ov_alpha};

    std::cout << "ov_alpha,nao === " << ov_alpha << ":" << nao << std::endl;
    sizes = {ov_alpha - freeze_core, ov_alpha - freeze_core,
             ov_beta - freeze_virtual, ov_beta - freeze_virtual};

    std::cout << "sizes vector -- \n";
    for(const auto& x : sizes) std::cout << x << ", ";
    std::cout << "\n";

    const long int total_orbitals = 2*ov_alpha+2*ov_beta;
    
    // Construction of tiled index space MO

    IndexSpace MO_IS{range(0, total_orbitals),
                    {{"occ", {range(0, 2*ov_alpha)}},
                     {"virt", {range(2*ov_alpha, total_orbitals)}}}};

    // IndexSpace MO_IS{range(0, total_orbitals),
    //                 {{"occ", {range(0, ov_alpha+ov_beta)}}, //0-7
    //                  {"virt", {range(total_orbitals/2, total_orbitals)}}, //7-14
    //                  {"alpha", {range(0, ov_alpha),range(ov_alpha+ov_beta,2*ov_alpha+ov_beta)}}, //0-5,7-12
    //                  {"beta", {range(ov_alpha,ov_alpha+ov_beta), range(2*ov_alpha+ov_beta,total_orbitals)}} //5-7,12-14   
    //                  }};
    const unsigned int ova = static_cast<unsigned int>(ov_alpha);
    const unsigned int ovb = static_cast<unsigned int>(ov_beta);
    TiledIndexSpace MO{MO_IS, {ova,ova,ovb,ovb}};

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
    TiledIndexSpace N = MO("all");

    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};
//    Tensor<T> d_x1{V, O};
//    Tensor<T> d_x2{V, V, O, O};
//    Tensor<T> d_y1{O,V};
//    Tensor<T> d_y2{O,O,V,V};
    Tensor<T> d_f1{N, N};
    Tensor<T> d_v2{N, N, N, N};

//CCSD Variables
    int maxiter    = 50;
    double thresh  = 1.0e-10;
    double zshiftl = 0.0;
    size_t ndiis   = 5;
//EOMCCSD Variables
    int nroots           = 4;
    int maxeomiter       = 50;
//    int eomsolver        = 1; //Indicates which solver to use. (LATER IMPLEMENTATION)
    double eomthresh     = 1.0e-10;
//    double x2guessthresh = 0.6; //Threshold for x2 initial guess (LATER IMPLEMENTATION)
    size_t microeomiter  = 5; //Number of iterations in a microcycle
    
  Tensor<double>::allocate(ec,d_t1,d_t2,d_f1,d_v2);
//  Tensor<double>::allocate(ec,d_t1,d_t2,d_x1,d_x2,d_f1,d_v2);
//  Tensor<double>::allocate(ec,d_t1,d_t2,d_y1,d_y2,d_f1,d_v2);

  Scheduler{ec}
      (d_t1() = 0)
      (d_t2() = 0)
//      (d_x1() = 0)
//      (d_x2() = 0)
//      (d_y1() = 0)
//      (d_y2() = 0)
      (d_f1() = 0)
      (d_v2() = 0)
    .execute();

  //Tensor Map 
  block_for(ec->pg(), d_f1(), [&](IndexVector it) {
    Tensor<T> tensor = d_f1().tensor();
    const TAMM_SIZE size = tensor.block_size(it);
    
    std::vector<T> buf(size);

    auto block_offset = tensor.block_offsets(it);
    auto block_dims = tensor.block_dims(it);

    TAMM_SIZE c=0;
    for (auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for (auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
           j++, c++) {
        buf[c] = F(i, j);
      }
    }
    d_f1.put(it,buf);
  });

  block_for(ec->pg(), d_v2(), [&](IndexVector it) {
      Tensor<T> tensor     = d_v2().tensor();
      const TAMM_SIZE size = tensor.block_size(it);

      std::vector<T> buf(size);

      auto block_dims = tensor.block_dims(it);
      auto block_offset = tensor.block_offsets(it);

      TAMM_SIZE c = 0;
      for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
              j++) {
              for(auto k = block_offset[2]; k < block_offset[2] + block_dims[2];
                  k++) {
                  for(auto l = block_offset[3];
                      l < block_offset[3] + block_dims[3]; l++, c++) {
                      buf[c] = V2(i,j,k,l);
                  }
              }
          }
      }
      d_v2.put(it, buf);
  });

//CCSD Routine:
//!!!(This can be removed and replaced by a call to the seperated CCSD routine)!!!
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  CHECK_NOTHROW(ccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, maxiter, thresh,
                               zshiftl, ndiis, hf_energy, total_orbitals,
                               2 * ov_alpha));

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  std::cout << "\nTime taken for CCSD: " << ccsd_time << " secs\n";

//EOMCCSD Routine:
  cc_t1 = std::chrono::high_resolution_clock::now();

  CHECK_NOTHROW(eomccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2,
//                                  d_x1, d_x2,
                                  nroots, maxeomiter, eomthresh, microeomiter,
                                  total_orbitals, 2 * ov_alpha));

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  std::cout << "\nTime taken for EOMCCSD: " << ccsd_time << " secs\n";

  Tensor<T>::deallocate(d_t1, d_t2, d_f1, d_v2);
//  Tensor<T>::deallocate(d_t1, d_t2, d_x1, d_x2, d_f1, d_v2);
//  Tensor<T>::deallocate(d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);
  MemoryManagerGA::destroy_coll(mgr);
  delete ec;

}
