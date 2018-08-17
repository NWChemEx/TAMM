#define CATCH_CONFIG_RUNNER

#include "toyHF/hartree_fock.hpp"
#include "toyHF/diis.hpp"
#include "toyHF/4index_transform_CD.hpp"
#include "toyHF/NK.hpp"
#include "catch/catch.hpp"
#include "tamm/tamm.hpp"
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

template<typename T>
void print_tensor(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << buf[i] << " ";
        std::cout << std::endl;
    }

}

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

template<typename T>
void ccsd_t1(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2, std::vector<Tensor<T> *> &chol) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{O, O};
    //Tensor<T> t1_2_2_1{O, V};
    //Tensor<T> t1_3_1{V, V};
    //Tensor<T> t1_5_1{O, V};
    Tensor<T> t1_6_1{O, O, O, V};

    Tensor<T> _a28{V, O};
    Tensor<T> _a34{V, O};

    Tensor<T> _a17{V, O};
    Tensor<T> _a22{V, O};

    Tensor<T> _a249{};
    Tensor<T> _a263{O, V};
    Tensor<T> _a287{O, V};
    Tensor<T> _a275{O, O};

    Tensor<T> _a4{V, O};
    Tensor<T> _a9{};
    Tensor<T> _a102{V, O};
    Tensor<T> _a128{V, O};

    Tensor<T> _a302{O, O};
    Tensor<T> _a303{};
    Tensor<T> _a378{O, O};
    Tensor<T> _a477{O, O};

    Tensor<T> _a62{O, O};
    Tensor<T> _a88{O, O};

    Tensor<T> _a209{O, O};
    Tensor<T> _a198{V, O};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8) = MO.labels<8>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8) = MO.labels<8>("occ");

    Scheduler sch{&ec};
    sch.allocate(t1_2_1, t1_6_1, _a28, _a34, _a17, _a102, _a128, _a22,  
                 _a263, _a249, _a275, _a287, _a4, _a9, _a302, _a303, _a378,
                 _a477, _a62, _a88, _a209, _a198);


    sch(i0(p2, h1) = f1(p2, h1));

    //(ccsd_t1_2_2_2_ |= t1_2_2_1(h7, p3) += -1 * t1(p5, h6) * v2(h6, h7, p3,p5)) ccsd_t1.chol -> t1_14
    for(auto x = 0; x < chol.size(); x++) {
        Tensor<T>& cholx = (*(chol.at(x)));
        sch (_a302(h2, h1) = 0)
            (_a303() = 0)
            (_a378(h2, h1) = 0)
            (_a477(h2, h1) = 0);

        sch
          (_a302(h2, h1) += 1.0 * t1(p1, h1) * cholx(h2, p1))
          (_a303() += 1.0 * t1(p3, h3) * cholx(h3, p3))
          (_a378(h2, h1) += 1.0 * _a302(h2, h1) * _a303())
          (i0(p2, h1) += -1.0 * t1(p2, h2) * _a378(h2, h1))
          (_a477(h2, h1) += 1.0 * _a302(h3, h1) * _a302(h2, h3))
          (i0(p2, h1) += 1.0 * t1(p2, h2) * _a477(h2, h1));
    
    //(ccsd_t1_2_3_ |= t1_2_1(h7, h1) += -1 * t1(p4, h5) * v2(h5, h7, h1, p4));
    // ccsd_t1.chol -> t1_9

        sch 
            (_a62(h2, h1) = 0)
            (_a88(h2, h1) = 0);

        sch
          (_a62(h2, h1) += 1.0 * cholx(h3, h1) * _a302(h2, h3))
          (i0(p2, h1) += 1.0 * t1(p2, h2) * _a62(h2, h1))
          (_a88(h2, h1) += 1.0 * cholx(h2, h1) * _a303())
          (i0(p2, h1) += -1.0 * t1(p2, h2) * _a88(h2, h1));
    
    //(ccsd_t1_2_4_ |= t1_2_1(h7, h1) += -0.5 * t2(p3, p4, h1, h5) * v2(h5, h7,
    //p3, p4)) ccsd_t1.chol -> t1_12

        sch(_a198(p1, h1) = 0)
        (_a209(h3, h1) = 0)
        (_a128(p3, h1) = 0)
        (_a275(h3, h1) = 0);

        sch(_a198(p1, h1) += 1.0 * t2(p1, p3, h2, h1) * cholx(h2, p3))
          (_a209(h3, h1) += 1.0 * cholx(h3, p1) * _a198(p1, h1))
          (i0(p2, h1) += 0.5 * t1(p2, h3) * _a209(h3, h1))
          (_a128(p3, h1) += 1.0 * t2(p1, p3, h2, h1) * cholx(h2, p1))
          (_a275(h3, h1) += 1.0 * cholx(h3, p3) * _a128(p3, h1))
          (i0(p2, h1) += -0.5 * t1(p2, h3) * _a275(h3, h1));
    

    //(ccsd_t1_3_2_  |= t1_3_1(p2,p3) += -1 * t1(p4,h5) * v2(h5,p2,p3,p4))
    // ccsd_t1.chol -> t1_10

        sch(_a102(p1, h1) = 0)
           (_a128(p3, h1) = 0);

        sch
          (_a102(p1, h1) += 1.0 * t1(p1, h1) * _a303())
          (i0(p2, h1) += 1.0 * cholx(p2, p1) * _a102(p1, h1))
          (_a128(p3, h1) += 1.0 * t1(p3, h2) * _a302(h2, h1))
          (i0(p2, h1) += -1.0 * cholx(p2, p3) * _a128(p3, h1));
    
    //(ccsd_t1_4_    |= i0(p2,h1) += -1   * t1(p3,h4) * v2(h4,p2,h1,p3))
    // ccsd_t1.chol -> t1_4

        sch(_a4(p2, h2) = 0)
            (_a9() = 0);
            
        sch(_a4(p2, h2) += 1.0 * t1(p1, h2) * cholx(p2, p1))
          (i0(p2, h1) += -1.0 * cholx(h2, h1) * _a4(p2, h2))
          (_a9() += 1.0 * t1(p1, h2) * cholx(h2, p1))
          (i0(p2, h1) += 1.0 * cholx(p2, h1) * _a9());
    
    //(ccsd_t1_5_2_  |= t1_5_1(h8,p7) +=  t1(p5,h6) * v2(h6,h8,p5,p7))
    // ccsd_t1.chol -> t1_13

        sch(_a249() = 0)
        (_a263(h2, p1) = 0)
        //(_a275(h2, h3) = 0)
        (_a287(h2, p1) = 0);

        sch(_a249() += 1.0 * t1(p3, h3) * cholx(h3, p3))
          (_a263(h2, p1) += 1.0 * cholx(h2, p1) * _a249())
          (i0(p2, h1) += 1.0 * t2(p1, p2, h2, h1) * _a263(h2, p1))
          //(_a275(h2, h3) += 1.0 * t1(p3, h3) * cholx(h2, p3))
          (_a287(h2, p1) += 1.0 * cholx(h3, p1) * _a302(h2, h3))
          (i0(p2, h1) += -1.0 * t2(p1, p2, h2, h1) * _a287(h2, p1));
    
    /*(ccsd_t1_6_1_  |= t1_6_1(h4,h5,h1,p3)   =  v2(h4,h5,h1,p3))
    (ccsd_t1_6_2_  |= t1_6_1(h4,h5,h1,p3) += -1 * t1(p6,h1) * v2(h4,h5,p3,p6))
    (ccsd_t1_6_    |= i0(p2,h1) += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))*/
    // ccsd_t1.chol -> t1_6,t1_11

        sch(_a17(p2, h2) = 0)
        (_a22(p2, h3) = 0)
        (_a28(p3, h1) = 0)
        (_a34(p1, h1) = 0);

        sch(_a17(p2, h2) += 1.0 * t2(p1, p2, h2, h3) * cholx(h3, p1))
          (i0(p2, h1) += 0.5 * cholx(h2, h1) * _a17(p2, h2))
          (_a22(p2, h3) += 1.0 * t2(p1, p2, h2, h3) * cholx(h2, p1))
          (i0(p2, h1) += -0.5 * cholx(h3, h1) * _a22(p2, h3))
          (i0(p2, h1) += 0.5 * _a17(p2, h2) * _a302(h2, h1))
          (i0(p2, h1) += -0.5 * _a22(p2, h3) * _a302(h3, h1))
          //(ccsd_t1_7_ |= i0(p2,h1) += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4)) ccsd_t1.chol -> t1_7
          (_a28(p3, h1) += 1.0 * t2(p1, p3, h2, h1) * cholx(h2, p1))
            (i0(p2, h1) += 0.5 * cholx(p2, p3) * _a28(p3, h1))
            (_a34(p1, h1) += 1.0 * t2(p1, p3, h2, h1) * cholx(h2, p3))
            (i0(p2, h1) += -0.5 * cholx(p2, p1) * _a34(p1, h1));
    }

    sch(t1_2_1(h7, h1) = f1(h7, h1))
    (t1_2_1(h7, h1) += t1(p3, h1) * f1(h7, p3))
    (i0(p2, h1) += -1 * t1(p2, h7) * t1_2_1(h7, h1))
    (i0(p2, h1) += t1(p3, h1) * f1(p2, p3));
    sch(i0(p2, h1) += t2(p2, p7, h1, h8) * f1(h8, p7));

    sch.deallocate(t1_2_1, t1_6_1, _a28, _a34, _a17,
                   _a102, _a128, _a22, _a263, _a249, _a275, _a287, 
                   _a4, _a9, _a302, _a303, _a378,
                   _a477, _a62, _a88, _a209, _a198);
    sch.execute();
}

template<typename T>
void ccsd_t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& f1,
             const Tensor<T>& v2, std::vector<Tensor<T> *> &chol) {
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    Tensor<T> t2_2_1{O, V, O, O};
    Tensor<T> t2_2_2_1{O, O, O, O};
    Tensor<T> t2_2_2_2_1{O, O, O, V};
    Tensor<T> t2_2_4_1{O, V};
    Tensor<T> t2_2_5_1{O, O, O, V};
    Tensor<T> t2_4_1{O, O};
    Tensor<T> t2_4_2_1{O, V};
    Tensor<T> t2_5_1{V, V};
    Tensor<T> t2_6_1{O, O, O, O};
    Tensor<T> t2_6_2_1{O, O, O, V};
    Tensor<T> t2_7_1{O, V, O, V};
    Tensor<T> vt1t1_1{O, V, O, O};
    Tensor<T> i0_temp{V, V, O, O};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9) = MO.labels<9>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

    Scheduler sch{&ec};


    Tensor<T> _a3{V, O};
    Tensor<T> _a16{V, O};

    Tensor<T> _a116{V, O};
    Tensor<T> _a119{V, O};
    Tensor<T> _a143{O, O};
    Tensor<T> _a146{V, O};

    Tensor<T> _a67{V, O};
    Tensor<T> _a68{V, O};
    Tensor<T> _a47{V, O};
    Tensor<T> _a42{V, V, O, O}; // VV_NS | OO_NS

    Tensor<T> _a385{V, V, O, O}; // VV_NS | OO_NS
    Tensor<T> _a54{V, V, O, O};  // VV_NS | OO_NS
    Tensor<T> _a30{V, V, O, O};  // VV_NS | OO_NS

    Tensor<T> _a482{};
    Tensor<T> _a496{V, V};
    Tensor<T> _a508{V, O};
    Tensor<T> _a505{V, O, O, O};

    Tensor<T> _a382{O, O};
    Tensor<T> _a405{V, O};
    Tensor<T> _a408{V, O};

    Tensor<T> _a430{V, O, O, O}; // | OO_NS
    Tensor<T> _a443{V, O, O, O}; // | OO_NS
                                 // Tensor<T> _a455 {V,O , O, O}; // | OO_NS
                                 // Tensor<T> _a468 {V,O , O, O}; // | OO_NS

    Tensor<T> _a333{O, O};
    Tensor<T> _a345{O, O};
    Tensor<T> _a357{};
    Tensor<T> _a371{O, O};

    Tensor<T> _a233{O, O};
    Tensor<T> _a238{V, V, O, O}; // VV_NS |  OO_NS
                                 // auto &_a257 {O , O};
                                 // auto &_a260 {V, V , O, O}; //VV_NS |  OO_NS

    Tensor<T> _a281{V, O};
    Tensor<T> _a282{V, O};
    Tensor<T> _a305{V, O, O, O}; // | OO_NS
    Tensor<T> _a318{V, O, O, O}; // | OO_NS

    Tensor<T> _a531{V, O, O, O}; // | OO_NS
    Tensor<T> _a532{V, O};
    Tensor<T> _a583{V, O, O, O}; // | OO_NS
    Tensor<T> _a595{O, O, O, O}; // OO_NS |

    Tensor<T> _a636{V, O};
    Tensor<T> _a650{O, O};

    //_a685,_a686,_a709,_a712
    Tensor<T> _a685{V, O};
    Tensor<T> _a686{V, O};
    Tensor<T> _a709{V, O, O, O}; // | OO_NS
    Tensor<T> _a712{V, O, O, O}; // | OO_NS

    //_a2069,_a2091
    Tensor<T> _a2091{V, O};
    Tensor<T> _a2069{O, O};

    //_a738,_a755,_a862, _a741
    Tensor<T> _a738{V, O};
    Tensor<T> _a755{V, O};
    Tensor<T> _a862{V, O};
    Tensor<T> _a741{O, O};

    //_a1134,_a1160,_a1136,_a1226,_a1256,_a1230
    Tensor<T> _a1134{O, O};
    Tensor<T> _a1160{V, V, O, O};
    Tensor<T> _a1136{O, O};
    Tensor<T> _a1226{O, O};
    Tensor<T> _a1256{V, V, O, O};
    Tensor<T> _a1230{O, O};

    //_a1566,_a1509,_a1511,_a1655,_a1598,_a1602
    Tensor<T> _a1566{O, O};
    Tensor<T> _a1509{O, O};
    Tensor<T> _a1511{O, O};
    Tensor<T> _a1655{O, O};
    Tensor<T> _a1598{O, O};
    Tensor<T> _a1602{};

    //_a1755,_a1724,_a1694,_a1850,_a1819,_a1789
    Tensor<T> _a1755{V, O, O, O};
    Tensor<T> _a1694{V, O, O, O};
    Tensor<T> _a1724{O, O, O, O}; // OO_NS|

    //_a1945,_a1911,_a1882,_a1972,_a2036,_a2003
    Tensor<T> _a1945{V, O, O, O};
    Tensor<T> _a1911{O, V};
    Tensor<T> _a1882{O, O};
    Tensor<T> _a2036{V, O, O, O};
    Tensor<T> _a2003{O, V};
    Tensor<T> _a1972{};

    //_a1324,_a1333,_a1323,_a1488,_a1420,_a1423
    Tensor<T> _a1324{V, O};
    Tensor<T> _a1333{V, O};
    Tensor<T> _a1323{O, O};
    Tensor<T> _a1488{V, O, O, O};
    Tensor<T> _a1420{O, O};
    Tensor<T> _a1423{V, O, O, O};

    //------------------------------CD------------------------------
      sch.allocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
            t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1, _a54, _a3, _a16, _a30,
            _a42, _a47, _a67, _a68, _a482, _a496, _a505, _a508, _a382, _a385,
            _a405, _a408, _a430, _a443, _a333, _a345, _a357, _a371, _a233,
            _a238, _a281, _a282, _a305, _a318, _a119, _a116, _a143, _a146,
            _a531, _a532, _a583, _a595, _a636, _a650, _a685, _a686, _a709,
            _a712, _a738, _a755, _a862, _a741, _a2069, 
            _a2091, _a1134, _a1160,
            _a1136, _a1226, _a1256, _a1230, _a1566, _a1509, _a1511, _a1655,
            _a1598, _a1602, _a1755, _a1724, _a1694, _a1945, _a1911, _a1882,
            _a1972, _a2036, _a2003, _a1324, _a1333, _a1323, _a1488, _a1420,
            _a1423,i0_temp);

    sch(i0(p3, p4, h1, h2) = 0);
        //(t2_4_1(h9, h1) = 0)
        //(t2_5_1(p3, p5) = 0);

    //ccsd_t2_1_: sch(i0(p3, p4, h1, h2) = v2(p3, p4, h1, h2))
    for(auto x = 0; x < chol.size(); x++) {
        Tensor<T>& cholx = (*(chol.at(x)));
        sch(i0(p3, p4, h1, h2) += 1.0 * cholx(p3, h1) * cholx(p4, h2))
           (i0(p3, p4, h1, h2) += -1.0 * cholx(p3, h2) * cholx(p4, h1));
    
    //ccsd_t2_2_1_: sch(t2_2_1(h10, p3, h1, h2) = v2(h10, p3, h1, h2));
   
        sch(_a3(p3, h1) = 0);
        sch(_a3(p3, h1) += 1.0 * t1(p3, h3) * cholx(h3, h1))
            (i0(p3, p4, h1, h2) += -1.0 * cholx(p4, h2) * _a3(p3, h1))
            (i0(p3, p4, h2, h1) += 1.0 * cholx(p4, h2) * _a3(p3, h1)) //4 perms
            (i0(p4, p3, h1, h2) += 1.0 * cholx(p4, h2) * _a3(p3, h1)) //perm symm
            (i0(p4, p3, h2, h1) += -1.0 * cholx(p4, h2) * _a3(p3, h1)); //perm symm

    //ccsd_t2_2_2_1_: sch(t2_2_2_1(h10, h11, h1, h2) = -1 * v2(h10, h11, h1, h2));

        sch(_a67(p4, h2) = 0);
        sch(_a67(p4, h2) += 1.0 * t1(p4, h3) * cholx(h3, h2))
            (i0(p3, p4, h1, h2) += 1 * _a67(p4, h2) * _a67(p3, h1))
            (i0(p3, p4, h2, h1) += -1 * _a67(p4, h2) * _a67(p3, h1)) //4 perms
            ;
    
    //sch(ccsd_t2_2_2_2_1_ |= t2_2_2_2_1(h10, h11, h1, p5) = v2(h10, h11, h1, p5));

        sch
        (_a741(h3, h2) = 0)
        (_a738(p3, h1) = 0)
        (_a755(p4, h2) = 0);

        sch(_a741(h3, h2) += 1.0 * t1(p1, h2) * cholx(h3, p1))
            (_a738(p3, h1) += 1.0 * t1(p3, h4) * cholx(h4, h1))
            (_a755(p4, h2) += 1.0 * t1(p4, h3) * _a741(h3, h2))
            (i0(p3, p4, h1, h2) += 1 * _a738(p3, h1) * _a755(p4, h2))
            (i0(p3, p4, h2, h1) += -1 * _a738(p3, h1) * _a755(p4, h2)) //4 perms
            (i0(p4, p3, h1, h2) += -1 * _a738(p3, h1) * _a755(p4, h2)) //perm symm
            (i0(p4, p3, h2, h1) += 1 * _a738(p3, h1) * _a755(p4, h2)) //perm symm
            ;
   
    //ccsd_t2_2_2_2_2_: sch(t2_2_2_2_1(h10, h11, h1, p5) += -0.5 * t1(p6, h1) * v2(h10, h11, p5, p6))

        sch(_a2069(h3, h1) = 0)
        (_a2091(p4, h1) = 0)
        (i0_temp(p3, p4, h1, h2) = 0);

        sch(_a2069(h3, h1) += 1.0 * t1(p1, h1) * cholx(h3, p1))
            (_a2091(p4, h1) += 1.0 * t1(p4, h3) * _a2069(h3, h1))
            (i0_temp(p3, p4, h1, h2) += _a2091(p4, h1) * _a2091(p3, h2))
            (i0(p3, p4, h1, h2) += -0.5 * i0_temp(p3, p4, h1, h2))
            (i0(p3, p4, h2, h1) += 0.5 * i0_temp(p3, p4, h1, h2)) //4 perms
            (i0(p4, p3, h1, h2) += 0.5 * i0_temp(p3, p4, h1, h2)) //perm symm
            (i0(p4, p3, h2, h1) += -0.5 * i0_temp(p3, p4, h1, h2)); //perm symm

    //ccsd_t2_2_2_3_: sch(t2_2_2_1(h10, h11, h1, h2) += -0.5 * t2(p7, p8, h1, h2) * v2(h10, h11, p7, p8))

        sch(_a1694(p1, h3, h1, h2) = 0)
        (_a1724(h4, h3, h1, h2) = 0)
        (_a1755(p3, h3, h1, h2) = 0);

        sch(_a1694(p1, h3, h1, h2) += 1.0 * t2(p1, p2, h1, h2) * cholx(h3, p2))
            (_a1724(h4, h3, h1, h2) += 1.0 * cholx(h4, p1) * _a1694(p1, h3, h1, h2))
            (_a1755(p3, h3, h1, h2) += 1.0 * t1(p3, h4) * _a1724(h4, h3, h1, h2))
            (i0(p3, p4, h1, h2) += 0.5 * t1(p4, h3) * _a1755(p3, h3, h1, h2))
            (i0(p4, p3, h1, h2) += -0.5 * t1(p4, h3) * _a1755(p3, h3, h1, h2)); //perm symm
    

    /* ------- REMOVE --------
         sch(t2_2_2_1() = 0)
         (t2_2_2_2_1() = 0)
         (t2_2_1(h10, p3, h1, h2) = 0);
     sch(t2_2_2_1(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
     (t2_2_2_1(h10, h11, h2, h1) += -1 * t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5)); //perm symm
     sch(t2_2_1(h10, p3, h1, h2) += 0.5 * t1(p3, h11) * t2_2_2_1(h10, h11, h1, h2)) 
     --------- REMOVE -------- */


    //ccsd_t2_2_4_2_: 
    //sch(t2_2_4_1(h10, p5) += -1 * t1(p6, h7) * v2(h7, h10, p5, p6));

        sch(_a1882(h3, h4) = 0)
        (_a1911(h3, p2) = 0)
        (_a1945(p3, h3, h1, h2) = 0)
        (_a1972() = 0)
        (_a2003(h3, p2) = 0)
        (_a2036(p3, h3, h1, h2) = 0);

        sch(_a1882(h3, h4) += 1.0 * t1(p1, h4) * cholx(h3, p1))
            (_a1911(h3, p2) += 1.0 * cholx(h4, p2) * _a1882(h3, h4))
            (_a1945(p3, h3, h1, h2) += 1.0 * t2(p2, p3, h1, h2) * _a1911(h3, p2))
            (_a1972() += 1.0 * t1(p1, h4) * cholx(h4, p1))
            (_a2003(h3, p2) += 1.0 * cholx(h3, p2) * _a1972())
            (_a2036(p3, h3, h1, h2) += 1.0 * t2(p2, p3, h1, h2) * _a2003(h3, p2))
            (i0(p3, p4, h1, h2) += -1 * t1(p4, h3) * _a1945(p3, h3, h1, h2)) //factor: 0.5->1
            (i0(p4, p3, h1, h2) += 1 * t1(p4, h3) * _a1945(p3, h3, h1, h2)) //perm symm
            (i0(p3, p4, h1, h2) += 1 * t1(p4, h3) * _a2036(p3, h3, h1, h2)) //factor: 0.5->1
            (i0(p4, p3, h1, h2) += -1 * t1(p4, h3) * _a2036(p3, h3, h1, h2)) //perm symm
            ;

   
    //ccsd_t2_2_5_1_: sch(t2_2_5_1(h7, h10, h1, p9) = v2(h7, h10, h1, p9))

        sch(_a281(p3, h2) = 0)
        (_a282(p4, h1) = 0)
        (_a305(p3, h4, h3, h2) = 0)
        (_a318(p3, h4, h1, h2) = 0);

        sch(_a281(p3, h2) += 1.0 * t2(p1, p3, h3, h2) * cholx(h3, p1))
            (_a282(p4, h1) += 1.0 * t1(p4, h4) * cholx(h4, h1))
            (i0(p3, p4, h1, h2) += 1.0 * _a281(p3, h2) * _a282(p4, h1))
            (i0(p3, p4, h2, h1) += -1.0 * _a281(p3, h2) * _a282(p4, h1)) //4 perms
            (i0(p4, p3, h1, h2) += -1.0 * _a281(p3, h2) * _a282(p4, h1)) //perm symm
            (i0(p4, p3, h2, h1) += 1.0 * _a281(p3, h2) * _a282(p4, h1)) //perm symm
            (_a305(p3, h4, h3, h2) += 1.0 * t2(p1, p3, h3, h2) * cholx(h4, p1))
            (_a318(p3, h4, h1, h2) += 1.0 * cholx(h3, h1) * _a305(p3, h4, h3, h2))
            (_a318(p3, h4, h2, h1) += -1.0 * cholx(h3, h1) * _a305(p3, h4, h3, h2)) //perm symm
            (i0(p3, p4, h1, h2) += -1.0 * t1(p4, h4) * _a318(p3, h4, h1, h2))
            (i0(p4, p3, h1, h2) += 1.0 * t1(p4, h4) * _a318(p3, h4, h1, h2)); //perm symm

    //ccsd_t2_2_5_2_: sch(t2_2_5_1(h7, h10, h1, p9) += t1(p5, h1) * v2(h7, h10, p5, p9))

        sch(_a1324(p3, h2) = 0)
        (_a1323(h3, h1) = 0)
        (_a1333(p4, h1) = 0)
        //(_a1420(h4, h1) = 0)
        (_a1423(p3, h3, h4, h2) = 0)
        (_a1488(p3, h3, h1, h2) = 0);

        sch(_a1324(p3, h2) += 1.0 * t2(p2, p3, h4, h2) * cholx(h4, p2))
            (_a1323(h3, h1) += 1.0 * t1(p1, h1) * cholx(h3, p1))
            (_a1333(p4, h1) += 1.0 * t1(p4, h3) * _a1323(h3, h1))
            //(_a1420(h4, h1) += 1.0 * t1(p1, h1) * cholx(h4, p1))
            (_a1423(p3, h3, h4, h2) += 1.0 * t2(p2, p3, h4, h2) * cholx(h3, p2))
            (_a1488(p3, h3, h1, h2) += 1.0 * _a1323(h4, h1) * _a1423(p3, h3, h4, h2))
            (_a1488(p3, h3, h2, h1) += -1.0 * _a1323(h4, h1) * _a1423(p3, h3, h4, h2)) //perm symm
            (i0(p3, p4, h1, h2) += 1.0 * _a1324(p3, h2) * _a1333(p4, h1))
            (i0(p3, p4, h2, h1) += -1.0 * _a1324(p3, h2) * _a1333(p4, h1)) //4 perms
            (i0(p4, p3, h1, h2) += -1.0 * _a1324(p3, h2) * _a1333(p4, h1)) //perm symm
            (i0(p4, p3, h2, h1) += 1.0 * _a1324(p3, h2) * _a1333(p4, h1)) //perm symm

            (i0(p3, p4, h1, h2) += -1.0 * t1(p4, h3) * _a1488(p3, h3, h1, h2))
            (i0(p4, p3, h1, h2) += 1.0 * t1(p4, h3) * _a1488(p3, h3, h1, h2)); //perm symm
    }
    

    sch(t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)); //perm

    //ccsd_t2_2_6_: sch(t2_2_1(h10, p3, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(h10, p3, p5, p6))
    for (auto x = 0; x < chol.size(); x++) {
        Tensor<T> &cholx = (*(chol.at(x)));

        sch(_a430(p2, h3, h1, h2) = 0)
        (_a443(p4, h3, h1, h2) = 0);

        sch(_a430(p2, h3, h1, h2) += 1.0 * t2(p1, p2, h1, h2) * cholx(h3, p1))
            (_a443(p4, h3, h1, h2) += 1.0 * cholx(p4, p2) * _a430(p2, h3, h1, h2))
            (i0(p3, p4, h1, h2) += -1 * t1(p3, h3) * _a443(p4, h3, h1, h2)) //factor -0.5 -> -1
            (i0(p4, p3, h1, h2) += 1 * t1(p3, h3) * _a443(p4, h3, h1, h2)); //perm symm

    }

    sch(t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
    (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
    (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)); //perm

    //@todo --- REMOVE ----
        //sch(t2_2_5_1() = 0);
    //sch(t2_2_4_1(h10, p5) = f1(h10, p5));

    sch(t2_2_1(h10, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * f1(h10, p5))
    //sch(t2_2_1(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
    //(t2_2_1(h10, p3, h2, h1) += -1 * t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9)) //perm symm
    
    (i0(p3, p4, h1, h2) += -1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
    (i0(p4, p3, h1, h2) += 1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2)); //perm sym

  // lccsd_t2_3x_:
  //  sch(i0(p3, p4, h1, h2) += -1 * t1(p5, h1) * v2(p3, p4, h2, p5))
  //  (i0(p3, p4, h2, h1) += 1 * t1(p5, h1) * v2(p3, p4, h2, p5)); //perm sym
  for (auto x = 0; x < chol.size(); x++) {
    Tensor<T> &cholx = (*(chol.at(x)));
    sch(_a16(p4, h2) = 0);
    sch(_a16(p4, h2) += 1.0 * t1(p1, h2) * cholx(p4, p1))
        (i0(p3, p4, h1, h2) += 1 * cholx(p3, h1) * _a16(p4, h2))
        (i0(p3, p4, h2, h1) += -1 * cholx(p3, h1) * _a16(p4, h2)) //4 perms 
        (i0(p4, p3, h1, h2) += -1 * cholx(p3, h1) * _a16(p4, h2)) //perm sym
        (i0(p4, p3, h2, h1) += 1 * cholx(p3, h1) * _a16(p4, h2)); //perm sym 
   
    //ccsd_t2_4_2_2_:
    //sch(t2_4_2_1(h9,p8) + =  t1(p6,h7) * v2(h7,h9,p6,p8));

        sch(_a1509(h3, h1) = 0)
       // (_a1511(h4, h3) = 0)
        (_a1566(h4, h1) = 0)
        (_a1602() = 0)
       // (_a1598(h4, h1) = 0)
        (_a1655(h4, h1) = 0);

        sch(_a1509(h3, h1) += 1.0 * t1(p1, h1) * cholx(h3, p1))
            //(_a1511(h4, h3) += 1.0 * t1(p2, h3) * cholx(h4, p2))
            (_a1566(h4, h1) += 1.0 * _a1509(h3, h1) * _a1509(h4, h3))
            (_a1602() += 1.0 * t1(p2, h3) * cholx(h3, p2))
            //(_a1598(h4, h1) += 1.0 * t1(p1, h1) * cholx(h4, p1))
            (_a1655(h4, h1) += 1.0 * _a1509(h4, h1) * _a1602())
            (_a1655(h4, h1) += -1.0 * _a1566(h4, h1))
            //(i0(p3, p4, h1, h2) += 1.0 * t2(p3, p4, h4, h2) * _a1566(h4, h1))
            //(i0(p3, p4, h2, h1) += -1.0 * t2(p3, p4, h4, h2) * _a1566(h4, h1)) //perm symm
            (i0(p3, p4, h1, h2) += -1.0 * t2(p3, p4, h4, h2) * _a1655(h4, h1))
            (i0(p3, p4, h2, h1) += 1.0 * t2(p3, p4, h4, h2) * _a1655(h4, h1)); //perm symm

    // (ccsd_t2_4_3_:
    // sch(t2_4_1(h9, h1) += -1 * t1(p6, h7) * v2(h7, h9, h1, p6))
        sch(_a333(h3, h4) = 0)
            (_a345(h3, h1) = 0)
            (_a357() = 0)
            (_a371(h3, h1) = 0)
            (_a636(p2, h2) = 0)
            (_a650(h3, h2) = 0);

        sch(_a333(h3, h4) += 1.0 * t1(p1, h4) * cholx(h3, p1))
            (_a345(h3, h1) += 1.0 * cholx(h4, h1) * _a333(h3, h4))
            (_a357() += 1.0 * t1(p1, h4) * cholx(h4, p1))
            (_a371(h3, h1) += 1.0 * cholx(h3, h1) * _a357())

    //ccsd_t2_4_4_:
    //sch(t2_4_1(h9, h1) += -0.5 * t2(p6, p7, h1, h8) * v2(h8, h9, p6, p7)) //move below

            (_a636(p2, h2) += 1.0 * t2(p1, p2, h4, h2) * cholx(h4, p1))
            (_a650(h3, h2) += 1.0 * cholx(h3, p2) * _a636(p2, h2))
            (_a650(h3, h2) += -1 * _a345(h3,h2))
            (_a650(h3, h2) += 1 * _a371(h3,h2))
            (i0(p3, p4, h1, h2) += 1.0 * t2(p3, p4, h3, h1) * _a650(h3, h2))
            (i0(p3, p4, h2, h1) += -1.0 * t2(p3, p4, h3, h1) * _a650(h3, h2)); //perm symm
    
    
    // ccsd_t2_5_2_:
    // sch(t2_5_1(p3, p5) += -1 * t1(p6, h7) * v2(h7, p3, p5, p6))

        sch(_a482() = 0)
            (_a496(p4, p1) = 0)
            (_a505(p3, h3, h1, h2) = 0)
            (_a508(p4, h3) = 0);

        sch(_a482() += 1.0 * t1(p2, h3) * cholx(h3, p2))
            (_a496(p4, p1) += 1.0 * cholx(p4, p1) * _a482())
            (i0(p3, p4, h1, h2) += -1.0 * t2(p1, p3, h1, h2) * _a496(p4, p1))
            (i0(p4, p3, h1, h2) += 1.0 * t2(p1, p3, h1, h2) * _a496(p4, p1)) //perm symm
            (_a505(p3, h3, h1, h2) += 1.0 * t2(p1, p3, h1, h2) * cholx(h3, p1))
            (_a508(p4, h3) += 1.0 * t1(p2, h3) * cholx(p4, p2))
            (i0(p3, p4, h1, h2) += 1.0 * _a505(p3, h3, h1, h2) * _a508(p4, h3))
            (i0(p4, p3, h1, h2) += -1.0 * _a505(p3, h3, h1, h2) * _a508(p4, h3)); //perm symm
    
    //ccsd_t2_5_3_:
    //sch(t2_5_1(p3, p5) += -0.5 * t2(p3, p6, h7, h8) * v2(h7, h8, p5, p6))

        sch(_a531(p4, h4, h1, h2) = 0)
            (_a532(p3, h4) = 0);

        sch(_a531(p4, h4, h1, h2) += 1.0 * t2(p1, p4, h1, h2) * cholx(h4, p1))
            (_a532(p3, h4) += 1.0 * t2(p2, p3, h3, h4) * cholx(h3, p2))
            (i0(p3, p4, h1, h2) += -1 * _a531(p4, h4, h1, h2) * _a532(p3, h4)) //factor -0.5-> -1
            (i0(p4, p3, h1, h2) += 1 * _a531(p4, h4, h1, h2) * _a532(p3, h4)); //perm symm
    
  //ccsd_t2_6_1_:
  //sch(t2_6_1(h9, h11, h1, h2) = -1 * v2(h9, h11, h1, h2));


    sch(_a30(p3, p4, h3, h2) = 0);

    sch(_a30(p3, p4, h3, h2) += 1.0 * t2(p3, p4, h3, h4) * cholx(h4, h2))
        (i0(p3, p4, h1, h2) += 0.5 * cholx(h3, h1) * _a30(p3, p4, h3, h2)) //factor 0.25->0.5 not 1
        (i0(p3, p4, h2, h1) += -0.5 * cholx(h3, h1) * _a30(p3, p4, h3, h2)); //perm symm

    //ccsd_t2_6_2_1_:
    //sch(t2_6_2_1(h9, h11, h1, p8) = v2(h9, h11, h1, p8));
    
        sch(_a233(h4, h2) = 0)
            (_a238(p3, p4, h3, h2) = 0);

        sch(_a233(h4, h2) += 1.0 * t1(p1, h2) * cholx(h4, p1))
            (_a238(p3, p4, h3, h2) += 1.0 * t2(p3, p4, h3, h4) * _a233(h4, h2))
            (i0(p3, p4, h1, h2) += 1 * cholx(h3, h1) * _a238(p3, p4, h3, h2)) //factor 0.5->1 not 2 - why?
            (i0(p3, p4, h2, h1) += -1 * cholx(h3, h1) * _a238(p3, p4, h3, h2)); //perm symm

    //ccsd_t2_6_2_2_:
    //sch(t2_6_2_1(h9, h11, h1, p8) += 0.5 * t1(p6, h1) * v2(h9, h11, p6, p8));

        sch(_a1134(h4, h1) = 0)
        (_a1160(p3, p4, h4, h2) = 0)
        (_a1226(h3, h1) = 0)
        (_a1230(h4, h2) = 0)
        (_a1256(p3, p4, h3, h2) = 0);

        sch(_a1134(h4, h1) += 1.0 * t1(p1, h1) * cholx(h4, p1))
            (_a1160(p3, p4, h4, h2) += 1.0 * t2(p3, p4, h3, h4) * _a1134(h3, h2))
            (_a1256(p3, p4, h3, h2) += 1.0 * t2(p3, p4, h3, h4) * _a1134(h4, h2))
            (i0(p3, p4, h1, h2) += -0.5 * _a1134(h4, h1) * _a1160(p3, p4, h4, h2)) //factor 0.25->0.5
            (i0(p3, p4, h2, h1) += 0.5 * _a1134(h4, h1) * _a1160(p3, p4, h4, h2)) //perm symm
            ;
    
    //(ccsd_t2_6_3_:
    //sch(t2_6_1(h9, h11, h1, h2) += -0.5 * t2(p5, p6, h1, h2) * v2(h9, h11, p5, p6));

        sch(_a583(p1, h4, h1, h2) = 0)
        (_a595(h3, h4, h1, h2) = 0);

        sch(_a583(p1, h4, h1, h2) += 1.0 * t2(p1, p2, h1, h2) * cholx(h4, p2))
            (_a595(h3, h4, h1, h2) += 1.0 * cholx(h3, p1) * _a583(p1, h4, h1, h2))
            (i0(p3, p4, h1, h2) += 0.5 * t2(p3, p4, h3, h4) * _a595(h3, h4, h1, h2)); //factor: 0.25->0.5 why?
    

    //ccsd_t2_7_1_:
    //sch(t2_7_1(h6, p3, h1, p5) = v2(h6, p3, h1, p5));

        sch(_a42(p3, p4, h3, h2) = 0)
        (_a47(p3, h2) = 0);

        sch
            (_a42(p3, p4, h3, h2) += 1.0 * t2(p1, p3, h3, h2) * cholx(p4, p1))
            (_a42(p4, p3, h3, h2) += -1.0 * t2(p1, p3, h3, h2) * cholx(p4, p1)) //perm - why ?
            (i0(p3, p4, h1, h2) += 1.0 * cholx(h3, h1) * _a42(p3, p4, h3, h2)) //factor not changed--why ?
            (i0(p3, p4, h2, h1) += -1.0 * cholx(h3, h1) * _a42(p3, p4, h3, h2)) //perm symm
            (_a47(p3, h2) += 1.0 * t2(p1, p3, h3, h2) * cholx(h3, p1))
            (i0(p3, p4, h1, h2) += -1.0 * cholx(p4, h1) * _a47(p3, h2))
            (i0(p3, p4, h2, h1) += 1.0 * cholx(p4, h1) * _a47(p3, h2)) //4 perms
            (i0(p4, p3, h1, h2) += 1.0 * cholx(p4, h1) * _a47(p3, h2)) //perm
            (i0(p4, p3, h2, h1) += -1.0 * cholx(p4, h1) * _a47(p3, h2)); //perm 
    

    // ccsd_t2_7_2_:
    //sch(t2_7_1(h6, p3, h1, p5) += -1 * t1(p7, h1) * v2(h6, p3, p5, p7));

        sch(_a382(h3, h1) = 0)
            (_a385(p1, p3, h2, h1) = 0)
            (_a405(p3, h2) = 0)
            (_a408(p4, h1) = 0);

        sch(_a382(h3, h1) += 1.0 * t1(p2, h1) * cholx(h3, p2))
            (_a385(p1, p3, h2, h1) += 1.0 * t2(p1, p3, h3, h2) * _a382(h3, h1))
            (_a385(p1, p3, h1, h2) += -1.0 * t2(p1, p3, h3, h2) * _a382(h3, h1)) //perm symm
            (i0(p3, p4, h1, h2) += 1.0 * cholx(p4, p1) * _a385(p1, p3, h2, h1))
            (i0(p4, p3, h1, h2) += -1.0 * cholx(p4, p1) * _a385(p1, p3, h2, h1)) //perm symm
            (_a405(p3, h2) += 1.0 * t2(p1, p3, h3, h2) * cholx(h3, p1))
            (_a408(p4, h1) += 1.0 * t1(p2, h1) * cholx(p4, p2))
            (i0(p3, p4, h1, h2) += -1.0 * _a405(p3, h2) * _a408(p4, h1))
            (i0(p3, p4, h2, h1) += 1.0 * _a405(p3, h2) * _a408(p4, h1)) //4 perms
            (i0(p4, p3, h1, h2) += 1.0 * _a405(p3, h2) * _a408(p4, h1)) //perm symm
            (i0(p4, p3, h2, h1) += -1.0 * _a405(p3, h2) * _a408(p4, h1)); //perm symm
    

    //ccsd_t2_7_3_:
    //sch(t2_7_1(h6, p3, h1, p5) += -0.5 * t2(p3, p7, h1, h8) * v2(h6, h8, p5, p7))

        sch (_a685(p4, h1) = 0)
            (_a686(p3, h2) = 0)
            (_a709(p4, h4, h3, h1) = 0)
            (_a712(p3, h3, h4, h2) = 0);

        sch(_a685(p4, h1) += 1.0 * t2(p1, p4, h3, h1) * cholx(h3, p1))
            //(_a686(p3, h2) += 1.0 * t2(p2, p3, h4, h2) * cholx(h4, p2))
            (_a709(p4, h4, h3, h1) += 1.0 * t2(p1, p4, h3, h1) * cholx(h4, p1))
            //(_a712(p3, h3, h4, h2) += 1.0 * t2(p2, p3, h4, h2) * cholx(h3, p2))

            (i0(p3, p4, h1, h2) += -1 * _a685(p4, h1) * _a685(p3, h2))
            (i0(p3, p4, h2, h1) += 1 * _a685(p4, h1) * _a685(p3, h2)) //4 perms
            // (i0(p4, p3, h1, h2) += 0.5 * _a685(p4, h1) * _a685(p3, h2)) //perm symm
            // (i0(p4, p3, h2, h1) += -0.5 * _a685(p4, h1) * _a685(p3, h2)) //perm symm

            (i0(p3, p4, h1, h2) += 1 * _a709(p4, h4, h3, h1) * _a709(p3, h3, h4, h2))
            (i0(p3, p4, h2, h1) += -1 * _a709(p4, h4, h3, h1) * _a709(p3, h3, h4, h2)) //4 perms
            // (i0(p4, p3, h1, h2) += -0.5 * _a709(p4, h4, h3, h1) * _a709(p3, h3, h4, h2)) //perm symm
            // (i0(p4, p3, h2, h1) += 0.5 * _a709(p4, h4, h3, h1) * _a709(p3, h3, h4, h2)) //perm symm
            ;

    // vt1t1_1_2_:
    // sch(vt1t1_1(h5, p3, h1, h2) += -2 * t1(p6, h1) * v2(h5, p3, h2, p6))
    // (vt1t1_1(h5, p3, h2, h1) += 2 * t1(p6, h1) * v2(h5, p3, h2, p6)); //perm symm

        sch (_a119(p4, h2) = 0)
            (_a116(p3, h1) = 0)
            (_a143(h3, h2) = 0)
            (_a146(p3, h2) = 0);

        sch(_a119(p4, h2) += 1.0 * t1(p1, h2) * cholx(p4, p1))
            (_a116(p3, h1) += 1.0 * t1(p3, h3) * cholx(h3, h1))
            (_a143(h3, h2) += 1.0 * t1(p1, h2) * cholx(h3, p1))
            (_a146(p3, h2) += 1.0 * t1(p3, h3) * _a143(h3, h2))
            (i0(p3, p4, h1, h2) += -1.0 * _a116(p3, h1) * _a119(p4, h2))
            (i0(p3, p4, h2, h1) += 1.0 * _a116(p3, h1) * _a119(p4, h2)) //4 perms
            (i0(p4, p3, h1, h2) += 1.0 * _a116(p3, h1) * _a119(p4, h2)) //perm
            (i0(p4, p3, h2, h1) += -1.0 * _a116(p3, h1) * _a119(p4, h2)) //perm

            (i0(p3, p4, h1, h2) += 1.0 * cholx(p4, h1) * _a146(p3, h2))
            (i0(p3, p4, h2, h1) += -1.0 * cholx(p4, h1) * _a146(p3, h2)) //4 perms
            (i0(p4, p3, h1, h2) += -1.0 * cholx(p4, h1) * _a146(p3, h2)) //perm
            (i0(p4, p3, h2, h1) += 1.0 * cholx(p4, h1) * _a146(p3, h2)) //perm
            ;
    }

    //(t2_6_2_1() = 0)
      //sch(t2_6_1() = 0);
    sch(t2_4_1(h9, h1) = f1(h9, h1))
    //(t2_4_2_1(h9, p8) = f1(h9, p8))
    (t2_4_1(h9, h1) += t1(p8, h1) * f1(h9, p8));

    sch(i0(p3, p4, h1, h2) += -1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
    (i0(p3, p4, h2, h1) += 1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2)) //perm sym
    //(t2_5_1(p3, p5) = f1(p3, p5));

    (i0(p3, p4, h1, h2) += 1 * t2(p3, p5, h1, h2) * f1(p4, p5))
    (i0(p4, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * f1(p4, p5)); //perm sym

    // REMOVE
    //sch(t2_6_1(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
   // (t2_6_1(h9, h11, h2, h1) += -1 * t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8)); //perm symm
    //sch(i0(p3, p4, h1, h2) += -0.5 * t2(p3, p4, h9, h11) * t2_6_1(h9, h11, h1, h2));

    // sch(t2_7_1() = 0);
    // sch(i0(p3, p4, h1, h2) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
    // (i0(p3, p4, h2, h1) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //4 perms
    // (i0(p4, p3, h1, h2) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm
    // (i0(p4, p3, h2, h1) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm

    // (vt1t1_1(h5, p3, h1, h2) = 0);

    // sch(i0(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
    // (i0(p4, p3, h1, h2) += 0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2)) //perm symm

    sch(t2(p1, p2, h3, h4) += 1 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h4, h3) += -1 * t1(p1, h3) * t1(p2, h4)) //4 perms
    ;

    //ccsd_t2_8_:
    //sch(i0(p3, p4, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(p3, p4, p5, p6));
    for (auto x = 0; x < chol.size(); x++) {
        Tensor<T> &cholx = (*(chol.at(x)));
        sch(_a54(p1, p4, h1, h2) = 0);
        sch(_a54(p1, p4, h1, h2) += 1.0 * t2(p1, p2, h1, h2) * cholx(p4, p2))
            (i0(p3, p4, h1, h2) += 0.5 * cholx(p3, p1) * _a54(p1, p4, h1, h2)) //factor 0.25->0.5
            (i0(p4, p3, h1, h2) += -0.5 * cholx(p3, p1) * _a54(p1, p4, h1, h2)); //perm symm
    }

    sch(t2(p1, p2, h3, h4) += -1 * t1(p1, h3) * t1(p2, h4))
    (t2(p1, p2, h4, h3) += 1 * t1(p1, h3) * t1(p2, h4)) //4 perms
    ;

  sch.deallocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1,
              t2_4_2_1, t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1, _a54, _a3,
              _a16, _a30, _a42, _a47, _a67, _a68, _a482, _a496, _a505, _a1226,
              _a508, _a116, _a119, _a143, _a146, _a382, _a385, _a405, _a408,
              _a430, _a443, _a333, _a345, _a357, _a371, _a233, _a238, _a281,
              _a282, _a305, _a318, _a531, _a532, _a583, _a595, _a636, _a650,
              _a685, _a686, _a709, _a712, _a2069, _a2091, _a738, _a755, _a862, _a741,
              _a1134, _a1160, _a1136, _a1256, _a1230, _a1566, _a1509,
              _a1511, _a1655, _a1598, _a1602, _a1755, _a1724, _a1694, _a1945,
              _a1911, _a1882, _a1972, _a2036, _a2003, _a1324, _a1333, _a1323,
              _a1488, _a1420, _a1423,i0_temp);
    //-----------------------------CD----------------------------------
    
    sch.execute();

}


/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double,double> rest(ExecutionContext& ec,
                              const TiledIndexSpace& MO,
                               Tensor<T>& d_r1,
                               Tensor<T>& d_r2,
                               Tensor<T>& d_t1,
                               Tensor<T>& d_t2,
                              const Tensor<T>& de,
                              std::vector<T>& p_evl_sorted, T zshiftl, 
                              const TAMM_SIZE& noab) {

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
        jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, false, p_evl_sorted,noab);
      };
      auto l2 = [&]() {
        jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, false, p_evl_sorted,noab);
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

template<typename T>
void ccsd_driver(ExecutionContext* ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                    std::vector<Tensor<T> *> &chol,
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
  std::vector<double> p_evl_sorted(total_orbitals);

    // Tensor<T> d_evl{N};
    // Tensor<T>::allocate(ec, d_evl);
    // TiledIndexLabel n1;
    // std::tie(n1) = MO.labels<1>("all");

    // sch(d_evl(n1) = 0.0)
    // .execute();

  {
      auto lambda = [&](const IndexVector& blockid) {
          if(blockid[0] == blockid[1]) {
              Tensor<T> tensor     = d_f1;
              const TAMM_SIZE size = tensor.block_size(blockid);

              std::vector<T> buf(size);
              tensor.get(blockid, buf);

            auto block_dims = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);


              auto dim    = block_dims[0];
              auto offset = block_offset[0];
              TAMM_SIZE i = 0;
              for(auto p = offset; p < offset + dim; p++, i++) {
                  p_evl_sorted[p] = buf[i * dim + i];
              }
          }
      };
      block_for(ec->pg(), d_f1(), lambda);
  }
  ec->pg().barrier();

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
          ccsd_t1(*ec, MO, d_r1, d_t1, d_t2, d_f1, d_v2, chol);
          ccsd_t2(*ec, MO, d_r2, d_t1, d_t2, d_f1, d_v2, chol);

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
    Tensor3D CholVpr;
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
                                        freeze_virtual, C, F, shells, CholVpr);
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
    Tensor<T> d_f1{N, N};
    Tensor<T> d_v2{N, N, N, N};
    int maxiter    = 50;
    double thresh  = 1.0e-10;
    double zshiftl = 0.0;
    size_t ndiis      = 5;

  Tensor<double>::allocate(ec,d_t1,d_t2,d_f1,d_v2);

  Scheduler{ec}
      (d_t1() = 0)
      (d_t2() = 0)
      (d_f1() = 0)
      (d_v2() = 0)
    .execute();

  // CD
  auto chol_dims = CholVpr.dimensions();
  auto chol_count = chol_dims[2];
  cout << "Number of cholesky vectors:" << chol_count << endl;
  std::vector<Tensor<T> *> chol_vecs(chol_count);

  for(auto x = 0; x < chol_count; x++) {
      Tensor<T>* cholvec = new Tensor<T>{N, N};
      Tensor<T>::allocate(ec, *cholvec);
      Scheduler{ec}((*cholvec)() = 0).execute();
      chol_vecs[x] = cholvec;
  }

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

  for(auto x = 0; x < chol_count; x++) {
      Tensor<T>* cholvec = chol_vecs.at(x);

      block_for(ec->pg(), (*cholvec)(), [&](IndexVector it) {
          Tensor<T> tensor     = (*cholvec)().tensor();
          const TAMM_SIZE size = tensor.block_size(it);

          std::vector<T> buf(size);

          auto block_offset = tensor.block_offsets(it);
          auto block_dims   = tensor.block_dims(it);

          TAMM_SIZE c = 0;
          for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
              i++) {
              for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
                  j++, c++) {
                  buf[c] = CholVpr(i,j, x);;
              }
          }
          (*cholvec).put(it, buf);
      });
  }

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  CHECK_NOTHROW(ccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, chol_vecs,
                               maxiter, thresh, zshiftl, ndiis, hf_energy,
                               total_orbitals, 2 * ov_alpha));

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  std::cout << "\nTime taken for Cholesky CCSD: " << ccsd_time << " secs\n";

  Tensor<T>::deallocate(d_t1, d_t2, d_f1, d_v2);
  for (auto x = 0; x < chol_count; x++) Tensor<T>::deallocate(*chol_vecs[x]);
  MemoryManagerGA::destroy_coll(mgr);

}
