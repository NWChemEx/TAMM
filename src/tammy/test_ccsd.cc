#include "tensor_sketch.h"

using tammy::IndexSpace;
using tammy::TiledIndexSpace;
using tammy::TiledIndexLabel;
using tammy::Tensor;
using tammy::range;

template<typename T>
void ccsd_e(const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{O, V};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.range_labels<5>("virt");
    std::tie(h3, h4, h5, h6)     = MO.range_labels<4>("occ");

    i1(h6, p5) = f1(h6, p5);
    i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5);
    de() = 0;
    de() += t1(p5, h6) * i1(h6, p5);
    de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2);
}

//template<typename T>
//void driver_e(const TiledIndexSpace& MO) {
//    // Construction of tiled index space MO from skretch
////    IndexSpace MO_IS{range(0, 200),
////                     {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
////    TiledIndexSpace MO{MO_IS, 10};
//
//    const TiledIndexSpace& O = MO("occ");
//    const TiledIndexSpace& V = MO("virt");
//    const TiledIndexSpace& N = MO("all");
//    Tensor<T> de{};
//    Tensor<T> t1{V, O};
//    Tensor<T> t2{V, V, O, O};
//    Tensor<T> f1{N, N};
//    Tensor<T> v2{N, N, N, N};
//    Tensor<T>::allocate(de, t1, t2, f1, v2);
//    ccsd_e(MO, de, t1, t2, f1, v2);
//    Tensor<T>::deallocate(de, t1, t2, f1, v2);
//}

template<typename T>
void ccsd_t1(const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
             const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{O, O};
    Tensor<T> t1_2_2_1{O, V};
    Tensor<T> t1_3_1{V, V};
    Tensor<T> t1_5_1{O, V};
    Tensor<T> t1_6_1{O, O, V, V};

    TiledIndexLabel p2, p3, p4, p5, p6, p7;
    TiledIndexLabel h1, h4, h5, h6, h7, h8;

    std::tie(p2, p3, p4, p5, p6, p7) = MO.range_labels<6>("virt");
    std::tie(h1, h4, h5, h6, h7, h8) = MO.range_labels<6>("occ");

    // Scheduler::allocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
    i0(p2, h1)       = f1(p2, h1);
    t1_2_1(h7, h1)   = f1(h7, h1);
    t1_2_2_1(h7, p3) = f1(h7, p3);
    t1_2_2_1(h7, p3) += -1 * t1(p5, h6) * v2(h6, h7, p3, p5);
    t1_2_1(h7, h1)  += t1(p3, h1) * t1_2_2_1(h7, p3);
    t1_2_1(h7, h1) += -1 * t1(p4, h5) * v2(h5, h7, h1, p4);
    t1_2_1(h7, h1) += -0.5 * t2(p3, p4, h1, h5) * v2(h5, h7, p3, p4);
    i0(p2, h1) += -1 * t1(p2, h7) * t1_2_1(h7, h1);
    t1_3_1(p2, p3) = f1(p2, p3);
    t1_3_1(p2, p3) += -1 * t1(p4, h5) * v2(h5, p2, p3, p4);
    i0(p2, h1) += t1(p3, h1) * t1_3_1(p2, p3);
    i0(p2, h1) += -1 * t1(p3, h4) * v2(h4, p2, h1, p3);
    t1_5_1(h8, p7) = f1(h8, p7);
    t1_5_1(h8, p7) += t1(p5, h6) * v2(h6, h8, p5, p7);
    i0(p2, h1) += t2(p2, p7, h1, h8) * t1_5_1(h8, p7);
    t1_6_1(h4, h5, h1, p3) = v2(h4, h5, h1, p3);
    t1_6_1(h4, h5, h1, p3) += -1 * t1(p6, h1) * v2(h4, h5, p3, p6);
    i0(p2, h1) += -0.5 * t2(p2, p3, h4, h5) * t1_6_1(h4, h5, h1, p3);
    i0(p2, h1) += -0.5 * t2(p3, p4, h1, h5) * v2(h5, p2, p3, p4);
    // Scheduler::deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
}

//template<typename T>
//void driver_t1(TiledIn) {
//    // Construction of tiled index space MO from skretch
////    IndexSpace MO_IS{range(0, 200),
////                     {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
////    TiledIndexSpace MO{MO_IS, 10};
//
//    const TiledIndexSpace& O = MO("occ");
//    const TiledIndexSpace& V = MO("virt");
//    const TiledIndexSpace& N = MO("all");
//
//    Tensor<T> i0{};
//    Tensor<T> t1{V, O};
//    Tensor<T> t2{V, V, O, O};
//    Tensor<T> f1{N, N};
//    Tensor<T> v2{N, N, N, N};
//    ccsd_t1(MO, i0, t1, t2, f1, v2);
//}

template<typename T>
void ccsd_t2(const TiledIndexSpace& MO, Tensor<T>& f1, Tensor<T>& i0,
             Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& v2) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");

    Tensor<T> t2_2_1{O,V,O,O};
    Tensor<T> t2_2_2_1{O,O,O,O} ;
    Tensor<T> t2_2_2_2_1{O,O,O,V};
    Tensor<T> t2_2_4_1{O,V};
    Tensor<T> t2_2_5_1{O,O,O,V};
    Tensor<T> t2_4_1{O,O};
    Tensor<T> t2_4_2_1{O,V};
    Tensor<T> t2_5_1{V,V};
    Tensor<T> t2_6_1{O,O,O,O};
    Tensor<T> t2_6_2_1{O,O,O,V};
    Tensor<T> t2_7_1{O,V,O,V};
    Tensor<T> vt1t1_1{O,V,O,O};

    TiledIndexLabel p1,p2,p3,p4,p5,p6,p7,p8,p9;
    TiledIndexLabel h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11;
    
    std::tie(p1,p2,p3,p4,p5,p6,p7,p8,p9) = MO.range_labels<9>("virt");
    std::tie(h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11) = MO.range_labels<11>("occ");


    // Scheduler::allocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
    //         t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1);

    i0(p3,p4,h1,h2)             =        v2(p3,p4,h1,h2);
    t2_2_1(h10,p3,h1,h2)        =        v2(h10,p3,h1,h2);
    t2_2_2_1(h10,h11,h1,h2)     = -1  *  v2(h10,h11,h1,h2);
    t2_2_2_2_1(h10,h11,h1,p5)   =        v2(h10,h11,h1,p5);
    t2_2_2_2_1(h10,h11,h1,p5)  += -0.5 * t1(p6,h1) * v2(h10,h11,p5,p6);
    t2_2_2_1(h10,h11,h1,h2)    +=        t1(p5,h1) * t2_2_2_2_1(h10,h11,h2,p5);
    t2_2_2_1(h10,h11,h1,h2)    += -0.5 * t2(p7,p8,h1,h2) * v2(h10,h11,p7,p8);
    t2_2_1(h10,p3,h1,h2)       += 0.5  * t1(p3,h11) * t2_2_2_1(h10,h11,h1,h2);
    t2_2_4_1(h10,p5)            =        f1(h10,p5);
    t2_2_4_1(h10,p5)           += -1   * t1(p6,h7) * v2(h7,h10,p5,p6);
    t2_2_1(h10,p3,h1,h2)       += -1   * t2(p3,p5,h1,h2) * t2_2_4_1(h10,p5);
    t2_2_5_1(h7,h10,h1,p9)      =        v2(h7,h10,h1,p9);
    t2_2_5_1(h7,h10,h1,p9)     +=        t1(p5,h1) * v2(h7,h10,p5,p9);
    t2_2_1(h10,p3,h1,h2)       +=        t2(p3,p9,h1,h7) * t2_2_5_1(h7,h10,h2,p9);
    t2(p1,p2,h3,h4)            += 0.5  * t1(p1,h3) * t1(p2,h4);
    t2_2_1(h10,p3,h1,h2)       += 0.5  * t2(p5,p6,h1,h2) * v2(h10,p3,p5,p6);
    t2(p1,p2,h3,h4)            += -0.5 * t1(p1,h3) * t1(p2,h4);
    i0(p3,p4,h1,h2)            += -1   * t1(p3,h10) * t2_2_1(h10,p4,h1,h2);
    i0(p3,p4,h1,h2)            += -1   * t1(p5,h1) * v2(p3,p4,h2,p5);
    t2_4_1(h9,h1)               =        f1(h9,h1);
    t2_4_2_1(h9,p8)             =        f1(h9,p8);
    t2_4_2_1(h9,p8)            +=        t1(p6,h7) * v2(h7,h9,p6,p8);
    t2_4_1(h9,h1)              +=        t1(p8,h1) * t2_4_2_1(h9,p8);
    t2_4_1(h9,h1)              += -1   * t1(p6,h7) * v2(h7,h9,h1,p6);
    t2_4_1(h9,h1)              += -0.5 * t2(p6,p7,h1,h8) * v2(h8,h9,p6,p7);
    i0(p3,p4,h1,h2)            += -1   * t2(p3,p4,h1,h9) * t2_4_1(h9,h2);
    t2_5_1(p3,p5)               =        f1(p3,p5);
    t2_5_1(p3,p5)              += -1   * t1(p6,h7) * v2(h7,p3,p5,p6);
    t2_5_1(p3,p5)              += -0.5 * t2(p3,p6,h7,h8) * v2(h7,h8,p5,p6);
    i0(p3,p4,h1,h2)            += 1    * t2(p3,p5,h1,h2) * t2_5_1(p4,p5);
    t2_6_1(h9,h11,h1,h2)        = -1   * v2(h9,h11,h1,h2);
    t2_6_2_1(h9,h11,h1,p8)      =        v2(h9,h11,h1,p8);
    t2_6_2_1(h9,h11,h1,p8)     += 0.5  * t1(p6,h1) * v2(h9,h11,p6,p8);
    t2_6_1(h9,h11,h1,h2)       +=        t1(p8,h1) * t2_6_2_1(h9,h11,h2,p8);
    t2_6_1(h9,h11,h1,h2)       += -0.5 * t2(p5,p6,h1,h2) * v2(h9,h11,p5,p6);
    i0(p3,p4,h1,h2)            += -0.5 * t2(p3,p4,h9,h11) * t2_6_1(h9,h11,h1,h2);
    t2_7_1(h6,p3,h1,p5)         =        v2(h6,p3,h1,p5);
    t2_7_1(h6,p3,h1,p5)        += -1   * t1(p7,h1) * v2(h6,p3,p5,p7);
    t2_7_1(h6,p3,h1,p5)        += -0.5 * t2(p3,p7,h1,h8) * v2(h6,h8,p5,p7);
    i0(p3,p4,h1,h2)            += -1   * t2(p3,p5,h1,h6) * t2_7_1(h6,p4,h2,p5);
    vt1t1_1(h5,p3,h1,h2)        = 0;
    vt1t1_1(h5,p3,h1,h2)       += -2   * t1(p6,h1) * v2(h5,p3,h2,p6);
    i0(p3,p4,h1,h2)            += -0.5 * t1(p3,h5) * vt1t1_1(h5,p4,h1,h2);
    t2(p1,p2,h3,h4)            += 0.5  * t1(p1,h3) * t1(p2,h4);
    i0(p3,p4,h1,h2)            += 0.5  * t2(p5,p6,h1,h2) * v2(p3,p4,p5,p6);
    t2(p1,p2,h3,h4)            += -0.5 * t1(p1,h3) * t1(p2,h4);

    // Scheduler::deallocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
    //          t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1);

}

int main() {

    // Construction of tiled index space MO from skretch
    IndexSpace MO_IS{range(0, 200),
                    {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
    TiledIndexSpace MO{MO_IS, 10};

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    using T = double;

    Tensor<T> de{};
    Tensor<T> i1{};
    Tensor<T> i2{};

    Tensor<T> t1{V, O};
    Tensor<T> t2{V, V, O, O};
    Tensor<T> f1{N, N};
    Tensor<T> v2{N, N, N, N};
 
    while(true) {
        ccsd_e(MO, de, t1, t2, f1, v2);
        ccsd_t1(MO, i1, t1, t2, f1, v2);
        ccsd_t2(MO, i2, t1, t2, f1, v2);
    }
    return 0;
}
