#include "diis.hpp"
#include "ccsd_common.hpp"


using namespace tamm;

template<typename T>
void lambda_ccsd_y1(ExecutionContext& ec, const TiledIndexSpace& MO,
                    Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2,
                    const Tensor<T>& y1, const Tensor<T>& y2,
                    const Tensor<T>& f1, const Tensor<T>& v2) {

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) = MO.labels<11>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

    Tensor<T> i_1     {{O , O},{1,1}};
    Tensor<T> i_1_1   {{O , V},{1,1}};
    Tensor<T> i_2     {{V , V},{1,1}};
    Tensor<T> i_2_1   {{O , V},{1,1}};
    Tensor<T> i_3     {{V , O},{1,1}};
    Tensor<T> i_3_1   {{O , O},{1,1}};
    Tensor<T> i_3_1_1 {{O , V},{1,1}};
    Tensor<T> i_3_2   {{V , V},{1,1}};
    Tensor<T> i_3_3   {{O , V},{1,1}};
    Tensor<T> i_3_4   {{O,O , O,V},{2,2}};
    Tensor<T> i_4     {{O,V , O,O},{2,2}};
    Tensor<T> i_4_1   {{O,O , O,O},{2,2}};
    Tensor<T> i_4_1_1 {{O,O , O,V},{2,2}};
    Tensor<T> i_4_2   {{O,V , O,V},{2,2}};
    Tensor<T> i_4_3   {{O , V},{1,1}};
    Tensor<T> i_4_4   {{O,O , O,V},{2,2}};
    Tensor<T> i_5     {{V,V , O,V},{2,2}};
    Tensor<T> i_6     {{V , O},{1,1}};
    Tensor<T> i_6_1   {{O , O},{1,1}};
    Tensor<T> i_6_2   {{O,O , O,V},{2,2}};
    Tensor<T> i_7     {{O , O},{1,1}};
    Tensor<T> i_8     {{O , O},{1,1}};
    Tensor<T> i_9     {{V , V},{1,1}};
    Tensor<T> i_10    {{O,O , O,V},{2,2}};
    Tensor<T> i_11    {{O,V , O,O},{2,2}};
    Tensor<T> i_11_1  {{O,O , O,O},{2,2}};
    Tensor<T> i_11_1_1{{O,O , O,V},{2,2}};
    Tensor<T> i_11_2  {{O,O , O,V},{2,2}};
    Tensor<T> i_11_3  {{O , O},{1,1}};
    Tensor<T> i_12    {{O,O , O,O},{2,2}};
    Tensor<T> i_12_1  {{O,O , O,V},{2,2}};
    Tensor<T> i_13    {{O,V , O,V},{2,2}};
    Tensor<T> i_13_1  {{O,O , O,V},{2,2}};

    Scheduler sch{ec};

    sch
      .allocate(i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3, i_3_4,
             i_4, i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6, i_6_1, i_6_2,
             i_7, i_8, i_9, i_10, i_11, i_11_1, i_11_1_1, i_11_2, i_11_3, i_12,
             i_12_1, i_13, i_13_1)
      ( i0(h2,p1)                     =         f1(h2,p1)                                   )
      (   i_1(h2,h7)                  =         f1(h2,h7)                                   )
      (     i_1_1(h2,p3)              =         f1(h2,p3)                                   )
      (     i_1_1(h2,p3)             +=         t1(p5,h6)          * v2(h2,h6,p3,p5)        )
      (   i_1(h2,h7)                 +=         t1(p3,h7)          * i_1_1(h2,p3)           )
      (   i_1(h2,h7)                 +=         t1(p3,h4)          * v2(h2,h4,h7,p3)        )
      (   i_1(h2,h7)                 += -0.5  * t2(p3,p4,h6,h7)    * v2(h2,h6,p3,p4)        )
      ( i0(h2,p1)                    += -1    * y1(h7,p1)          * i_1(h2,h7)             )
      (   i_2(p7,p1)                  =         f1(p7,p1)                                   )
      (   i_2(p7,p1)                 += -1    * t1(p3,h4)          * v2(h4,p7,p1,p3)        )
      (     i_2_1(h4,p1)              =  0                                                  )
      (     i_2_1(h4,p1)             +=         t1(p5,h6)          * v2(h4,h6,p1,p5)        )
      (   i_2(p7,p1)                 += -1    * t1(p7,h4)          * i_2_1(h4,p1)           )
      ( i0(h2,p1)                    +=         y1(h2,p7)          * i_2(p7,p1)             )
      ( i0(h2,p1)                    += -1    * y1(h4,p3)          * v2(h2,p3,h4,p1)        )
      (   i_3(p9,h11)                 =         f1(p9,h11)                                  )
      (     i_3_1(h10,h11)            =         f1(h10,h11)                                 )
      (       i_3_1_1(h10,p3)         =         f1(h10,p3)                                  )
      (       i_3_1_1(h10,p3)        += -1    * t1(p7,h8)          * v2(h8,h10,p3,p7)       ) 
      (     i_3_1(h10,h11)           +=         t1(p3,h11)         * i_3_1_1(h10,p3)        )
      (     i_3_1(h10,h11)           += -1    * t1(p5,h6)          * v2(h6,h10,h11,p5)      )
      (     i_3_1(h10,h11)           +=  0.5  * t2(p3,p4,h6,h11)   * v2(h6,h10,p3,p4)       )
      (   i_3(p9,h11)                += -1    * t1(p9,h10)         * i_3_1(h10,h11)         )
      (     i_3_2(p9,p7)              =         f1(p9,p7)                                   )
      (     i_3_2(p9,p7)             +=         t1(p5,h6)          * v2(h6,p9,p5,p7)        )
      (   i_3(p9,h11)                +=         t1(p7,h11)         * i_3_2(p9,p7)           )
      (   i_3(p9,h11)                += -1    * t1(p3,h4)          * v2(h4,p9,h11,p3)       )
      (     i_3_3(h5,p4)              =         f1(h5,p4)                                   )
      (     i_3_3(h5,p4)             +=         t1(p7,h8)          * v2(h5,h8,p4,p7)        )
      (   i_3(p9,h11)                +=         t2(p4,p9,h5,h11)   * i_3_3(h5,p4)           )
      (     i_3_4(h5,h6,h11,p4)       =         v2(h5,h6,h11,p4)                            )
      (     i_3_4(h5,h6,h11,p4)      += -1    * t1(p7,h11)         * v2(h5,h6,p4,p7)        )
      (   i_3(p9,h11)                +=  0.5  * t2(p4,p9,h5,h6)    * i_3_4(h5,h6,h11,p4)    )
      (   i_3(p9,h11)                +=  0.5  * t2(p3,p4,h6,h11)   * v2(h6,p9,p3,p4)        )
      ( i0(h2,p1)                    +=         y2(h2,h11,p1,p9)   * i_3(p9,h11)            )      
      (   i_4(h2,p9,h11,h12)          =         v2(h2,p9,h11,h12)                           )
      (     i_4_1(h2,h7,h11,h12)      =         v2(h2,h7,h11,h12)                           )
      (       i_4_1_1(h2,h7,h12,p3)   =         v2(h2,h7,h12,p3)                            )
      (       i_4_1_1(h2,h7,h12,p3)  += -0.5  * t1(p5,h12)         * v2(h2,h7,p3,p5)        )
      (     i_4_1(h2,h7,h11,h12)     += -2    * t1(p3,h11)         * i_4_1_1(h2,h7,h12,p3)  )
      (     i_4_1(h2,h7,h11,h12)     +=  0.5  * t2(p3,p4,h11,h12)  * v2(h2,h7,p3,p4)        )
      (   i_4(h2,p9,h11,h12)         += -1    * t1(p9,h7)          * i_4_1(h2,h7,h11,h12)   )
      (     i_4_2(h2,p9,h12,p3)       =         v2(h2,p9,h12,p3)                            )
      (     i_4_2(h2,p9,h12,p3)      += -0.5  * t1(p5,h12)         * v2(h2,p9,p3,p5)        )
      (   i_4(h2,p9,h11,h12)         += -2    * t1(p3,h11)         * i_4_2(h2,p9,h12,p3)    )
      (     i_4_3(h2,p5)              =         f1(h2,p5)                                   )
      (     i_4_3(h2,p5)             +=         t1(p7,h8)          * v2(h2,h8,p5,p7)        )
      (   i_4(h2,p9,h11,h12)         +=         t2(p5,p9,h11,h12)  * i_4_3(h2,p5)           )
      (     i_4_4(h2,h6,h12,p4)       =         v2(h2,h6,h12,p4)                            )
      (     i_4_4(h2,h6,h12,p4)      += -1    * t1(p7,h12)         * v2(h2,h6,p4,p7)        )
      (   i_4(h2,p9,h11,h12)         += -2    * t2(p4,p9,h6,h11)   * i_4_4(h2,h6,h12,p4)    )
      (   i_4(h2,p9,h11,h12)         +=  0.5  * t2(p3,p4,h11,h12)  * v2(h2,p9,p3,p4)        )
      ( i0(h2,p1)                    += -0.5  * y2(h11,h12,p1,p9)  * i_4(h2,p9,h11,h12)     )
      (   i_5(p5,p8,h7,p1)            = -1    * v2(p5,p8,h7,p1)                             )
      (   i_5(p5,p8,h7,p1)           +=         t1(p3,h7)          * v2(p5,p8,p1,p3)        )
      ( i0(h2,p1)                    +=  0.5  * y2(h2,h7,p5,p8)    * i_5(p5,p8,h7,p1)       )
      (   i_6(p9,h10)                 =         t1(p9,h10)                                  )
      (   i_6(p9,h10)                +=         t2(p3,p9,h5,h10)   * y1(h5,p3)              )
      (     i_6_1(h6,h10)             =  0                                                  )
      (     i_6_1(h6,h10)            +=         t1(p5,h10)         * y1(h6,p5)              )
      (     i_6_1(h6,h10)            +=  0.5  * t2(p3,p4,h5,h10)   * y2(h5,h6,p3,p4)        )
      (   i_6(p9,h10)                += -1    * t1(p9,h6)          * i_6_1(h6,h10)          )
      (     i_6_2(h5,h6,h10,p3)       =  0                                                  )
      (     i_6_2(h5,h6,h10,p3)      +=         t1(p7,h10)         * y2(h5,h6,p3,p7)        )
      (   i_6(p9,h10)                += -0.5  * t2(p3,p9,h5,h6)    * i_6_2(h5,h6,h10,p3)    )
      ( i0(h2,p1)                    +=         i_6(p9,h10)        * v2(h2,h10,p1,p9)       )
      (   i_7(h2,h3)                  =  0                                                  )
      (   i_7(h2,h3)                 +=         t1(p4,h3)          * y1(h2,p4)              )
      (   i_7(h2,h3)                 +=  0.5  * t2(p4,p5,h3,h6)    * y2(h2,h6,p4,p5)        )
      ( i0(h2,p1)                    += -1    * i_7(h2,h3)         * f1(h3,p1)              )
      (   i_8(h6,h8)                  =  0                                                  )
      (   i_8(h6,h8)                 +=         t1(p3,h8)          * y1(h6,p3)              )
      (   i_8(h6,h8)                 +=  0.5  * t2(p3,p4,h5,h8)    * y2(h5,h6,p3,p4)        )
      ( i0(h2,p1)                    +=         i_8(h6,h8)         * v2(h2,h8,h6,p1)        )
      (   i_9(p7,p8)                  =  0                                                  )
      (   i_9(p7,p8)                 +=         t1(p7,h4)          * y1(h4,p8)              )
      (   i_9(p7,p8)                 +=  0.5  * t2(p3,p7,h5,h6)    * y2(h5,h6,p3,p8)        )
      ( i0(h2,p1)                    +=         i_9(p7,p8)         * v2(h2,p8,p1,p7)        )
      (   i_10(h2,h6,h4,p5)           =  0                                                  )
      (   i_10(h2,h6,h4,p5)          +=         t1(p3,h4)          * y2(h2,h6,p3,p5)        )
      ( i0(h2,p1)                    +=         i_10(h2,h6,h4,p5)  * v2(h4,p5,h6,p1)        )
      (   i_11(h2,p9,h6,h12)          =  0                                                  )
      (   i_11(h2,p9,h6,h12)         += -1    * t2(p3,p9,h6,h12)   * y1(h2,p3)              )
      (     i_11_1(h2,h10,h6,h12)     =  0                                                  )
      (     i_11_1(h2,h10,h6,h12)    += -1    * t2(p3,p4,h6,h12)   * y2(h2,h10,p3,p4)       )
      (       i_11_1_1(h2,h10,h6,p5)  =  0                                                  )
      (       i_11_1_1(h2,h10,h6,p5) +=         t1(p7,h6)          * y2(h2,h10,p5,p7)       )
      (     i_11_1(h2,h10,h6,h12)    +=  2    * t1(p5,h12)         * i_11_1_1(h2,h10,h6,p5) )
      (   i_11(h2,p9,h6,h12)         += -0.5  * t1(p9,h10)         * i_11_1(h2,h10,h6,h12)  )
      (     i_11_2(h2,h5,h6,p3)       =  0                                                  )
      (     i_11_2(h2,h5,h6,p3)      +=         t1(p7,h6)          * y2(h2,h5,p3,p7)        )
      (   i_11(h2,p9,h6,h12)         +=  2    * t2(p3,p9,h5,h12)   * i_11_2(h2,h5,h6,p3)    )
      (     i_11_3(h2,h12)            =  0                                                  )
      (     i_11_3(h2,h12)           +=         t2(p3,p4,h5,h12)   * y2(h2,h5,p3,p4)        )
      (   i_11(h2,p9,h6,h12)         += -1    * t1(p9,h6)          * i_11_3(h2,h12)         )
      ( i0(h2,p1)                    +=  0.5  * i_11(h2,p9,h6,h12) * v2(h6,h12,p1,p9)       )
      (   i_12(h2,h7,h6,h8)           =  0                                                  )
      (   i_12(h2,h7,h6,h8)          += -1    * t2(p3,p4,h6,h8)    * y2(h2,h7,p3,p4)        )
      (     i_12_1(h2,h7,h6,p3)       =  0                                                  )
      (     i_12_1(h2,h7,h6,p3)      +=         t1(p5,h6)          * y2(h2,h7,p3,p5)        )
      (   i_12(h2,h7,h6,h8)          +=  2    * t1(p3,h8)          * i_12_1(h2,h7,h6,p3)    )
      ( i0(h2,p1)                    +=  0.25 * i_12(h2,h7,h6,h8)  * v2(h6,h8,h7,p1)        )
      (   i_13(h2,p8,h6,p7)           =  0                                                  )
      (   i_13(h2,p8,h6,p7)          +=         t2(p3,p8,h5,h6)    * y2(h2,h5,p3,p7)        )
      (     i_13_1(h2,h4,h6,p7)       =  0                                                  )
      (     i_13_1(h2,h4,h6,p7)      +=         t1(p5,h6)          * y2(h2,h4,p5,p7)        )
      (   i_13(h2,p8,h6,p7)          += -1    * t1(p8,h4)          * i_13_1(h2,h4,h6,p7)    )
      ( i0(h2,p1)                    +=         i_13(h2,p8,h6,p7)  * v2(h6,p7,p1,p8)        )      
   .deallocate(i_1, i_1_1, i_2, i_2_1, i_3, i_3_1, i_3_1_1, i_3_2, i_3_3,
            i_3_4, i_4, i_4_1, i_4_1_1, i_4_2, i_4_3, i_4_4, i_5, i_6,
            i_6_1, i_6_2, i_7, i_8, i_9, i_10, i_11, i_11_1, i_11_1_1,
            i_11_2, i_11_3, i_12, i_12_1, i_13, i_13_1).execute();
}

template<typename T>
void lambda_ccsd_y2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& y1, Tensor<T>& y2,
             const Tensor<T>& f1,  const Tensor<T>& v2) {

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11) = MO.labels<11>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

  Tensor<T> i_1    {{O,V},{1,1}};
  Tensor<T> i_2    {{O,O,O,V},{2,2}};
  Tensor<T> i_3    {{O,O},{1,1}};
  Tensor<T> i_3_1  {{O,V},{1,1}};
  Tensor<T> i_4    {{V,V},{1,1}};
  Tensor<T> i_4_1  {{O,V},{1,1}};
  Tensor<T> i_5    {{O,O,O,O},{2,2}};
  Tensor<T> i_5_1  {{O,O,O,V},{2,2}};
  Tensor<T> i_6    {{O,V,O,V},{2,2}};
  Tensor<T> i_7    {{O,O},{1,1}};
  Tensor<T> i_8    {{O,O,O,V},{2,2}};
  Tensor<T> i_9    {{O,O,O,V},{2,2}};
  Tensor<T> i_10   {{O,O,O,V},{2,2}};
  Tensor<T> i_11   {{V,V},{1,1}};
  Tensor<T> i_12   {{O,O,O,O},{2,2}};
  Tensor<T> i_12_1 {{O,O,O,V},{2,2}};
  Tensor<T> i_13   {{O,V,O,V},{2,2}};
  Tensor<T> i_13_1 {{O,O,O,V},{2,2}};

  Scheduler sch{ec};

  sch.allocate(i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5,
            i_5_1, i_6, i_7, i_8, i_9, i_10, i_11,
            i_12, i_12_1, i_13, i_13_1)
      ( i0(h3,h4,p1,p2)          =         v2(h3,h4,p1,p2)                         )
      (   i_1(h3,p1)             =         f1(h3,p1)                               )
      (   i_1(h3,p1)            +=         t1(p5,h6)         * v2(h3,h6,p1,p5)     )
      ( i0(h3,h4,p1,p2)         +=         y1(h3,p1)         * i_1(h4,p2)          )
      ( i0(h3,h4,p2,p1)         += -1.0  * y1(h3,p1)         * i_1(h4,p2)          ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         += -1.0  * y1(h3,p1)         * i_1(h4,p2)          ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         +=         y1(h3,p1)         * i_1(h4,p2)          ) //P(p1/p2,h3/h4)
      (   i_2(h3,h4,h7,p1)       =         v2(h3,h4,h7,p1)                         )
      (   i_2(h3,h4,h7,p1)      += -1    * t1(p5,h7)         * v2(h3,h4,p1,p5)     )
      ( i0(h3,h4,p1,p2)         += -1    * y1(h7,p1)         * i_2(h3,h4,h7,p2)    )
      ( i0(h3,h4,p2,p1)         +=         y1(h7,p1)         * i_2(h3,h4,h7,p2)    ) //P(p1/p2)
      ( i0(h3,h4,p1,p2)         += -1    * y1(h3,p5)         * v2(h4,p5,p1,p2)     )
      ( i0(h4,h3,p1,p2)         +=         y1(h3,p5)         * v2(h4,p5,p1,p2)     ) //P(h3/h4)
      (   i_3(h3,h9)             =         f1(h3,h9)                               )
      (     i_3_1(h3,p5)         =         f1(h3,p5)                               )
      (     i_3_1(h3,p5)        +=         t1(p7,h8)         * v2(h3,h8,p5,p7)     )
      (   i_3(h3,h9)            +=         t1(p5,h9)         * i_3_1(h3,p5)        )
      (   i_3(h3,h9)            +=         t1(p5,h6)         * v2(h3,h6,h9,p5)     )
      (   i_3(h3,h9)            += -0.5  * t2(p5,p6,h8,h9)   * v2(h3,h8,p5,p6)     )
      ( i0(h3,h4,p1,p2)         += -1    * y2(h3,h9,p1,p2)   * i_3(h4,h9)          )
      ( i0(h4,h3,p1,p2)         +=         y2(h3,h9,p1,p2)   * i_3(h4,h9)          ) //P(h3/h4)
      (   i_4(p10,p1)            =         f1(p10,p1)                              )
      (   i_4(p10,p1)           += -1    * t1(p5,h6)         * v2(h6,p10,p1,p5)    )
      (   i_4(p10,p1)           +=  0.5  * t2(p6,p10,h7,h8)  * v2(h7,h8,p1,p6)     )
      (     i_4_1(h6,p1)         =  0                                              )
      (     i_4_1(h6,p1)        +=         t1(p7,h8)         * v2(h6,h8,p1,p7)     )
      (   i_4(p10,p1)           += -1    * t1(p10,h6)        * i_4_1(h6,p1)        )
      ( i0(h3,h4,p1,p2)         +=         y2(h3,h4,p1,p10)  * i_4(p10,p2)         )
      ( i0(h3,h4,p2,p1)         += -1    * y2(h3,h4,p1,p10)  * i_4(p10,p2)         ) //P(p1/p2)
      (   i_5(h3,h4,h9,h10)      =         v2(h3,h4,h9,h10)                        )
      (     i_5_1(h3,h4,h10,p5)  =         v2(h3,h4,h10,p5)                        )
      (     i_5_1(h3,h4,h10,p5) += -0.5  * t1(p7,h10)        * v2(h3,h4,p5,p7)     )
      (   i_5(h3,h4,h9,h10)     += -2    * t1(p5,h9)         * i_5_1(h3,h4,h10,p5) )
      (   i_5(h3,h4,h9,h10)     +=  0.5  * t2(p5,p6,h9,h10)  * v2(h3,h4,p5,p6)     )
      ( i0(h3,h4,p1,p2)         +=  0.5  * y2(h9,h10,p1,p2)  * i_5(h3,h4,h9,h10)   )
      (   i_6(h3,p7,h9,p1)       =         v2(h3,p7,h9,p1)                         )
      (   i_6(h3,p7,h9,p1)      += -1    * t1(p5,h9)         * v2(h3,p7,p1,p5)     )
      (   i_6(h3,p7,h9,p1)      += -1    * t2(p6,p7,h8,h9)   * v2(h3,h8,p1,p6)     )
      ( i0(h3,h4,p1,p2)         += -1    * y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    )
      ( i0(h3,h4,p2,p1)         +=         y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         +=         y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         += -1    * y2(h3,h9,p1,p7)   * i_6(h4,p7,h9,p2)    ) //P(p1/p2,h3/h3)
      ( i0(h3,h4,p1,p2)         +=  0.5  * y2(h3,h4,p5,p6)   * v2(p5,p6,p1,p2)     )
      (   i_7(h3,h9)             =  0                                              )
      (   i_7(h3,h9)            +=         t1(p5,h9)         * y1(h3,p5)           )
      (   i_7(h3,h9)            += -0.5  * t2(p5,p6,h7,h9)   * y2(h3,h7,p5,p6)     )
      ( i0(h3,h4,p1,p2)         +=         i_7(h3,h9)        * v2(h4,h9,p1,p2)     )
      ( i0(h4,h3,p1,p2)         += -1    * i_7(h3,h9)        * v2(h4,h9,p1,p2)     ) //P(h3/h4)
      (   i_8(h3,h4,h5,p1)       =  0                                              )
      (   i_8(h3,h4,h5,p1)      += -1    * t1(p6,h5)         * y2(h3,h4,p1,p6)     )
      ( i0(h3,h4,p1,p2)         +=         i_8(h3,h4,h5,p1)  * f1(h5,p2)           )
      ( i0(h3,h4,p1,p2)         += -1    * i_8(h3,h4,h5,p1)  * f1(h5,p2)           ) //P(p1/p2)
      (   i_9(h3,h7,h6,p1)       =  0                                              )
      (   i_9(h3,h7,h6,p1)      +=         t1(p5,h6)         * y2(h3,h7,p1,p5)     )
      ( i0(h3,h4,p1,p2)         +=         i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     )
      ( i0(h3,h4,p2,p1)         += -1    * i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         += -1    * i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         +=         i_9(h3,h7,h6,p1)  * v2(h4,h6,h7,p2)     ) //P(p1/p2,h3/h3)
      (   i_10(h3,h4,h6,p7)      =  0                                              )
      (   i_10(h3,h4,h6,p7)     += -1    * t1(p5,h6)         * y2(h3,h4,p5,p7)     )
      ( i0(h3,h4,p1,p2)         +=         i_10(h3,h4,h6,p7) * v2(h6,p7,p1,p2)     )
      (   i_11(p6,p1)            =  0                                              )
      (   i_11(p6,p1)           +=         t2(p5,p6,h7,h8)   * y2(h7,h8,p1,p5)     )
      ( i0(h3,h4,p1,p2)         += -0.5  * i_11(p6,p1)       * v2(h3,h4,p2,p6)     )
      ( i0(h3,h4,p2,p1)         +=  0.5  * i_11(p6,p1)       * v2(h3,h4,p2,p6)     ) //P(p1/p2)
      (   i_12(h3,h4,h8,h9)      =  0                                              )
      (   i_12(h3,h4,h8,h9)     +=         t2(p5,p6,h8,h9)   * y2(h3,h4,p5,p6)     )
      (     i_12_1(h3,h4,h8,p5)  =  0                                              )
      (     i_12_1(h3,h4,h8,p5) += -1    * t1(p7,h8)         * y2(h3,h4,p5,p7)     )
      (   i_12(h3,h4,h8,h9)     +=  2    * t1(p5,h9)         * i_12_1(h3,h4,h8,p5) )
      ( i0(h3,h4,p1,p2)         +=  0.25 * i_12(h3,h4,h8,h9) * v2(h8,h9,p1,p2)     )
      (     i_13_1(h3,h6,h8,p1)  =  0                                              )
      (     i_13_1(h3,h6,h8,p1) +=         t1(p7,h8)         * y2(h3,h6,p1,p7)     )
      (   i_13(h3,p5,h8,p1)      =  0                                              )
      (   i_13(h3,p5,h8,p1)     +=         t1(p5,h6)         * i_13_1(h3,h6,h8,p1) )
      ( i0(h3,h4,p1,p2)         += -1    * i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     )
      ( i0(h3,h4,p2,p1)         +=         i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(p1/p2)
      ( i0(h4,h3,p1,p2)         +=         i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(h3/h4)
      ( i0(h4,h3,p2,p1)         += -1    * i_13(h3,p5,h8,p1) * v2(h4,h8,p2,p5)     ) //P(p1/p2,h3/h4)
   .deallocate(i_1, i_2, i_3, i_3_1, i_4, i_4_1, i_5, 
            i_5_1, i_6, i_7, i_8, i_9, i_10, i_11, 
            i_12, i_12_1, i_13, i_13_1).execute();                                           
}

template<typename T>
std::tuple<double,double> lambda_ccsd_driver(ExecutionContext& ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, Tensor<T>& d_v2,
                   Tensor<T>& d_r1, Tensor<T>& d_r2,
                   Tensor<T>& d_y1, Tensor<T>& d_y2,
                   std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, 
                   std::vector<Tensor<T>>& d_y1s, std::vector<Tensor<T>>& d_y2s, 
                   std::vector<T>& p_evl_sorted,
                   size_t maxiter, double thresh,
                   double zshiftl,
                   size_t ndiis, const TAMM_SIZE& noab) {

// TO DO: LAMBDA DOES NOT HAVE THE SAME ITERATION CONVERGENCE PROTOCOL
//        AND NEEDS TO BE UPDATED.

    std::cout.precision(15);

  double residual = 0.0;
  double energy = 0.0;

  for(size_t titer = 0; titer < maxiter; titer += ndiis) {
      for(size_t iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {

          const auto timer_start = std::chrono::high_resolution_clock::now();
          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(&ec, d_e, d_r1_residual, d_r2_residual);

          Scheduler sch{ec};
          
          sch(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
            .execute();

          sch
          ((d_y1s[off])() = d_y1())
          ((d_y2s[off])() = d_y2())
            .execute();

          lambda_ccsd_y1(ec, MO, d_r1, d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);
          lambda_ccsd_y2(ec, MO, d_r2, d_t1, d_t2, d_y1, d_y2, d_f1, d_v2);

          std::tie(residual, energy) = rest<T>(ec, MO, d_r1, d_r2, d_y1, d_y2,
                                            d_e, p_evl_sorted, zshiftl, noab,true);

          update_tensor(d_r2(), lambdar2);

          sch
          ((d_r1s[off])() = d_r1())
          ((d_r2s[off])() = d_r2())
            .execute();

          const auto timer_end = std::chrono::high_resolution_clock::now();
          auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

          iteration_print_lambda(ec.pg(), iter, residual, iter_time);
          Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

          if(residual < thresh) { break; }
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
      std::vector<std::vector<Tensor<T>>> ys{d_y1s, d_y2s};
      std::vector<Tensor<T>> next_y{d_y1, d_y2};
      diis<T>(ec, rs, ys, next_y);
  }

  return std::make_tuple(residual,energy);

}


void ccsd_driver();
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

    ccsd_driver();
    
    GA_Terminate();
    MPI_Finalize();

    return 0;
}


void ccsd_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext ec{pg, &distribution, mgr};
    auto rank = ec.pg().rank();

    //TODO: read from input file, assume no freezing for now
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    auto [options_map, ov_alpha, nao, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis] 
                    = hartree_fock_driver<T>(ec,filename);

    auto [MO,total_orbitals] = setupMOIS(nao,ov_alpha,freeze_core,freeze_virtual);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs] = cd_svd_driver<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map);


    CCSDOptions ccsd_options = options_map["CCSD"];
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

  auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s]
       = setupTensors(ec,MO,d_f1,ndiis);

  Tensor<T> d_v2 = setupV2<T>(ec,MO,cholVpr, chol_count, total_orbitals, ov_alpha, nao - ov_alpha);
  Tensor<T>::deallocate(cholVpr);

  auto cc_t1 = std::chrono::high_resolution_clock::now();
  auto [residual, energy] = ccsd_spin_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, 
                              d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, p_evl_sorted, 
                              maxiter, thresh, zshiftl, ndiis, 2 * ov_alpha);

  ccsd_stats(ec, hf_energy,residual,energy,thresh);

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for CCSD: " << ccsd_time << " secs\n";

  free_tensors(d_r1, d_r2);
  free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

  auto [l_r1,l_r2,d_y1,d_y2,l_r1s,l_r2s,d_y1s,d_y2s] = setupLambdaTensors<T>(ec,MO,ndiis);

  cc_t1 = std::chrono::high_resolution_clock::now();
  std::tie(residual,energy) = lambda_ccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1,
                        d_v2, l_r1,l_r2,d_y1, d_y2,l_r1s,l_r2s, d_y1s,d_y2s,
                        p_evl_sorted, maxiter, thresh, zshiftl, ndiis,
                        2 * ov_alpha);
  cc_t2 = std::chrono::high_resolution_clock::now();

  if(rank == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < thresh) {
        std::cout << " Iterations converged" << std::endl;
    }
  }

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for Lambda CCSD: " << ccsd_time << " secs\n"; 

  free_tensors(d_t1,d_t2,d_f1,d_v2,l_r1,l_r2,d_y1,d_y2);
  free_vec_tensors(l_r1s,l_r2s,d_y1s,d_y2s);

  ec.flush_and_sync();
  MemoryManagerGA::destroy_coll(mgr);
  // delete ec;
}
