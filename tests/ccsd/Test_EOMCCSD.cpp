// #define CATCH_CONFIG_RUNNER

#include "diis.hpp"
#include "eomguess.hpp"

using namespace tamm;


template<typename T>
void eomccsd_x1(ExecutionContext& ec, const TiledIndexSpace& MO,
               Tensor<T>& i0, const Tensor<T>& t1, const Tensor<T>& t2,
               const Tensor<T>& x1, const Tensor<T>& x2,
               const Tensor<T>& f1, const Tensor<T>& v2) {

   const TiledIndexSpace &O = MO("occ");
   const TiledIndexSpace &V = MO("virt");

   auto [h1, h3, h4, h5, h6, h8] = MO.labels<6>("occ");
   auto [p2, p3, p4, p5, p6, p7] = MO.labels<6>("virt");

   Tensor<T> i_1   {{O, O},{1,1}};
   Tensor<T> i_1_1 {{O, V},{1,1}};
   Tensor<T> i_2   {{V, V},{1,1}};
   Tensor<T> i_3   {{O, V},{1,1}};
   Tensor<T> i_4   {{O, O, O, V},{2,2}};
   Tensor<T> i_5_1 {{O, V},{1,1}};
   Tensor<T> i_5   {{O, O},{1,1}};
   Tensor<T> i_5_2 {{O, V},{1,1}};
   Tensor<T> i_6   {{V, V},{1,1}};
   Tensor<T> i_7   {{O, V},{1,1}};
   Tensor<T> i_8   {{O, O, O, V},{2,2}};

   Scheduler sch{ec};

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

   Tensor<T> i_1     {{O, V, O, O},{2,2}};
   Tensor<T> i_1_1   {{O, V, O, V},{2,2}};
   Tensor<T> i_1_2   {{O, V},{1,1}};
   Tensor<T> i_1_3   {{O, O, O, V},{2,2}};
   Tensor<T> i_2     {{O, O},{1,1}};
   Tensor<T> i_2_1   {{O, V},{1,1}};
   Tensor<T> i_3     {{V, V},{1,1}};
   Tensor<T> i_4     {{O, O, O, O},{2,2}};
   Tensor<T> i_4_1   {{O, O, O, V},{2,2}};
   Tensor<T> i_5     {{O, V, O, V},{2,2}};
   Tensor<T> i_6_1   {{O, O, O, O},{2,2}};
   Tensor<T> i_6_1_1 {{O, O, O, V},{2,2}};
   Tensor<T> i_6     {{O, V, O, O},{2,2}};
   Tensor<T> i_6_2   {{O, V},{1,1}};
   Tensor<T> i_6_3   {{O, O, O, V},{2,2}};
   Tensor<T> i_6_4   {{O, O, O, O},{2,2}};
   Tensor<T> i_6_4_1 {{O, O, O, V},{2,2}};
   Tensor<T> i_6_5   {{O, V, O, V},{2,2}};
   Tensor<T> i_6_6   {{O, V},{1,1}};
   Tensor<T> i_6_7   {{O, O, O, V},{2,2}};
   Tensor<T> i_7     {{V, V, O, V},{2,2}};
   Tensor<T> i_8_1   {{O, V},{1,1}};
   Tensor<T> i_8     {{O, O},{1,1}};
   Tensor<T> i_8_2   {{O, V},{1,1}};
   Tensor<T> i_9     {{O, O, O, O},{2,2}};
   Tensor<T> i_9_1   {{O, O, O, V},{2,2}};
   Tensor<T> i_10    {{V, V},{1,1}};
   Tensor<T> i_11    {{O, V, O, V},{2,2}};

   Scheduler sch{ec};

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
void eomccsd_driver(ExecutionContext& ec, const TiledIndexSpace& MO,
                   Tensor<T>& t1, Tensor<T>& t2,
                   Tensor<T>& f1, Tensor<T>& v2,
                   std::vector<T> p_evl_sorted,
                   int nroots, int maxeomiter,
                   double eomthresh, int microeomiter,
                   long int total_orbitals, const TAMM_SIZE& noab) {

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");

  std::cout.precision(15);

  Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  auto populate_vector_of_tensors = [&] (std::vector<Tensor<T>> &vec, bool is2D=true){
      for(auto x=0;x<vec.size();x++){
         if(is2D) vec[x] = Tensor<T>{{V,O},{1,1}};
         else     vec[x] = Tensor<T>{{V,V,O,O},{2,2}};
         Tensor<T>::allocate(&ec,vec[x]);
      }
  };


//FOR JACOBI STEP
  double zshiftl = 0;
  bool transpose=false;

//INITIAL GUESS WILL MOVE TO HERE
//TO DO: NXTRIALS IS SET TO NROOTS BECAUSE WE ONLY ALLOW #NROOTS# INITIAL GUESSES
//       WHEN THE EOM_GUESS ROUTINE IS UPDATED TO BE LIKE THE INITIAL GUESS IN TCE
//       THEN THE FOLLOWING LINE WILL BE "INT NXTRIALS = INITVECS", WHERE INITVEC
//       IS THE NUMBER OF INITIAL VECTORS. THE {X1,X2} AND {XP1,XP2} TENSORS WILL
//       BE OF DIMENSION (ninitvecs + NROOTS*(MICROEOMITER-1)) FOR THE FIRST
//       MICROCYCLE AND (NROOTS*MICROEOMITER) FOR THE REMAINING.

  int ninitvecs = nroots;
  const auto hbardim = ninitvecs + nroots*(microeomiter-1);

  Matrix hbar = Matrix::Zero(hbardim,hbardim);
  Matrix hbar_right;

  Tensor<T> u1{{V, O},{1,1}};
  Tensor<T> u2{{V, V, O, O},{2,2}};
  Tensor<T> uu2{{V, V, O, O},{2,2}};
  Tensor<T> uuu2{{V, V, O, O},{2,2}};
  Tensor<T>::allocate(&ec,u1,u2,uu2,uuu2);

  using std::vector;

  vector<Tensor<T>> x1(hbardim);
  populate_vector_of_tensors(x1);
  vector<Tensor<T>> x2(hbardim);
  populate_vector_of_tensors(x2,false);
  vector<Tensor<T>> xp1(hbardim);
  populate_vector_of_tensors(xp1);
  vector<Tensor<T>> xp2(hbardim);
  populate_vector_of_tensors(xp2,false);
  vector<Tensor<T>> xc1(nroots);
  populate_vector_of_tensors(xc1);
  vector<Tensor<T>> xc2(nroots);
  populate_vector_of_tensors(xc2,false);
  vector<Tensor<T>> r1(nroots);
  populate_vector_of_tensors(r1);
  vector<Tensor<T>> r2(nroots);
  populate_vector_of_tensors(r2,false);

  Tensor<T> d_r1{};
  Tensor<T> oscalar{};
  Tensor<T>::allocate(&ec, d_r1, oscalar);
  bool convflag=false;
  double au2ev = 27.2113961;

//################################################################################
//  CALL THE EOM_GUESS ROUTINE (EXTERNAL ROUTINE)
//################################################################################

  eom_guess(nroots,noab,p_evl_sorted,x1);

//################################################################################
//  PRINT THE HEADER FOR THE EOM ITERATIONS
//################################################################################

  if(ec.pg().rank() == 0){
    std::cout << "\n\n";
    std::cout << " No. of initial right vectors " << ninitvecs << std::endl;
    std::cout << "\n";
    std::cout << " EOM-CCSD right-hand side iterations" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    std::cout <<
        "     Residuum       Omega / hartree  Omega / eV    Cpu    Wall"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;
  }

//################################################################################
//  MAIN ITERATION LOOP
//################################################################################

  for(int iter = 0; iter < maxeomiter;){

     int nxtrials = 0;
     int newnxtrials = ninitvecs;

     for(int micro = 0; micro < microeomiter; iter++, micro++){

        for(int root= nxtrials; root < newnxtrials; root++){

           eomccsd_x1(ec, MO, xp1.at(root), t1, t2, x1.at(root), x2.at(root), f1, v2);
           eomccsd_x2(ec, MO, xp2.at(root), t1, t2, x1.at(root), x2.at(root), f1, v2);

        }

//################################################################################
//  UPDATE HBAR: ELEMENTS FOR THE NEWEST X AND XP VECTORS ARE COMPUTED
//################################################################################

        if(micro == 0){

           for(int ivec = 0; ivec < newnxtrials; ivec++){

              for(int jvec = 0; jvec < newnxtrials; jvec++){

                 sch(u2()  = xp2.at(jvec)())
                    (uu2() = x2.at(ivec)()).execute();

                 update_tensor(u2(), lambdar2);
                 update_tensor(uu2(), lambdar2);

                 sch(d_r1()  = 0)
                    (d_r1() += xp1.at(jvec)() * x1.at(ivec)())
                    (d_r1() += u2() * uu2()).execute();
                 T r1;
                 d_r1.get({}, {&r1, 1});

                 hbar(ivec,jvec) = r1;
              }
           }
        } else {

           for(int ivec = 0; ivec < newnxtrials; ivec++){

              for(int jvec = nxtrials; jvec < newnxtrials; jvec++){

                 sch(u2()  = xp2.at(jvec)())
                    (uu2() = x2.at(ivec)()).execute();

                    update_tensor(u2(), lambdar2);
                    update_tensor(uu2(), lambdar2);

                    sch(d_r1()  = 0)
                       (d_r1() += xp1.at(jvec)() * x1.at(ivec)())
                       (d_r1() += u2() * uu2()).execute();
                    T r1;
                    d_r1.get({}, {&r1, 1});

                    hbar(ivec,jvec) = r1;
              }
           }

           for(int ivec = nxtrials; ivec < newnxtrials; ivec++){
              for(int jvec = 0; jvec < nxtrials; jvec++){

                 sch(u2()  = xp2.at(jvec)())
                    (uu2() = x2.at(ivec)()).execute();

                 update_tensor(u2(), lambdar2);
                 update_tensor(uu2(), lambdar2);

                 sch(d_r1()  = 0)
                    (d_r1() += xp1.at(jvec)() * x1.at(ivec)())
                    (d_r1() += u2() * uu2()).execute();
                 T r1;
                 d_r1.get({}, {&r1, 1});

                 hbar(ivec,jvec) = r1;
              }
           }
        }

//################################################################################
//  DIAGONALIZE HBAR
//################################################################################

  Eigen::EigenSolver<Matrix> hbardiag(hbar.block(0,0,newnxtrials,newnxtrials));
  auto omegar1 = hbardiag.eigenvalues();

  const auto nev = omegar1.rows();
  std::vector<T> omegar(nev);
  for (auto x=0; x<nev;x++)
  omegar[x] = real(omegar1(x));

//################################################################################
//  SORT THE EIGENVECTORS AND CORRESPONDING EIGENVALUES
//################################################################################

  std::vector<size_t> omegar_sorted_order = sort_indexes(omegar);
  std::sort(omegar.begin(), omegar.end());

  auto hbar_right1 = hbardiag.eigenvectors();
  assert(hbar_right1.rows() == nev && hbar_right1.cols() == nev);
  hbar_right.resize(nev,nev);
  hbar_right.setZero();

  for(auto x=0;x<nev;x++)
     hbar_right.col(x) = hbar_right1.col(omegar_sorted_order[x]).real();

        if(ec.pg().rank() == 0) {
          std::cout << "\n";
          std::cout << " Iteration " << iter+1 << " using "
                    << newnxtrials << " trial vectors"<< std::endl;
        }

        nxtrials = newnxtrials;

        for(auto root = 0; root < nroots; root++){
//################################################################################
//  FORM RESIDUAL VECTORS
//################################################################################

           sch(r1.at(root)()  = 0)
              (r2.at(root)()  = 0).execute();

           for(int i = 0; i < nxtrials; i++){

              T omegar_hbar_scalar = -1 * omegar[root] * hbar_right(i,root);

              sch(r1.at(root)() += omegar_hbar_scalar * x1.at(i)())
                 (r2.at(root)() += omegar_hbar_scalar * x2.at(i)())
                 (r1.at(root)() += hbar_right(i,root) * xp1.at(i)())
                 (r2.at(root)() += hbar_right(i,root) * xp2.at(i)()).execute();
           }

           sch(u1() = r1.at(root)())
              (u2() = r2.at(root)()).execute();

           update_tensor(u2(), lambdar2);

           sch(oscalar() = 0)
              (oscalar() += u1() * u1())
              (oscalar() += u2() * u2()).execute();

           T tmps = get_scalar(oscalar);
           T xresidual = sqrt(tmps);
           T newsc = 1/sqrt(tmps);

           if(ec.pg().rank() == 0) {
             std::cout.precision(13);
             std::cout << "   " << xresidual << "   " << omegar[root]
                       << "    "
                       << omegar[root]*au2ev << std::endl;
           }

//################################################################################
//  EXPAND ITERATIVE SPACE WITH NEW ORTHONORMAL VECTORS
//################################################################################
           if(xresidual > eomthresh){
             int ivec=newnxtrials;
             newnxtrials++;

             if(newnxtrials <= hbardim){
               sch(u1() = 0 )
                  (u2() = 0 ).execute();

               jacobi(ec, r1.at(root), u1, 0.0, false, p_evl_sorted, noab);
               jacobi(ec, r2.at(root), u2, 0.0, false, p_evl_sorted, noab);

               sch(x1.at(ivec)() = newsc * u1())
                  (x2.at(ivec)() = newsc * u2()).execute();

               sch(u1() = x1.at(ivec)())
                  (u2() = x2.at(ivec)())
                  (uu2() = x2.at(ivec)()).execute();

                  update_tensor(uu2(), lambdar2);

               for(int jvec = 0; jvec<ivec; jvec++){

                  sch(uuu2() = x2.at(jvec)()).execute();

                  update_tensor(uuu2(), lambdar2);

                  sch(oscalar() = 0)
                     (oscalar() += x1.at(ivec)() * x1.at(jvec)())
                     (oscalar() += uu2() * uuu2()).execute();

                  T tmps = get_scalar(oscalar);

                  sch(u1() += -1 * tmps * x1.at(jvec)())
                     (u2() += -1 * tmps * x2.at(jvec)()).execute();
               }

               sch(x1.at(ivec)() = u1())
                  (x2.at(ivec)() = u2()).execute();

               update_tensor(u2(), lambdar2);

               sch(oscalar() = 0)
                  (oscalar() += u1() * u1())
                  (oscalar() += u2() * u2()).execute();

               sch(u1() = x1.at(ivec)())
                  (u2() = x2.at(ivec)()).execute();

               T tmps = get_scalar(oscalar);
               T newsc = 1/sqrt(tmps);

               sch(x1.at(ivec)() = 0)
                  (x2.at(ivec)() = 0)
                  (x1.at(ivec)() += newsc * u1())
                  (x2.at(ivec)() += newsc * u2()).execute();

             }
           }
        }

//################################################################################
//  CHECK CONVERGENGE
//################################################################################
        if(nxtrials == newnxtrials){
           if(ec.pg().rank() == 0) {
             std::cout << std::string(62, '-') << std::endl;
             std::cout << " Iterations converged" << std::endl;
           }
           convflag = true;
           for(auto root = 0; root < nroots; root++){

              sch(xc1.at(root)() = 0)
                 (xc2.at(root)() = 0).execute();

              for(int i = 0; i < nxtrials; i++){

                 T hbr_scalar = hbar_right(i,root);

                 sch(xc1.at(root)() += hbr_scalar * x1.at(i)())
                    (xc2.at(root)() += hbr_scalar * x2.at(i)()).execute();
              }
           }
           break;
        }

     } //END MICRO

  if(convflag) break;

//################################################################################
//  FORM INITAL VECTORS FOR NEWEST MICRO INTERATIONS
//################################################################################
  if(ec.pg().rank() == 0){
    std::cout << " END OF MICROITERATIONS: COLLAPSING WITH NEW INITAL VECTORS" << std::endl;
  }

  ninitvecs = nroots;

  for(auto root = 0; root < nroots; root++){

     sch(xc1.at(root)() = 0)
        (xc2.at(root)() = 0).execute();

     for(int i = 0; i < nxtrials; i++){

        T hbr_scalar = hbar_right(i,root);

        sch(xc1.at(root)() += hbr_scalar * x1.at(i)())
           (xc2.at(root)() += hbr_scalar * x2.at(i)()).execute();
     }
  }

  for(auto root = 0; root < nroots; root++){

     sch(x1.at(root)() = xc1.at(root)())
        (x2.at(root)() = xc2.at(root)()).execute();

  }

  }

  Tensor<T>::deallocate(u1,u2,uu2,uuu2);
  Tensor<T>::deallocate(d_r1);
  Tensor<T>::deallocate(oscalar);

  free_vec_tensors(x1, x2, xp1, xp2, xc1, xc2, r1, r2);

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


    CCSDOptions ccsd_options = options_map.ccsd_options;
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

  auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
                        = setupTensors(ec,MO,d_f1,ndiis);

  Tensor<T> d_v2 = setupV2<T>(ec,MO,cholVpr,chol_count, total_orbitals, ov_alpha, nao - ov_alpha);
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

  free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

//EOMCCSD Variables
    int nroots           = 4;
    int maxeomiter       = 50;
//    int eomsolver        = 1; //INDICATES WHICH SOLVER TO USE. (LATER IMPLEMENTATION)
    double eomthresh     = 1.0e-10;
//    double x2guessthresh = 0.6; //THRESHOLD FOR X2 INITIAL GUESS (LATER IMPLEMENTATION)
    size_t microeomiter  = 25; //Number of iterations in a microcycle


//EOMCCSD Routine:
  cc_t1 = std::chrono::high_resolution_clock::now();

  eomccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, p_evl_sorted,
                      nroots, maxeomiter, eomthresh, microeomiter,
                      total_orbitals, 2 * ov_alpha);

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank==0) std::cout << "\nTime taken for EOMCCSD: " << ccsd_time << " secs\n";

  free_tensors(d_r1, d_r2, d_t1, d_t2, d_f1, d_v2);
  
  ec.flush_and_sync();
  MemoryManagerGA::destroy_coll(mgr);
//   delete ec;
}




