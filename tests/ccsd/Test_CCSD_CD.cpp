#define CATCH_CONFIG_RUNNER

#include "diis.hpp"
#include "4index_transform_CD.hpp"
#include "catch/catch.hpp"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "ga-mpi.h"


using namespace tamm;

template<typename T>
void ccsd_e(ExecutionContext &ec,
            const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, std::vector<Tensor<T> *> &chol){ //, const Tensor<T>& v2) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{{O,V},{1,1}};
    Tensor<T> _a01{};
    Tensor<T> _a02{{O,O},{1,1}};
    Tensor<T> _a03{{O,V},{1,1}};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
    std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

    Scheduler sch{ec};
    sch.allocate(i1,_a01,_a02,_a03);
    //sch (i1(h6, p5) = f1(h6, p5));
    //sch (i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5))
    
    sch (de() = 0);
    for(auto x = 0U; x < chol.size(); x++) {
        Tensor<T>& cholx = (*(chol.at(x)));
        sch (_a01() = 0)
            (_a02(h4, h6) = 0)
            (_a03(h4, p2) = 0)
            (_a01() += t1(p3, h4) * cholx(h4, p3))
            (_a02(h4, h6) += t1(p3, h4) * cholx(h6, p3))
            (de() +=  0.5 * _a01() * _a01())
            (de() += -0.5 * _a02(h4, h6) * _a02(h6, h4))
            (_a03(h4, p2) += t2(p1, p2, h3, h4) * cholx(h3, p1))
            (de() += 0.5 * _a03(h4, p1) * cholx(h4, p1));
    }
    sch.deallocate(i1,_a01,_a02,_a03);
    sch.execute();
}

template<typename T>
void ccsd_t1(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& f1,
             std::vector<Tensor<T> *> &chol) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> t1_2_1{{O,O},{1,1}};
 
    Tensor<T> _a007{{O,V},{1,1}};
    Tensor<T> _a001{{O,O},{1,1}};
    Tensor<T> _a002{};
    Tensor<T> _a005{{O,O},{1,1}};
    Tensor<T> _a003{{V,O},{1,1}};

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8) = MO.labels<8>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8) = MO.labels<8>("occ");

    Scheduler sch{ec};
    sch.allocate(t1_2_1, _a007, _a001, _a002, _a005, _a003);

    sch(i0(p2, h1) = f1(p2, h1));

    for(auto x = 0U; x < chol.size(); x++) {
        Tensor<T>& cholx = (*(chol.at(x)));
        sch 
          (_a001(h2, h1) = 0)
          (_a005(h2, h1) = 0)
          (_a002()       = 0)
          (_a003(p1, h1) = 0)
          (_a007(h2, p1) = 0);

        sch
          (_a001(h2, h1) +=  1.0 * t1(p1, h1) * cholx(h2, p1))
          (_a003(p1, h1) +=  1.0 * t2(p1, p3, h2, h1) * cholx(h2, p3))
          (_a002()       +=  1.0 * t1(p3, h3) * cholx(h3, p3))

          (_a005(h2, h1) +=  1.0 * cholx(h2, p1) * _a003(p1, h1))
          (i0(p2, h1)    +=  1.0 * t1(p2, h2) * _a005(h2, h1))//t1-ch-t2-ch
          
          (i0(p1, h2)    +=  1.0 * cholx(p1, h2) * _a002())//ch
          
          (_a007(h2, p1) += -1.0 * cholx(h3, p1) * _a001(h2, h3))
          (i0(p2, h1)    +=  1.0 * t2(p1, p2, h2, h1) * _a007(h2, p1))//t2-ch-t1-ch
          
          (i0(p2, h1)    += -1.0 * cholx(p2, p1) * _a003(p1, h1))//ch-t2-ch
          
          (_a003(p2, h2) += -1.0 * t1(p1, h2) * cholx(p2, p1))
          (i0(p1, h2)    += -1.0 * _a003(p1, h2) * _a002())//t2-ch + t1-ch
          
          (_a003(p2, h3) += -1.0 * t1(p2, h3) * _a002())
          (_a003(p2, h3) +=  1.0 * t1(p2, h2) * _a001(h2, h3))
          (_a001(h3, h1) +=  1.0 * cholx(h3, h1))
          (i0(p2, h1)    +=  1.0 * _a001(h3, h1) * _a003(p2, h3)) //
          
          ;
    }

    sch(t1_2_1(h7, h1) = f1(h7, h1))
    (t1_2_1(h7, h1) += t1(p3, h1) * f1(h7, p3))
    (i0(p2, h1) += -1 * t1(p2, h7) * t1_2_1(h7, h1))
    (i0(p2, h1) += t1(p3, h1) * f1(p2, p3));
    sch(i0(p2, h1) += t2(p2, p7, h1, h8) * f1(h8, p7));

    sch.deallocate(t1_2_1, _a007, _a001, _a002, _a005, _a003);
    sch.execute();
}

template<typename T>
void ccsd_t2(ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& i0,
             const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& f1,
             std::vector<Tensor<T> *> &chol) {
    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");

    TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
    TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

    std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9) = MO.labels<9>("virt");
    std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

    Scheduler sch{ec};

    Tensor<T> _a001{{V,V}, {1,1}};
    Tensor<T> _a004{{V,O,O,O}, {2,2}};
    Tensor<T> _a006{{O,O}, {1,1}};
    Tensor<T> _a007{};
    Tensor<T> _a008{{O,O}, {1,1}};
    Tensor<T> _a009{{O,O}, {1,1}};
    Tensor<T> _a017{{V,O}, {1,1}};
    Tensor<T> _a019{{O,O,O,O}, {2,2}};
    Tensor<T> _a020{{V,O,V,O}, {2,2}};
    Tensor<T> _a021{{V,V}, {1,1}};
    Tensor<T> _a022{{V,V,O,O}, {2,2}};
    Tensor<T> i0_temp{{V,V,O,O}, {2,2}};
    
 //------------------------------CD------------------------------
    sch.allocate(_a001, _a004, _a006, _a007, 
                 _a008, _a009, _a017, _a019, _a020, _a021,
                 _a022,
                 i0_temp);

    sch (_a001(p1, p2) = 0)
        (_a006(h3, h2) = 0)
        (_a019(h3, h4, h1, h2) = 0)
        (_a020(p3, h3, p4, h2) = 0)
        (i0(p3, p4, h1, h2) = 0)
        (i0_temp(p3, p4, h1, h2) = 0)
        ;
    
    for(auto x = 0U; x < chol.size(); x++) {
        Tensor<T>& cholx = (*(chol.at(x)));

        sch (_a007() = 0)
            (_a008(h3, h1) = 0)
            (_a009(h3, h2) = 0)
            (_a017(p3, h2) = 0)
            (_a021(p3, p1) = 0)
            (_a004(p1, h4, h1, h2) = 0)
            (_a022(p1, p4, h1, h2) = 0)
            ;

        sch (_a017(p3, h2) += -1.0 * t2(p1, p3, h3, h2) * cholx(h3, p1))
            (_a006(h4, h1) += -1.0 * cholx(h4, p2) * _a017(p2, h1))
            (_a007()       +=  1.0 * cholx(h4, p1) * t1(p1, h4))
            (_a009(h3, h2) +=  1.0 * cholx(h3, p1) * t1(p1, h2))
            (_a021(p3, p1) += -0.5 * cholx(h3, p1) * t1(p3, h3))
            (_a021(p3, p1) +=  0.5 * cholx(p3, p1))
            (_a017(p3, h2) += -2.0 * t1(p2, h2) * _a021(p3, p2))
            (_a008(h3, h1) +=  1.0 * _a009(h3, h1))//t1
            (_a009(h3, h1) +=  1.0 * cholx(h3, h1))
            ;
            
        sch (_a001(p4, p2) += -2.0 * _a021(p4, p2) * _a007())
            (_a001(p4, p2) += -1.0 * _a017(p4, h2) * cholx(h2, p2))
            (_a006(h4, h1) +=  1.0 * _a009(h4, h1) * _a007())
            (_a006(h4, h1) += -1.0 * _a009(h3, h1) * _a008(h4, h3))
            (_a019(h4, h3, h1, h2) +=  0.25 * _a009(h4, h1) * _a009(h3, h2)) 
            (_a020(p4, h4, p1, h1) += -2.0  * _a009(h4, h1) * _a021(p4, p1))
            ;
        
        sch (_a017(p3, h2) +=  1.0 * t1(p3, h3) * cholx(h3, h2))
            (_a017(p3, h2) += -1.0 * cholx(p3, h2))
            (i0_temp(p3, p4, h1, h2) +=  0.5 * _a017(p3, h1) * _a017(p4, h2))
            
            (_a022(p2, p3, h1, h2)   += 1.0 * t2(p2, p1, h1, h2) * _a021(p3, p1))
            (i0_temp(p3, p4, h1, h2) += 1.0 * _a021(p3, p1) * _a022(p1, p4, h1, h2))
            
            (_a004(p1, h3, h1, h2) +=  1.0   * cholx(p2, h3) * t2(p1, p2, h1, h2))
            (_a019(h3, h4, h1, h2) += -0.125 * cholx(p1, h4) * _a004(p1, h3, h1, h2))
            (_a020(p3, h1, p4, h2) +=  0.5   * cholx(p4, h4) * _a004(p3, h1, h4, h2))
            ;            
    }

    // 
    sch (_a001(p4, p1) += -1 * f1(p4, p1))
        (i0_temp(p3, p4, h1, h2) += -0.5 * t2(p3, p2, h1, h2) * _a001(p4, p2))

        (i0_temp(p3, p4, h1, h2) +=  1.0 * _a019(h4, h3, h1, h2) * t2(p3, p4, h4, h3))
        
        (i0_temp(p3, p4, h1, h2) +=  1.0 * _a020(p4, h4, p1, h1) * t2(p3, p1, h4, h2))
        ;

    sch (_a006(h9, h1) += f1(h9, h1))
        (_a006(h9, h1) += t1(p8, h1) * f1(h9, p8));

    sch (i0_temp(p3, p4, h2, h1) += -0.5 * t2(p3, p4, h3, h1) * _a006(h3, h2))
        (i0(p3, p4, h1, h2) +=  1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p3, p4, h2, h1) += -1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p4, p3, h1, h2) += -1.0 * i0_temp(p3, p4, h1, h2))
        (i0(p4, p3, h2, h1) +=  1.0 * i0_temp(p3, p4, h1, h2))    
        ;
 
    
    //sch(_a009(p3, p5) = 0)
    //   (_a009(p3, p5) +=  1.0 * t1(p3, h10) * f1(h10, p5))
    //   (i0(p3, p4, h1, h2) += -1 * _a009(p3, p5) * t2(p4, p5, h1, h2))
    //   (i0(p4, p3, h1, h2) +=  1 * _a009(p3, p5) * t2(p4, p5, h1, h2));

  sch.deallocate(_a001, _a004, _a006, _a007, 
                 _a008, _a009, _a017, _a019, _a020, _a021,
                 _a022,
                 i0_temp);
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
    Scheduler sch{ec};
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

void iteration_print(const ProcGroup& pg, int iter, double residual, double energy, double time) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << time;
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
  }
}

template<typename T>
void ccsd_driver(ExecutionContext& ec, const TiledIndexSpace& MO,
                   Tensor<T>& d_t1, Tensor<T>& d_t2,
                   Tensor<T>& d_f1, //Tensor<T>& d_v2,
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
  if(ec.pg().rank() == 0) std::cout << "Total orbitals = " << total_orbitals << std::endl;
  std::vector<double> p_evl_sorted(total_orbitals);

    // Tensor<T> d_evl{N};
    // Tensor<T>::allocate(ec, d_evl);
    // TiledIndexLabel n1;
    // std::tie(n1) = MO.labels<1>("all");

    // sch(d_evl(n1) = 0.0)
    // .execute();

    auto lambda = [&](Tensor<T> tensor, const IndexVector& blockid, span<T> buf){
        if(blockid[0] == blockid[1]) {
            auto block_dims = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);
            auto dim    = block_dims[0];
            auto offset = block_offset[0];
            TAMM_SIZE i = 0;
            for(auto p = offset; p < offset + dim; p++, i++)
                p_evl_sorted[p] = buf[i * dim + i]; 
        }
    };
    update_tensor_general(d_f1(), lambda);

//   {
//       auto lambda = [&](const IndexVector& blockid) {
//           if(blockid[0] == blockid[1]) {
//               Tensor<T> tensor     = d_f1;
//               const TAMM_SIZE size = tensor.block_size(blockid);

//               std::vector<T> buf(size);
//               tensor.get(blockid, buf);

//             auto block_dims = tensor.block_dims(blockid);
//             auto block_offset = tensor.block_offsets(blockid);


//               auto dim    = block_dims[0];
//               auto offset = block_offset[0];
//               TAMM_SIZE i = 0;
//               for(auto p = offset; p < offset + dim; p++, i++) {
//                   p_evl_sorted[p] = buf[i * dim + i];
//               }
//           }
//       };
//       block_for(ec, d_f1(), lambda);
//   }
//   ec.pg().barrier();

//   if(ec->pg().rank() == 0) {
//     std::cout << "p_evl_sorted:" << '\n';
//     for(size_t p = 0; p < p_evl_sorted.size(); p++)
//       std::cout << p_evl_sorted[p] << '\n';
//   }

  if(ec.pg().rank() == 0) {
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
    Tensor<T>::allocate(&ec,*d_r1s[i], *d_r2s[i], *d_t1s[i], *d_t2s[i]);
  }
 
  Tensor<T> d_r1{V,O};
  Tensor<T> d_r2{V,V,O,O};
  Tensor<T>::allocate(&ec,d_r1, d_r2);

  Scheduler{ec}   
  (d_r1() = 0)
  (d_r2() = 0)
  .execute();

  double residual = 0.0;
  double energy = 0.0;

auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
    if(blockid[0] != blockid[1]) {
        for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
    }
};

update_tensor(d_f1(),lambda2);

auto lambdar2 = [&](const IndexVector& blockid, span<T> buf){
    if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
        for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
    }
};

  for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          
          const auto timer_start = std::chrono::high_resolution_clock::now();

          int off = iter - titer;

          Tensor<T> d_e{};
          Tensor<T> d_r1_residual{};
          Tensor<T> d_r2_residual{};

          Tensor<T>::allocate(&ec, d_e, d_r1_residual, d_r2_residual);

          Scheduler{ec}(d_e() = 0)(d_r1_residual() = 0)(d_r2_residual() = 0)
            .execute();

          Scheduler{ec}((*d_t1s[off])() = d_t1())((*d_t2s[off])() = d_t2())
            .execute();

          ccsd_e(ec, MO, d_e, d_t1, d_t2, d_f1, chol);
          ccsd_t1(ec, MO, d_r1, d_t1, d_t2, d_f1, chol);
          ccsd_t2(ec, MO, d_r2, d_t1, d_t2, d_f1, chol);

          GA_Sync();
          std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2,
                                            d_e, p_evl_sorted, zshiftl, noab);

          update_tensor(d_r2(), lambdar2);
          

          Scheduler{ec}((*d_r1s[off])() = d_r1())((*d_r2s[off])() = d_r2())
            .execute();

          const auto timer_end = std::chrono::high_resolution_clock::now();
          auto iter_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

          iteration_print(ec.pg(), iter, residual, energy, iter_time);
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

      std::vector<std::vector<Tensor<T>*>*> rs{&d_r1s, &d_r2s};
      std::vector<std::vector<Tensor<T>*>*> ts{&d_t1s, &d_t2s};
      std::vector<Tensor<T>*> next_t{&d_t1, &d_t2};
      diis<T>(ec, rs, ts, next_t);
  }

  if(ec.pg().rank() == 0) {
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

  for(auto i=0; i<ndiis; i++) {
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

    auto rank = GA_Nodeid();
    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    Matrix C;
    Matrix F;
    // Tensor4D V2;
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
    if(rank == 0) std::cout << "\nTime taken for Hartree-Fock: " << hf_time << " secs\n";

    hf_t1        = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::RowVectorXd> evec;
    Tensor3D Vpsigma;
    //std::tie(V2) = 
    four_index_transform(ov_alpha, nao, freeze_core,
                                        freeze_virtual, C, F, shells, CholVpr, evec, Vpsigma);
    hf_t2        = std::chrono::high_resolution_clock::now();
    double two_4index_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for 4-index transform: " << two_4index_time
              << " secs\n";

    TAMM_SIZE ov_beta{nao - ov_alpha};

    // std::cout << "ov_alpha,nao === " << ov_alpha << ":" << nao << std::endl;
    sizes = {ov_alpha - freeze_core, ov_alpha - freeze_core,
             ov_beta - freeze_virtual, ov_beta - freeze_virtual};

    // std::cout << "sizes vector -- \n";
    // for(const auto& x : sizes) std::cout << x << ", ";
    // std::cout << "\n";

    const long int total_orbitals = 2*ov_alpha+2*ov_beta;
    
    // Construction of tiled index space MO

    IndexSpace MO_IS{range(0, total_orbitals),
                    {
                     {"occ", {range(0, 2*ov_alpha)}},
                     {"virt", {range(2*ov_alpha, total_orbitals)}}
                    },
                     { 
                      {Spin{1}, {range(0, ov_alpha), range(2*ov_alpha,2*ov_alpha+ov_beta)}},
                      {Spin{2}, {range(ov_alpha, 2*ov_alpha), range(2*ov_alpha+ov_beta, total_orbitals)}} 
                     }
                     };

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

    Tensor<T> d_t1{{V,O},{1,1}};
    Tensor<T> d_t2{{V,V,O,O},{2,2}};
    Tensor<T> d_f1{{N,N},{1,1}};
    // Tensor<T> d_v2{{N,N,N,N},{2,2}};
    int maxiter    = 50;
    double thresh  = 1.0e-10;
    double zshiftl = 0.0;
    size_t ndiis      = 5;

  Tensor<double>::allocate(ec,d_t1,d_t2,d_f1);//,d_v2);

  Scheduler{*ec}
      (d_t1() = 0)
      (d_t2() = 0)
      (d_f1() = 0)
    //   (d_v2() = 0)
    .execute();

  // CD
  auto chol_dims = CholVpr.dimensions();
  auto chol_count = chol_dims[2];
  if(rank == 0) cout << "Number of cholesky vectors:" << chol_count << endl;
  std::vector<Tensor<T> *> chol_vecs(chol_count);

  for(auto x = 0; x < chol_count; x++) {
      Tensor<T>* cholvec = new Tensor<T>{{N,N},{1,1}};
      Tensor<T>::allocate(ec, *cholvec);
      Scheduler{*ec}((*cholvec)() = 0).execute();
      chol_vecs[x] = cholvec;
  }

  //Tensor Map 
  eigen_to_tamm_tensor(d_f1, F);
//   eigen_to_tamm_tensor(d_v2, V2);

  for(auto x = 0; x < chol_count; x++) {
      Tensor<T>* cholvec = chol_vecs.at(x);

        auto lambdacv = [&](Tensor<T> tensor, const IndexVector& blockid, span<T> buf){
            auto block_dims = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);
                
            TAMM_SIZE c = 0;
            for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
                    j++, c++) {
                buf[c] = CholVpr(i,j,x);
                }
            }
        };
    update_tensor_general((*cholvec)(), lambdacv);
   }



//   for(auto x = 0; x < chol_count; x++) {
//       Tensor<T>* cholvec = chol_vecs.at(x);

//       block_for(*ec, (*cholvec)(), [&](IndexVector it) {
//           Tensor<T> tensor     = (*cholvec)().tensor();
//           const TAMM_SIZE size = tensor.block_size(it);

//           std::vector<T> buf(size);

//           auto block_offset = tensor.block_offsets(it);
//           auto block_dims   = tensor.block_dims(it);

//           TAMM_SIZE c = 0;
//           for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
//               i++) {
//               for(auto j = block_offset[1]; j < block_offset[1] + block_dims[1];
//                   j++, c++) {
//                   buf[c] = CholVpr(i,j,x);;
//               }
//           }
//           (*cholvec).put(it, buf);
//       });
//   }

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  CHECK_NOTHROW(ccsd_driver<T>(*ec, MO, d_t1, d_t2, d_f1, chol_vecs,
                               maxiter, thresh, zshiftl, ndiis, hf_energy,
                               total_orbitals, 2 * ov_alpha));

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for Cholesky CCSD: " << ccsd_time << " secs\n";

  Tensor<T>::deallocate(d_t1, d_t2, d_f1);//, d_v2);
  for (auto x = 0; x < chol_count; x++) Tensor<T>::deallocate(*chol_vecs[x]);
  MemoryManagerGA::destroy_coll(mgr);
  delete ec;

}
