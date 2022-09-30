#include "ga/ga-mpi.h"
#include "ga/ga.h"
#include "ga/macdecls.h"
#include "mpi.h"

#include <tamm/op_executor.hpp>
#include <tamm/tamm.hpp>

using namespace tamm;
using namespace tamm::new_ops;

template<typename T>
void ccsd_e(Scheduler& sch, const TiledIndexSpace& LMO, const TiledIndexSpace& PNO, Tensor<T>& E,
            const Tensor<T>& F, const Tensor<T>& V, const Tensor<T>& t1, const Tensor<T>& t2) {
  auto [i, j, k, l, m, n] = LMO.labels<6>("all");
  auto [a, e, f, at]      = PNO.labels<4>("all");

  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, E, F, V, t1, t2);
  TAMM_REGISTER_SYMBOLS(symbol_table, i, j, k, l, m, n, a, e, f, at);

  auto op = 2.0 * (LTOp) F(m, e(m, m)) * (LTOp) t1(m, e(m, m)) +
            2.0 * (LTOp) V(m, n, e(m, n), f(m, n)) * (LTOp) t2(m, n, e(m, n), f(m, n)) +
            2.0 * (LTOp) V(m, n, e(m, m), f(n, n)) * (LTOp) t1(m, e(m, m)) * (LTOp) t1(n, f(n, n)) +
            -1.0 * (LTOp) V(m, n, f(m, n), e(m, n)) * (LTOp) t2(m, n, e(m, n), f(m, n)) +
            -1.0 * (LTOp) V(m, n, f(n, n), e(m, m)) * (LTOp) t1(m, e(m, m)) * (LTOp) t1(n, f(n, n));

  E().update(op);

  OpExecutor op_exec{sch, symbol_table};
  op_exec.pretty_print_binarized(E);
  op_exec.execute(E);

  std::cout << "E: " << get_scalar(E) << "\n";
}

template<typename T>
void ccsd_t1(Scheduler& sch, const TiledIndexSpace& LMO, const TiledIndexSpace& PNO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& S_1, const Tensor<T>& S_2,
             const Tensor<T>& S_3, const Tensor<T>& S_4, const Tensor<T>& F_ai,
             const Tensor<T>& F_ia, const Tensor<T>& F_ab, const Tensor<T>& F_ij,
             const Tensor<T>& V_mn_ef, const Tensor<T>& V_mn_ei, const Tensor<T>& V_ma_ei,
             const Tensor<T>& V_ma_ie, const Tensor<T>& V_ma_ef) {
  // Index labels
  auto [i, j, m, n]  = LMO.labels<4>("all");
  auto [a, e, f, at] = PNO.labels<4>("all");

  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, i0, t1, t2, S_1, S_2, S_3, S_4, F_ai, F_ia, F_ab, F_ij,
                        V_mn_ef, V_mn_ei, V_ma_ei, V_ma_ie, V_ma_ef);
  TAMM_REGISTER_SYMBOLS(symbol_table, i, j, m, n, a, e, f, at);

  auto op =
    // S1. (@ \textcolor{blue}{$ +f^{a_{ii}}_{i} $ }@)
    ((LTOp) F_ai(a(i, i), i)) +
    // S2. (@ \textcolor{blue}{$-2 f^{m}_{e_{ii}} t_{m}^{\tilde{a}_{mm}} t_{i}^{e_{ii}}
    // S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}} $  }@)
    (-2.0 * (LTOp) F_ia(m, e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, at(m, m)) *
     (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S3. (@ \textcolor{blue}{$+f^{a_{ii}}_{e_{ii}} t_{i}^{e_{ii}}$  }@)
    ((LTOp) F_ab(i, a(i, i), e(i, i)) * (LTOp) t1(i, e(i, i))) +
    // S4. (@ \textcolor{blue}{$-2v^{mn}_{e_{ii} f_{mn}} t_{mn}^{\tilde{a}_{mn} f_{mn}}
    // S^{(mn)(ii)}_{\tilde{a}_{mn} a_{ii}}$ }@)
    (-2.0 * (LTOp) V_mn_ef(m, n, e(i, i), f(m, n)) * (LTOp) t1(i, e(i, i)) *
     (LTOp) t2(m, n, at(m, n), f(m, n)) * (LTOp) S_2(m, n, i, at(m, n), a(i, i))) +
    // S5. (@ \textcolor{blue}{$-2v^{mn}_{e_{ii} f_{nn}} t_{m}^{\tilde{a}_{mm}} t_{n}^{f_{nn}}
    // t_{i}^{e_{ii}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$ }@)
    (-2.0 * (LTOp) V_mn_ef(m, n, e(i, i), f(n, n)) * (LTOp) t1(i, e(i, i)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, f(n, n)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S6. (@ \textcolor{blue}{$+v^{nm}_{e_{ii} f_{mn}}  t_{mn}^{\tilde{a}_{mn} f_{mn}}
    // t_{i}^{e_{ii}}  S^{(mn)(ii)}_{\tilde{a}_{mn} a_{ii}}$ }@)
    ((LTOp) V_mn_ef(n, m, e(i, i), f(m, n)) * (LTOp) t1(i, e(i, i)) *
     (LTOp) t2(m, n, at(m, n), f(m, n)) * (LTOp) S_2(m, n, i, at(m, n), a(i, i))) +
    // S7. (@ \textcolor{blue}{$+v^{nm}_{e_{ii} f_{nn}} t_{m}^{\tilde{a}_{mm}} t_{n}^{f_{nn}}
    // t_{i}^{e_{ii}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$  }@)
    ((LTOp) V_mn_ef(n, m, e(i, i), f(n, n)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, at(m, m)) *
     (LTOp) t1(n, f(n, n)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S8. (@ \textcolor{blue}{ $-f^{m}_{i} t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm}
    // a_{ii}}$ }@)
    (-1.0 * (LTOp) F_ij(m, i) * (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S9. (@ \textcolor{blue}{ $-2v^{mn}_{e_{in} f_{in}} t_{in}^{e_{in} f_{in}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$  }@)
    (-2.0 * (LTOp) V_mn_ef(m, n, e(i, n), f(i, n)) * (LTOp) t2(i, n, e(i, n), f(i, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S10. (@ \textcolor{blue}{ $-2v^{mn}_{e_{ii} f_{nn}} t_{i}^{e_{ii}} t_{n}^{f_{nn}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$ }@)
    (-2.0 * (LTOp) V_mn_ef(m, n, e(i, i), f(n, n)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(n, f(n, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S11. (@ \textcolor{blue}{  $+v^{mn}_{f_{in} e_{in}} t_{in}^{e_{in} f_{in}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$ }@)
    ((LTOp) V_mn_ef(m, n, f(i, n), e(i, n)) * (LTOp) t2(i, n, e(i, n), f(i, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S12. (@ \textcolor{blue}{  $+v^{mn}_{f_{nn} e_{ii}} t_{i}^{e_{ii}} t_{n}^{f_{nn}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}} $ }@)
    ((LTOp) V_mn_ef(m, n, f(n, n), e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(n, f(n, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S13. (@ \textcolor{blue}{  $+2f^{m}_{e_{mi}} t_{mi}^{e_{mi} \tilde{a}_{mi}}
    // S^{(mi)(ii)}_{\tilde{a}_{mi} a_{ii}}$ }@)
    (2.0 * (LTOp) F_ia(m, e(m, i)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
     (LTOp) S_3(m, i, at(m, i), a(i, i))) +
    // S14. (@ \textcolor{blue}{ $-f^{m}_{e_{im}} t_{im}^{e_{im}\tilde{a}_{im}}
    // S^{(im)(ii)}_{\tilde{a}_{im} a_{ii}}$  }@)
    (-1.0 * (LTOp) F_ia(m, e(i, m)) * (LTOp) t2(i, m, e(i, m), at(i, m)) *
     (LTOp) S_4(i, m, at(i, m), a(i, i))) +
    // S15. (@ \textcolor{blue}{  $+f^{m}_{e_{ii}} t_{i}^{e_{ii}} t_{m}^{\tilde{a}_{mm}}
    // S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$  }@)
    ((LTOp) F_ia(m, e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, at(m, m)) *
     (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S16. (@ \textcolor{blue}{ $+4v^{mn}_{e_{mi} f_{nn}} t_{n}^{f_{nn}}
    // t_{mi}^{e_{mi}\tilde{a}_{mi}} S^{(mi)(ii)}_{\tilde{a}_{mi} a_{ii}}$ }@)
    (4.0 * (LTOp) V_mn_ef(m, n, e(m, i), f(n, n)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
     (LTOp) t1(n, f(n, n)) * (LTOp) S_3(m, i, at(m, i), a(i, i))) +
    // S17. (@ \textcolor{blue}{  $-2v^{mn}_{e_{im} f_{nn}} t_{n}^{f_{nn}}
    // t_{im}^{e_{im}\tilde{a}_{im}} S^{(im)(ii)}_{\tilde{a}_{im} a_{ii}}$  }@)
    (-2.0 * (LTOp) V_mn_ef(m, n, e(i, m), f(n, n)) * (LTOp) t2(i, m, e(i, m), at(i, m)) *
     (LTOp) t1(n, f(n, n)) * (LTOp) S_4(i, m, at(i, m), a(i, i))) +
    // S18. (@ \textcolor{blue}{  $+2v^{mn}_{e_{ii} f_{nn}} t_{n}^{f_{nn}} t_{i}^{e_{ii}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm} a_{ii}}$ }@)
    (2.0 * (LTOp) V_mn_ef(m, n, e(i, i), f(n, n)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(n, f(n, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S19. (@ \textcolor{blue}{ $-2v^{mn}_{f_{nn} e_{mi}} t_{n}^{f_{nn}} t_{mi}^{e_{mi}
    // \tilde{a}_{mi}} S^{(mi)(ii)}_{\tilde{a}_{mi} a_{ii}}$}@)
    (-2.0 * (LTOp) V_mn_ef(m, n, f(n, n), e(m, i)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
     (LTOp) t1(n, f(n, n)) * (LTOp) S_3(m, i, at(m, i), a(i, i))) +
    // S20. (@ \textcolor{blue}{  $+v^{mn}_{f_{nn} e_{im}} t_{n}^{f_{nn}}
    // t_{im}^{e_{im}\tilde{a}_{im}} S^{(im)(ii)}_{\tilde{a}_{im} a_{ii}}$ }@)
    ((LTOp) V_mn_ef(m, n, f(n, n), e(i, m)) * (LTOp) t2(i, m, e(i, m), at(i, m)) *
     (LTOp) t1(n, f(n, n)) * (LTOp) S_4(i, m, at(i, m), a(i, i))) +
    // S21. (@ \textcolor{blue}{ $-v^{mn}_{f_{nn} e_{ii}} t_{n}^{f_{nn}}t_{i}^{e_{ii}}
    // t_{m}^{\tilde{a}_{mm}} S^{(mm)(ii)}_{\tilde{a}_{mm}a_{ii}}$}@)
    (-1.0 * (LTOp) V_mn_ef(m, n, f(n, n), e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(n, f(n, n)) *
     (LTOp) t1(m, at(m, m)) * (LTOp) S_1(m, i, at(m, m), a(i, i))) +
    // S22. (@ \textcolor{blue}{ $+2v^{m a_{ii}}_{e_{mm} i} t_{m}^{e_{mm}} $ }@)
    (2.0 * (LTOp) V_ma_ei(m, a(i, i), e(m, m), i) * (LTOp) t1(m, e(m, m))) +
    // S23. (@ \textcolor{blue}{ $-v^{m a_{ii}}_{i e_{mm}} t_{m}^{e_{mm}} $ }@)
    (-1.0 * (LTOp) V_ma_ie(m, a(i, i), i, e(m, m)) * (LTOp) t1(m, e(m, m))) +
    // S24. (@ \textcolor{blue}{ $+2v^{m a_{ii}}_{e_{mi} f_{mi}} t_{mi}^{e_{mi} f_{mi}}$ }@)
    (2.0 * (LTOp) V_ma_ef(m, a(i, i), e(m, i), f(m, i)) * (LTOp) t2(m, i, e(m, i), f(m, i))) +
    // S25. (@ \textcolor{blue}{ $+2v^{m a_{ii}}_{e_{mm} f_{ii}} t_{m}^{e_{mm}} t_{i}^{f_{ii}} $ }@)
    (2.0 * (LTOp) V_ma_ef(m, a(i, i), e(m, m), f(i, i)) * (LTOp) t1(i, f(i, i))) *
      (LTOp) t1(m, e(m, m)) +
    // S26. (@ \textcolor{blue}{ $-v^{m a_{ii}}_{f_{mi} e_{mi}} t_{mi}^{e_{mi} f_{mi}} $ }@)
    (-1.0 * (LTOp) V_ma_ef(m, a(i, i), f(m, i), e(m, i)) * (LTOp) t2(m, i, e(m, i), f(m, i))) +
    // S27. (@ \textcolor{blue}{ $-v^{m a_{ii}}_{f_{ii} e_{mm}} t_{m}^{e_{mm}} t_{i}^{f_{ii}} $ }@)
    (-1.0 * (LTOp) V_ma_ef(m, a(i, i), f(i, i), e(m, m)) * (LTOp) t1(i, f(i, i)) *
     (LTOp) t1(m, e(m, m))) +
    // S28. (@ \textcolor{blue}{ $-2v^{mn}_{e_{mn} i} t_{mn}^{e_{mn} \tilde{a}_{mn}}
    // S^{(mn)(ii)}_{\tilde{a}_{mn} a_{ii}}$  }@)
    (-2.0 * (LTOp) V_mn_ei(m, n, e(m, n), i) * (LTOp) t2(m, n, e(m, n), at(m, n)) *
     (LTOp) S_2(m, n, i, at(m, n), a(i, i))) +
    // S29. (@ \textcolor{blue}{ $-2v^{mn}_{e_{mm} i} t_{m}^{e_{mm}} t_{n}^{\tilde{a}_{nn}}
    // S^{(nn)(ii)}_{\tilde{a}_{nn} a_{ii}}$ }@)
    (-2.0 * (LTOp) V_mn_ei(m, n, e(m, m), i) * (LTOp) t1(m, e(m, m)) * (LTOp) t1(n, at(n, n)) *
     (LTOp) S_1(n, i, at(n, n), a(i, i))) +
    // S30. (@ \textcolor{blue}{ $+v^{nm}_{e_{mn} i} t_{mn}^{e_{mn} \tilde{a}_{mn}}
    // S^{(mn)(ii)}_{\tilde{a}_{mn} a_{ii}}$ }@)
    ((LTOp) V_mn_ei(n, m, e(m, n), i) * (LTOp) t2(m, n, e(m, n), at(m, n)) *
     (LTOp) S_2(m, n, i, at(m, n), a(i, i))) +
    // S31. (@ \textcolor{blue}{ $+v^{nm}_{e_{mm} i} t_{m}^{e_{mm}} t_{n}^{\tilde{a}_{nn}}
    // S^{(nn)(ii)}_{\tilde{a}_{nn} a_{ii}}$ }@)
    ((LTOp) V_mn_ei(n, m, e(m, m), i) * (LTOp) t1(m, e(m, m)) * (LTOp) t1(n, at(n, n)) *
     (LTOp) S_1(n, i, at(n, n), a(i, i)));

  i0(a(i, i), i).update(op);
  OpExecutor op_exec{sch, symbol_table};
  op_exec.pretty_print_binarized(i0);
  op_exec.execute(i0);

  std::cout << "i0(a(i, i), i):"
            << "\n";
  print_tensor(i0);
}

template<typename T>
void ccsd_t2(Scheduler& sch, const TiledIndexSpace& LMO, const TiledIndexSpace& PNO, Tensor<T>& i0,
             const Tensor<T>& t1, const Tensor<T>& t2, const Tensor<T>& S, const Tensor<T>& F_ia,
             const Tensor<T>& F_ab, const Tensor<T>& F_ij, const Tensor<T>& V_mn_ef,
             const Tensor<T>& V_mn_ei, const Tensor<T>& V_ma_ie, const Tensor<T>& V_ma_ef,
             const Tensor<T>& V_ab_ij, const Tensor<T>& V_mn_ij, const Tensor<T>& V_mn_ie,
             const Tensor<T>& V_ab_cd, const Tensor<T>& V_am_ef, const Tensor<T>& V_ab_ie,
             const Tensor<T>& V_am_ij, const Tensor<T>& V_am_ie) {
  // Index labels
  auto [i, j, m, n]         = LMO.labels<4>("all");
  auto [a, b, e, f, at, bt] = PNO.labels<6>("all");

  SymbolTable symbol_table;

  TAMM_REGISTER_SYMBOLS(symbol_table, i0, t1, t2, S, F_ia, F_ab, F_ij, V_mn_ef, V_mn_ei, V_ma_ie,
                        V_ma_ef, V_ab_ij, V_mn_ij, V_mn_ie, V_ab_cd, V_am_ef, V_ab_ie, V_am_ij,
                        V_am_ie);
  TAMM_REGISTER_SYMBOLS(symbol_table, i, j, m, n, a, b, e, f, at, bt);

  auto op =
    // D1. (@ \textcolor{blue}{$+v^{a_{ij} b_{ij}}_{ij}$} @)
    ((LTOp) V_ab_ij(a(i, j), b(i, j), i, j))

    // D2. (@ \textcolor{blue}{$+v^{mn}_{ij} t_{mn}^{\tilde{a}_{mn}\tilde{b}_{mn}}
    // S^{(mn)(ij)}_{\tilde{b}_{mn} b_{ij}} $} @)
    + ((LTOp) V_mn_ij(m, n, i, j) * (LTOp) t2(m, n, at(m, n), bt(m, n)) *
       (LTOp) S(m, n, i, j, at(m, n), a(i, j)) * (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D3. (@ \textcolor{blue}{$+v^{mn}_{ie_{jj}} t_{j}^{e_{jj}}t_{mn}^{\tilde{a}_{mn}
    // \tilde{b}_{mn}} S^{(mn)(ij)}_{\tilde{a}_{mn}a_{ij}} S^{(mn)(ij)}_{\tilde{b}_{mn} b_{ij}} $}
    // @)
    + ((LTOp) V_mn_ie(m, n, i, e(j, j)) * (LTOp) t1(j, e(j, j)) *
       (LTOp) t2(m, n, at(m, n), bt(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)) *
       (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D4. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, i} j} t_{i}^{e_{i, i}} t_{m,n}^{\tilde{a}_{m, n}
    // \tilde{b}_{m, n}} S^{(m, n)(i, j)}_{\tilde{a}_{m,n} a_{i, j}} S^{(m, n)(i, j)}_{\tilde{b}_{m,
    // n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ei(m, n, e(i, i), j) * (LTOp) t1(i, e(i, i)) *
       (LTOp) t2(m, n, at(m, n), bt(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)) *
       (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D5. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, j} f_{i, j}} t_{i, j}^{e_{i, j} f_{i, j}} t_{m,
    // n}^{\tilde{a}_{m, n} \tilde{b}_{m, n}} S^{(m, n)(i, j)}_{\tilde{a}_{m, n} a_{i, j}} S^{(m,
    // n)(i, j)}_{\tilde{b}_{m, n} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(m, n, e(i, j), f(i, j)) * (LTOp) t2(i, j, e(i, j), f(i, j)) *
       (LTOp) t2(m, n, at(m, n), bt(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)) *
       (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D6. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, i} f_{j, j}} t_{i}^{e_{i, i}} t_{j}^{f_{j, j}}
    // t_{m, n}^{\tilde{a}_{m, n} \tilde{b}_{m, n}} S^{(m, n)(i, j)}_{\tilde{a}_{m, n} a_{i, j}}
    // S^{(m, n)(i, j)}_{\tilde{b}_{m, n} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(m, n, e(i, i), f(j, j)) * ((LTOp) t1(i, e(i, i)) * (LTOp) t1(j, f(j, j))) *
       (LTOp) t2(m, n, at(m, n), bt(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)) *
       (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D7. (@ \textcolor{blue}{$+v^{m, n}_{i, j} t_{m}^{\tilde{a}_{m, m}} t_{n}^{\tilde{b}_{n, n}}
    // S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} a_{i, j}}$}
    // @)
    + ((LTOp) V_mn_ij(m, n, i, j) * (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), a(i, j)))

    // D8. (@ \textcolor{blue}{$+v^{m, n}_{ie_{j, j}} t_{j}^{e_{j, j}} t_{m}^{\tilde{a}_{m, m}}
    // t_{n}^{\tilde{b}_{n, n}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} a_{i, j}}$} @)
    + ((LTOp) V_mn_ie(m, n, i, e(j, j)) * (LTOp) t1(j, e(j, j)) * (LTOp) t1(m, at(m, m)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), a(i, j)))

    // // D9. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, i} j} t_{i}^{e_{i, i}} t_{m}^{\tilde{a}_{m, m}}
    // t_{n}^{\tilde{b}_{n, n}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ei(m, n, e(i, i), j) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, at(m, m)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D10. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, j} f_{i, j}} t_{i, j}^{e_{i, j} b_{i, j}}
    // t_{m}^{\tilde{a}_{m, m}} t_{n}^{\tilde{b}_{n, n}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i,
    // j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, e(i, j), f(i, j)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D11. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, i} f_{j, j}} t_{i}^{e_{i, i}} t_{j}^{f_{j, j}}
    // t_{m}^{\tilde{a}_{m, m}} t_{n}^{\tilde{b}_{n, n}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i,
    // j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, e(i, i), f(j, j)) * ((LTOp) t1(i, e(i, i)) * (LTOp) t1(j, f(j, j))) *
       (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D12. (@ \textcolor{blue}{$+v^{a_{i, j} b_{i, j}}_{e_{i, j} f_{i, j}} t_{i, j}^{e_{i, j} f_{i,
    // j}}$} @)
    + ((LTOp) V_ab_cd(i, j, a(i, j), b(i, j), e(i, j), f(i, j)) * (LTOp) t2(i, j, e(i, j), f(i, j)))

    // D13. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{e_{i, j} f_{i, j}} t_{m}^{\tilde{b}_{m, m}} t_{i,
    // j}^{e_{i, j} f_{i, j}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}}$} @)
    - ((LTOp) V_am_ef(a(i, j), m, e(i, j), f(i, j)) * (LTOp) t2(i, j, e(i, j), f(i, j)) *
       (LTOp) t1(m, bt(m, m)) * (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D14. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{e_{i, j} f_{i, j}} t_{m}^{\tilde{a}_{m, m}} t_{i,
    // j}^{e_{i, j} f_{i, j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} $} @)
    - ((LTOp) V_ma_ef(m, b(i, j), e(i, j), f(i, j)) * (LTOp) t2(i, j, e(i, j), f(i, j)) *
       (LTOp) t1(m, at(m, m)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D15. (@ \textcolor{blue}{$+v^{a_{i, j} b_{i, j}}_{e_{i, i} f_{j, j}} t_{i}^{e_{i, i}}
    // t_{j}^{f_{j, j}} $} @)
    + ((LTOp) V_ab_cd(i, j, a(i, j), b(i, j), e(i, i), f(j, j)) * (LTOp) t1(i, e(i, i)) *
       (LTOp) t1(j, f(j, j)))
    // D16. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{e_{i, i} f_{j, j}} t_{m}^{\tilde{b}_{m, m}}
    // t_{i}^{e_{i, i}} t_{j}^{f_{j, j}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}}$} @)
    + ((LTOp) V_am_ef(a(i, j), m, e(i, i), f(j, j)) *
       ((LTOp) t1(i, e(i, i)) * (LTOp) t1(j, f(j, j))) * (LTOp) t1(m, bt(m, m)) *
       (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D17. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{e_{i, i} f_{j, j}} t_{m}^{\tilde{a}_{m, m}}
    // t_{i}^{e_{i, i}} t_{j}^{f_{j, j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} $} @)
    + ((LTOp) V_ma_ef(m, b(i, j), e(i, i), f(j, j)) *
       ((LTOp) t1(i, e(i, i)) * (LTOp) t1(j, f(j, j))) * (LTOp) t1(m, at(m, m)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)))
    ///@todo what should F_ab's first index should be?
    // D18. (@ \textcolor{blue}{$+f^{a_{i, j}}_{e_{i, j}} t_{i, j}^{e_{i, j} b_{i, j}} $} @)
    + ((LTOp) F_ab(i, a(i, j), e(i, j)) * (LTOp) t2(i, j, e(i, j), b(i, j)))

    // D19. (@ \textcolor{blue}{$+f^{b_{i, j}}_{e_{i, j}} t_{ji}^{e_{i, j} a_{i, j}} $} @)
    + ((LTOp) F_ab(i, b(i, j), e(i, j)) * (LTOp) t2(j, i, e(i, j), a(i, j)))

    // D20. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, j} f_{m, n}} t_{m, n}^{\tilde{a}_{m, n} f_{m,
    // n}} t_{i, j}^{e_{i, j} b_{i, j}} S^{(m, n)(i, j)}_{\tilde{a}_{m, n} a_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, j), f(m, n)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t2(m, n, at(m, n), f(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)))

    // D21. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, j} f_{m, n}} t_{m, n}^{\tilde{b}_{m, n} f_{m,
    // n}} t_{ji}^{e_{i, j} a_{i, j}} S^{(m, n)(i, j)}_{\tilde{b}_{m, n} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, j), f(m, n)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t2(m, n, bt(m, n), f(m, n)) * (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    // D22. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, j} f_{n, n}} t_{m}^{\tilde{a}_{m, m}}
    // t_{n}^{f_{n, n}} t_{i, j}^{e_{i, j} b_{i, j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}}$}
    // @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, j), f(n, n)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, f(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D23. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, j} f_{n, n}} t_{m}^{\tilde{b}_{m, m}}
    // t_{n}^{f_{n, n}} t_{ji}^{e_{i, j} a_{i, j}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}} $}
    // @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, j), f(n, n)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t1(m, bt(m, m)) * (LTOp) t1(n, f(n, n)) * (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D24. (@ \textcolor{blue}{$+v^{nm}_{e_{i, j} f_{m, n}} t_{m, n}^{\tilde{a}_{m, n} f_{m, n}}
    // t_{i, j}^{e_{i, j} b_{i, j}} S^{(m, n)(i, j)}_{\tilde{a}_{m, n} a_{i, j}}$} @)
    + ((LTOp) V_mn_ef(n, m, e(i, j), f(m, n)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t2(m, n, at(m, n), f(m, n)) * (LTOp) S(m, n, i, j, at(m, n), a(i, j)))

    // D25. (@ \textcolor{blue}{$+v^{nm}_{e_{i, j} f_{m, n}} t_{m, n}^{\tilde{b}_{m, n} f_{m, n}}
    // t_{ji}^{e_{i, j} a_{i, j}} S^{(m, n)(i, j)}_{\tilde{b}_{m, n} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(n, m, e(i, j), f(m, n)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t2(m, n, bt(m, n), f(m, n)) * (LTOp) S(m, n, i, j, bt(m, n), b(i, j)))

    //////////////////////////////////////////////////

    // D26. (@ \textcolor{blue}{$+v^{nm}_{e_{i, j} f_{n, n}} t_{m}^{\tilde{a}_{m, m}} t_{n}^{f_{n,
    // n}} t_{i, j}^{e_{i, j} b_{i, j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} $} @)
    + ((LTOp) V_mn_ef(n, m, e(i, j), f(n, n)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t1(m, at(m, m)) * (LTOp) t1(n, f(n, n)) * (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D27. (@ \textcolor{blue}{$+v^{nm}_{e_{i, j} f_{n, n}} t_{m}^{\tilde{b}_{m, m}} t_{n}^{f_{n,
    // n}} t_{ji}^{e_{i, j} a_{i, j}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(n, m, e(i, j), f(n, n)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t1(m, bt(m, m)) * (LTOp) t1(n, f(n, n)) * (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D28. (@ \textcolor{blue}{$-f^{m}_{e_{i, j}} t_{m}^{\tilde{a}_{m, m}} t_{i, j}^{e_{i, j} b_{i,
    // j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}}$} @)
    - ((LTOp) F_ia(m, e(i, j)) * (LTOp) t2(i, j, e(i, j), b(i, j)) * (LTOp) t1(m, at(m, m)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D29. (@ \textcolor{blue}{$-f^{m}_{e_{i, j}} t_{m}^{\tilde{b}_{m, m}} t_{ji}^{e_{i, j} a_{i,
    // j}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}} $} @)
    - ((LTOp) F_ia(m, e(i, j)) * (LTOp) t2(j, i, e(i, j), a(i, j)) * (LTOp) t1(m, bt(m, m)) *
       (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D30. (@ \textcolor{blue}{$+2 v^{a_{i, j} m}_{e_{i, j} f_{m, m}} t_{m}^{f_{m, m}} t_{i,
    // j}^{e_{i, j} b_{i, j}} $} @)
    + (2.0 * (LTOp) V_am_ef(a(i, j), m, e(i, j), f(m, m)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t1(m, f(m, m)))

    // D31. (@ \textcolor{blue}{$ +2 v^{b_{i, j} m}_{e_{i, j} f_{m, m}} t_{m}^{f_{m, m}}
    // t_{ji}^{e_{i, j} a_{i, j}}  $} @)
    + (2.0 * (LTOp) V_am_ef(b(i, j), m, e(i, j), f(m, m)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t1(m, f(m, m)))

    // D32. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{f_{m, m} e_{i, j}} t_{m}^{f_{m, m}} t_{i,
    // j}^{e_{i, j} b_{i, j}} $} @)
    - ((LTOp) V_am_ef(a(i, j), m, f(m, m), e(i, j)) * (LTOp) t2(i, j, e(i, j), b(i, j)) *
       (LTOp) t1(m, f(m, m)))

    // D33. (@ \textcolor{blue}{$-v^{b_{i, j} m}_{f_{m, m} e_{i, j}} t_{m}^{f_{m, m}} t_{ji}^{e_{i,
    // j} a_{i, j}} $} @)
    - ((LTOp) V_am_ef(b(i, j), m, f(m, m), e(i, j)) * (LTOp) t2(j, i, e(i, j), a(i, j)) *
       (LTOp) t1(m, f(m, m)))

    // D34. (@ \textcolor{blue}{$- f^{m}_{i} t_{m, j}^{\tilde{a}_{m, j} \tilde{b}_{m, j}} S^{(m,
    // j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) F_ij(m, i) * (LTOp) t2(m, j, at(m, j), bt(m, j)) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D35. (@ \textcolor{blue}{$- f^{m}_{j} t_{m, i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m,
    // i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}} $} @)
    - ((LTOp) F_ij(m, j) * (LTOp) t2(m, i, bt(m, i), at(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D36. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, n} f_{i, n}} t_{i, n}^{e_{i, n} f_{i, n}} t_{m,
    // j}^{\tilde{a}_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, n), f(i, n)) * (LTOp) t2(i, n, e(i, n), f(i, n)) *
       (LTOp) t2(m, j, at(m, j), bt(m, j)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D37. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{j, n} f_{j, n}} t_{j, n}^{e_{j, n} f_{j, n}} t_{m,
    // i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m,
    // i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(j, n), f(j, n)) * (LTOp) t2(j, n, e(j, n), f(j, n)) *
       (LTOp) t2(m, i, bt(m, i), at(m, i)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D38. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{i, i} f_{n, n}} t_{i}^{e_{i, i}} t_{n}^{f_{n, n}}
    // t_{m, j}^{\tilde{a}_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}}
    // S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(i, i), f(n, n)) * (LTOp) t1(i, e(i, i)) *
       (LTOp) t1(n, f(n, n)) * (LTOp) t2(m, j, at(m, j), bt(m, j)) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D39. (@ \textcolor{blue}{$-2 v^{m, n}_{e_{j, j} f_{n, n}} t_{j}^{e_{j, j}} t_{n}^{f_{n, n}}
    // t_{m, i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}
    // S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(j, j), f(n, n)) * (LTOp) t1(j, e(j, j)) *
       (LTOp) t1(n, f(n, n)) * (LTOp) t2(m, i, bt(m, i), at(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D40. (@ \textcolor{blue}{$+v^{m, n}_{f_{i, n} e_{i, n}} t_{i, n}^{e_{i, n} f_{i, n}} t_{m,
    // j}^{\tilde{a}_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(m, n, f(i, n), e(i, n)) * (LTOp) t2(i, n, e(i, n), f(i, n)) *
       (LTOp) t2(m, j, at(m, j), bt(m, j)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D41. (@ \textcolor{blue}{$+v^{m, n}_{f_{j, n} e_{j, n}} t_{j, n}^{e_{j, n} f_{j, n}} t_{m,
    // i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m,
    // i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(m, n, f(j, n), e(j, n)) * (LTOp) t2(j, n, e(j, n), f(j, n)) *
       (LTOp) t2(m, i, bt(m, i), at(m, i)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D42. (@ \textcolor{blue}{$+v^{m, n}_{f_{n, n} e_{i, i}}  t_{i}^{e_{i, i}} t_{n}^{f_{n, n}}
    // t_{m, j}^{\tilde{a}_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}}
    // S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, f(n, n), e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(n, f(n, n)) *
       (LTOp) t2(m, j, at(m, j), bt(m, j)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D43. (@ \textcolor{blue}{$+v^{m, n}_{f_{n, n} e_{j, j}}  t_{j}^{e_{j, j}} t_{n}^{f_{n, n}}
    // t_{m, i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}
    // S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, f(n, n), e(j, j)) * (LTOp) t1(j, e(j, j)) * (LTOp) t1(n, f(n, n)) *
       (LTOp) t2(m, i, bt(m, i), at(m, i)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D44. (@ \textcolor{blue}{$-f^{m}_{e_{i, i}} t_{i}^{e_{i, i}} t_{m, j}^{\tilde{a}_{m, j}
    // \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) F_ia(m, e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t2(m, j, at(m, j), bt(m, j)) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D45. (@ \textcolor{blue}{$-f^{m}_{e_{j, j}} t_{j}^{e_{j, j}} t_{m, i}^{\tilde{b}_{m, i}
    // \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m, i)(i,
    // j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    - ((LTOp) F_ia(m, e(j, j)) * (LTOp) t1(j, e(j, j)) * (LTOp) t2(m, i, bt(m, i), at(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D46. (@ \textcolor{blue}{$-2v^{m, n}_{i e_{n, n}} t_{n}^{e_{n, n}} t_{m, j}^{\tilde{a}_{m, j}
    // \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ie(m, n, i, e(n, n)) * (LTOp) t1(n, e(n, n)) *
       (LTOp) t2(m, j, at(m, j), bt(m, j)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D47. (@ \textcolor{blue}{$-2v^{m, n}_{j e_{n, n}} t_{n}^{e_{n, n}} t_{m, i}^{\tilde{b}_{m, i}
    // \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m, i)(i,
    // j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ie(m, n, j, e(n, n)) * (LTOp) t1(n, e(n, n)) *
       (LTOp) t2(m, i, bt(m, i), at(m, i)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D48. (@ \textcolor{blue}{$+v^{nm}_{i e_{n, n}} t_{n}^{e_{n, n}} t_{m, j}^{\tilde{a}_{m, j}
    // \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(n, m, i, e(n, n)) * (LTOp) t1(n, e(n, n)) *
       (LTOp) t2(m, j, at(m, j), bt(m, j)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D49. (@ \textcolor{blue}{$+v^{nm}_{j e_{n, n}} t_{n}^{e_{n, n}} t_{m, i}^{\tilde{b}_{m, i}
    // \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m, i)(i,
    // j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(n, m, j, e(n, n)) * (LTOp) t1(n, e(n, n)) *
       (LTOp) t2(m, i, bt(m, i), at(m, i)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D50. (@ \textcolor{blue}{$+v^{a_{i, j} b_{i, j}}_{i e_{j, j}} t_{j}^{e_{j, j}} $} @)
    + ((LTOp) V_ab_ie(a(i, j), b(i, j), i, e(j, j)) * (LTOp) t1(j, e(j, j)))
    ////////////////////////////////////////////////////

    // D51. (@ \textcolor{blue}{$+v^{b_{i, j} a_{i, j}}_{j e_{i, i}} t_{i}^{e_{i, i}} $} @)
    + ((LTOp) V_ab_ie(b(i, j), a(i, j), j, e(i, i)) * (LTOp) t1(i, e(i, i)))

    // D52. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{i e_{j, j}} t_{m}^{\tilde{a}_{m, m}} t_{j}^{e_{j,
    // j}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, b(i, j), i, e(j, j)) * (LTOp) t1(j, e(j, j)) * (LTOp) t1(m, at(m, m)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D53. (@ \textcolor{blue}{$-v^{m a_{i, j}}_{j e_{i, i}} t_{m}^{\tilde{b}_{m, m}} t_{i}^{e_{i,
    // i}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, a(i, j), j, e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, bt(m, m)) *
       (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D54. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{i, j} t_{m}^{\tilde{b}_{m, m}} $} @)
    - ((LTOp) V_am_ij(a(i, j), m, i, j) * (LTOp) t1(bt(m, m), m) *
       (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D55. (@ \textcolor{blue}{$-v^{b_{i, j} m}_{ji} t_{m}^{\tilde{a}_{m, m}} S^{(m, m)(i,
    // j)}_{\tilde{a}_{m, m} a_{i, j}}$} @)
    - ((LTOp) V_am_ij(b(i, j), m, j, i) * (LTOp) t1(m, at(m, m)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D56. (@ \textcolor{blue}{$-v^{b_{i, j} m}_{i e_{j, j}} t_{j}^{e_{j, j}} t_{m}^{\tilde{b}_{m,
    // m}} S^{(m, m)(i, j)}_{\tilde{b}_{m, m} b_{i, j}}$} @)
    - ((LTOp) V_am_ie(b(i, j), m, i, e(j, j)) * (LTOp) t1(j, e(j, j)) * (LTOp) t1(m, bt(m, m)) *
       (LTOp) S(m, m, i, j, bt(m, m), b(i, j)))

    // D57. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{j e_{i, i}} t_{i}^{e_{i, i}} t_{m}^{\tilde{a}_{m,
    // m}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}}$} @)
    - ((LTOp) V_am_ie(a(i, j), m, j, e(i, i)) * (LTOp) t1(i, e(i, i)) * (LTOp) t1(m, at(m, m)) *
       (LTOp) S(m, m, i, j, at(m, m), a(i, j)))

    // D58. (@ \textcolor{blue}{$+2v^{a_{i, j} m}_{i e_{m, j}} t_{m, j}^{e_{m, j} \tilde{b}_{m, j}}
    // S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (2.0 * (LTOp) V_am_ie(a(i, j), m, i, e(m, j)) * (LTOp) t2(m, j, e(m, j), bt(m, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D59. (@ \textcolor{blue}{$+2v^{b_{i, j} m}_{j e_{m, i}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}}
    // S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    + (2.0 * (LTOp) V_am_ie(b(i, j), m, j, e(m, i)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D60. (@ \textcolor{blue}{$-2v^{nm}_{i e_{m, j}} t_{n}^{a_{n, n}} t_{m, j}^{e_{m, j}
    // \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ie(n, m, i, e(m, j)) * (LTOp) t2(e(m, j), bt(m, j), m, j) *
       (LTOp) t1(n, a(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D61. (@ \textcolor{blue}{$-2v^{nm}_{j e_{m, i}} t_{n}^{b_{n, n}} t_{m, i}^{e_{m, i}
    // \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ie(n, m, j, e(m, i)) * (LTOp) t2(e(m, i), at(m, i), m, i) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D62. (@ \textcolor{blue}{$+2v^{a_{i, j} m}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}} t_{m,
    // j}^{e_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + (2.0 * (LTOp) V_am_ef(a(i, j), m, f(i, i), e(m, j)) *
       ((LTOp) t2(m, j, e(m, j), bt(m, j)) * (LTOp) t1(i, f(i, i))) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D63. (@ \textcolor{blue}{$+2v^{b_{i, j} m}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{m,
    // i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} $} @)
    + (2.0 * (LTOp) V_am_ef(b(i, j), m, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D64. (@ \textcolor{blue}{$-v^{m, n}_{e_{m, j} f_{i, n}} t_{i, n}^{f_{i, n} \tilde{a}_{i, n}}
    // t_{m, j}^{e_{m, j} \tilde{b}_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) V_mn_ef(m, n, e(m, j), f(i, n)) *
       ((LTOp) t2(i, n, f(i, n), at(i, n)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D65. (@ \textcolor{blue}{$-v^{m, n}_{e_{m, i} f_{j, n}} t_{j, n}^{f_{j, n} \tilde{b}_{j, n}}
    // t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j,
    // n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    - ((LTOp) V_mn_ef(m, n, e(m, i), f(j, n)) *
       ((LTOp) t2(j, n, f(j, n), bt(j, n)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D66. (@ \textcolor{blue}{$-2v^{m, n}_{e_{m, j} f_{i, i}} t_{i}^{f_{i, i}}
    // t_{n}^{\tilde{a}_{n, n}} t_{m, j}^{e_{m, j} \tilde{b}_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n,
    // n} a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(m, j), f(i, i)) *
       ((LTOp) t2(m, j, e(m, j), bt(m, j)) * (LTOp) t1(i, f(i, i))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D67. (@ \textcolor{blue}{$-2v^{m, n}_{e_{m, i} f_{j, j}} t_{j}^{f_{j, j}}
    // t_{n}^{\tilde{b}_{n, n}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m,
    // i} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + (-2.0 * (LTOp) V_mn_ef(m, n, e(m, i), f(j, j)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, e(m, i), at(m, i))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D68. (@ \textcolor{blue}{$+2v^{m, n}_{e_{m, j} f_{i, n}} t_{i, n}^{\tilde{a}_{i, n} f_{i, n}}
    // t_{m, j}^{e_{m, j}\tilde{b}_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (2.0 * (LTOp) V_mn_ef(m, n, e(m, j), f(i, n)) *
       ((LTOp) t2(i, n, at(i, n), f(i, n)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D69. (@ \textcolor{blue}{$+2v^{m, n}_{e_{m, i} f_{j, n}} t_{j, n}^{\tilde{b}_{j, n} f_{j, n}}
    // t_{m, i}^{e_{m, i}\tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j, n)(i,
    // j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    + (2.0 * (LTOp) V_mn_ef(m, n, e(m, i), f(j, n)) *
       ((LTOp) t2(j, n, bt(j, n), f(j, n)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D70. (@ \textcolor{blue}{$-v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{\tilde{a}_{i, n} f_{i, n}}
    // t_{m, j}^{e_{m, j} \tilde{b}_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + (2.0 * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, at(i, n), f(i, n)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D71. (@ \textcolor{blue}{$-v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{\tilde{b}_{j, n} f_{j, n}}
    // t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j,
    // n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}} $} @)
    + (2.0 * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, bt(j, n), f(j, n)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))
    /////////////////////////////////////////////////////

    // D72. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{i e_{m, j}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}}
    // S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) V_am_ie(a(i, j), m, i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D73. (@ \textcolor{blue}{$-v^{b_{i, j} m}_{j e_{m, i}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}}
    // S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    - ((LTOp) V_am_ie(b(i, j), m, j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D74. (@ \textcolor{blue}{$+v^{nm}_{i e_{m, j}} t_{n}^{\tilde{a}_{n, n}} t_{m,
    // j}^{\tilde{b}_{m, j} e_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(n, m, i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) t1(n, at(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D75. (@ \textcolor{blue}{$+v^{nm}_{j e_{m, i}} t_{n}^{\tilde{b}_{n, n}} t_{m,
    // i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(n, m, j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D76. (@ \textcolor{blue}{$-v^{a_{i, j} m}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}} t_{m,
    // j}^{\tilde{b}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    - ((LTOp) V_am_ef(a(i, j), m, f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D77. (@ \textcolor{blue}{$-v^{b_{i, j} m}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{m,
    // i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} $} @)
    - ((LTOp) V_am_ef(b(i, j), m, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D78. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{e_{m, j} f_{i, n}} t_{i, n}^{f_{i,
    // n}\tilde{a}_{i, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n}
    // a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, e(m, j), f(i, n)) *
       ((LTOp) t2(i, n, f(i, n), at(i, n)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D79. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{e_{m, i} f_{j, n}} t_{j, n}^{f_{j,
    // n}\tilde{b}_{j, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i}
    // a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, e(m, i), f(j, n)) *
       ((LTOp) t2(j, n, f(j, n), bt(j, n)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D80. (@ \textcolor{blue}{$+v^{m, n}_{e_{m, j} f_{i, i}} t_{i}^{f_{i, i}} t_{n}^{\tilde{a}_{n,
    // n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, e(m, j), f(i, i)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D81. (@ \textcolor{blue}{$+v^{m, n}_{e_{m, i} f_{j, j}} t_{j}^{f_{j, j}} t_{n}^{\tilde{b}_{n,
    // n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n,
    // n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, e(m, i), f(j, j)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D82. (@ \textcolor{blue}{$-v^{m, n}_{e_{m, j} f_{i, n}} t_{i, n}^{\tilde{a}_{i, n} f_{i, n}}
    // t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) V_mn_ef(m, n, e(m, j), f(i, n)) *
       ((LTOp) t2(i, n, at(i, n), f(i, n)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D83. (@ \textcolor{blue}{$-v^{m, n}_{e_{m, i} f_{j, n}} t_{j, n}^{\tilde{b}_{j, n} f_{j, n}}
    // t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j,
    // n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    - ((LTOp) V_mn_ef(m, n, e(m, i), f(j, n)) *
       ((LTOp) t2(j, n, bt(j, n), f(j, n)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D84. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{\tilde{a}_{i,
    // n} f_{i, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n} a_{i,
    // j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, at(i, n), f(i, n)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D85. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{\tilde{b}_{j,
    // n} f_{j, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i,
    // j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, bt(j, n), f(j, n)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))
    /////////////////////////////////////////////////////////

    // D86. (@ \textcolor{blue}{$-v^{m a_{i, j}}_{i e_{m, j}} t_{m, j}^{e_{m, j} \tilde{b}_{m, j}}
    // S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, a(i, j), i, e(m, j)) * (LTOp) t2(m, j, e(m, j), bt(m, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D87. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{j e_{m, i}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}}
    // S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, b(i, j), j, e(m, i)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D88. (@ \textcolor{blue}{$+v^{m, n}_{i e_{m, j}} t_{n}^{\tilde{a}_{n, n}} t_{m, j}^{e_{m, j}
    // \tilde{b}_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(m, n, i, e(m, j)) * (LTOp) t2(m, j, e(m, j), bt(m, j)) *
       (LTOp) t1(n, at(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D89. (@ \textcolor{blue}{$+v^{m, n}_{j e_{m, i}} t_{n}^{\tilde{b}_{n, n}} t_{m, i}^{e_{m, i}
    // \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(m, n, j, e(m, i)) * (LTOp) t2(m, i, e(m, i), at(m, i)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D90. (@ \textcolor{blue}{$-v^{m a_{i, j}}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}} t_{m,
    // j}^{e_{m, j} \tilde{b}_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    - ((LTOp) V_ma_ef(m, a(i, j), f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D91. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{m,
    // i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    - ((LTOp) V_ma_ef(m, b(i, j), f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D92. (@ \textcolor{blue}{$\frac{1}{2} v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{f_{i, n}
    // \tilde{a}_{i, n}} t_{m, j}^{e_{m, j} \tilde{b}_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n}
    // a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, f(i, n), at(i, n)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D93. (@ \textcolor{blue}{$\frac{1}{2} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{f_{j, n}
    // \tilde{b}_{j, n}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i}
    // a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, f(j, n), bt(j, n)) * (LTOp) t2(m, i, e(m, i), at(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D94. (@ \textcolor{blue}{$+v^{m, n}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}} t_{n}^{\tilde{a}_{n,
    // n}} t_{m, j}^{e_{m, j} \tilde{b}_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m,
    // j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + ((LTOp) V_mn_ef(m, n, f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, e(m, j), bt(m, j))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D95. (@ \textcolor{blue}{$+v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{n}^{\tilde{b}_{n,
    // n}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n,
    // n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, e(m, i), at(m, i))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))
    ////////////////////////////////////////////////////////////////

    // D96. (@ \textcolor{blue}{$+\frac{1}{2} v^{m a_{i, j}}_{i e_{m, j}} t_{m, j}^{\tilde{b}_{m, j}
    // e_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_ma_ie(m, a(i, j), i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D97. (@ \textcolor{blue}{$+\frac{1}{2}v^{m b_{i, j}}_{j e_{m, i}} t_{m, i}^{\tilde{a}_{m, i}
    // e_{m, i} } S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_ma_ie(m, b(i, j), j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D98. (@ \textcolor{blue}{$-\frac{1}{2} v^{m, n}_{i e_{m, j}} t_{n}^{\tilde{a}_{n, n}} t_{m,
    // j}^{\tilde{b}_{m, j} e_{m, j} } S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-(1.0 / 2.0) * (LTOp) V_mn_ie(m, n, i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) t1(n, at(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D99. (@ \textcolor{blue}{$-\frac{1}{2}v^{m, n}_{j e_{m, i}} t_{n}^{\tilde{b}_{n, n}} t_{m,
    // i}^{\tilde{a}_{m, i}e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + (-(1.0 / 2.0) * (LTOp) V_mn_ie(m, n, j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D100. (@ \textcolor{blue}{$+\frac{1}{2} v^{m a_{i, j}}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}}
    // t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_ma_ef(m, a(i, j), f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D101. (@ \textcolor{blue}{$+\frac{1}{2} v^{m b_{i, j}}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}}
    // t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_ma_ef(m, b(i, j), f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D102. (@ \textcolor{blue}{$-\frac{1}{4} v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{f_{i, n}
    // \tilde{a}_{i, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n}
    // a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-(1.0 / 4.0) * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, f(i, n), at(i, n)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D103. (@ \textcolor{blue}{$-\frac{1}{4} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{f_{j, n}
    // \tilde{b}_{j, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i}
    // a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
    + (-(1.0 / 4.0) * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, f(j, n), bt(j, n)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D104. (@ \textcolor{blue}{$-\frac{1}{2}v^{m, n}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}}
    // t_{n}^{\tilde{a}_{n, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n,
    // n} a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + (-(1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D105. (@ \textcolor{blue}{$-\frac{1}{2} v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}}
    // t_{n}^{\tilde{b}_{n, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m,
    // i} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + (-(1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))
    ///////////////////////////////////////

    // D106. (@ \textcolor{blue}{$-\frac{1}{2} v^{m a_{i, j}}_{i e_{m, j}} t_{m, j}^{\tilde{b}_{m,
    // j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (-(1.0 / 2.0) * (LTOp) V_ma_ie(m, a(i, j), i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D107. (@ \textcolor{blue}{$-\frac{1}{2} v^{m b_{i, j}}_{j e_{m, i}} t_{m, i}^{\tilde{a}_{m,
    // i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
    + (-(1.0 / 2.0) * (LTOp) V_ma_ie(m, b(i, j), j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D108. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{i e_{m, j}} t_{n}^{\tilde{a}_{n, n}} t_{m,
    // j}^{\tilde{b}_{m, j} e_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, j)(i,
    // j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + (+(1.0 / 2.0) * (LTOp) V_mn_ie(m, n, i, e(m, j)) * (LTOp) t2(m, j, bt(m, j), e(m, j)) *
       (LTOp) t1(n, at(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D109. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{j e_{m, i}} t_{n}^{\tilde{b}_{n, n}} t_{m,
    // i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ie(m, n, j, e(m, i)) * (LTOp) t2(m, i, at(m, i), e(m, i)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, i, i, j, at(m, i), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D110. (@ \textcolor{blue}{$-\frac{1}{2} v^{m a_{i, j}}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}}
    // t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + (-(1.0 / 2.0) * (LTOp) V_ma_ef(m, a(i, j), f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D111. (@ \textcolor{blue}{$-\frac{1}{2} v^{m b_{i, j}}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}}
    // t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} $} @)
    + (-(1.0 / 2.0) * (LTOp) V_ma_ef(m, b(i, j), f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)))

    // D112. (@ \textcolor{blue}{$+\frac{1}{4} v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{f_{i, n}
    // \tilde{a}_{i, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(i, n)(i, j)}_{\tilde{a}_{i, n}
    // a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}} $} @)
    + ((1.0 / 4.0) * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, f(i, n), at(i, n)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) *
       (LTOp) S(i, n, i, j, at(i, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D113. (@ \textcolor{blue}{$+\frac{1}{4} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{f_{j, n}
    // \tilde{b}_{j, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i}
    // a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}} $} @)
    + ((1.0 / 4.0) * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, f(j, n), bt(j, n)) * (LTOp) t2(m, i, at(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(j, n, i, j, bt(j, n), b(i, j)))

    // D114. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}}
    // t_{n}^{\tilde{a}_{n, n}} t_{m, j}^{\tilde{b}_{m, j} e_{m, j}} S^{(n, n)(i, j)}_{\tilde{a}_{n,
    // n} a_{i, j}} S^{(m, j)(i, j)}_{\tilde{b}_{m, j} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, bt(m, j), e(m, j))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, j, i, j, bt(m, j), b(i, j)))

    // D115. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}}
    // t_{n}^{\tilde{b}_{n, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m,
    // i} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, at(m, i), e(m, i))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, i, i, j, at(m, i), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))
    /////////////////////////////////////////////////////////////////

    // D116. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{i e_{m, j}}  t_{m, j}^{\tilde{a}_{m, j} e_{m, j}}
    // S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, b(i, j), i, e(m, j)) * (LTOp) t2(m, j, at(m, j), e(m, j)) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)))

    // D117. (@ \textcolor{blue}{$-v^{m a_{i, j}}_{j e_{m, i}}  t_{m, i}^{\tilde{b}_{m, i} e_{m, i}}
    // S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    - ((LTOp) V_ma_ie(m, a(i, j), j, e(m, i)) * (LTOp) t2(m, i, bt(m, i), e(m, i)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D118. (@ \textcolor{blue}{$+v^{m, n}_{i e_{m, j}} t_{n}^{\tilde{b}_{n, n}} t_{m,
    // j}^{\tilde{a}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}} S^{(n, n)(i,
    // j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(m, n, i, e(m, j)) * (LTOp) t2(m, j, at(m, j), e(m, j)) *
       (LTOp) t1(n, bt(n, n)) * (LTOp) S(m, j, i, j, at(m, j), a(i, j)) *
       (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D119. (@ \textcolor{blue}{$+v^{m, n}_{j e_{m, i}} t_{n}^{\tilde{a}_{n, n}} t_{m,
    // i}^{\tilde{b}_{m, i} e_{m, i}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, i)(i,
    // j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + ((LTOp) V_mn_ie(m, n, j, e(m, i)) * (LTOp) t2(m, i, bt(m, i), e(m, i)) *
       (LTOp) t1(n, at(n, n)) * (LTOp) S(n, n, i, j, at(n, n), a(i, j)) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D120. (@ \textcolor{blue}{$-v^{m b_{i, j}}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}} t_{m,
    // j}^{\tilde{a}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j} a_{i, j}}$} @)
    - ((LTOp) V_ma_ef(m, b(i, j), f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, at(m, j), e(m, j))) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)))

    // D121. (@ \textcolor{blue}{$-v^{m a_{i, j}}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{m,
    // i}^{\tilde{b}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    - ((LTOp) V_ma_ef(m, a(i, j), f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, bt(m, i), e(m, i))) *
       (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D122. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{i, n} e_{m, j}} t_{i, n}^{f_{i, n}
    // \tilde{b}_{i, n}} t_{m, j}^{\tilde{a}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m, j}
    // a_{i, j}} S^{(i, n)(i, j)}_{\tilde{b}_{i, n} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(i, n), e(m, j)) *
       ((LTOp) t2(i, n, f(i, n), bt(i, n)) * (LTOp) t2(m, j, at(m, j), e(m, j))) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)) * (LTOp) S(i, n, i, j, bt(i, n), b(i, j)))

    // D123. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{f_{j, n}
    // \tilde{a}_{j, n}} t_{m, i}^{\tilde{b}_{m, i} e_{m, i}} S^{(j, n)(i, j)}_{\tilde{a}_{j, n}
    // a_{i, j}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
    + ((1.0 / 2.0) * (LTOp) V_mn_ef(m, n, f(j, n), e(m, i)) *
       ((LTOp) t2(j, n, f(j, n), at(j, n)) * (LTOp) t2(m, i, bt(m, i), e(m, i))) *
       (LTOp) S(j, n, i, j, at(j, n), a(i, j)) * (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    // D124. (@ \textcolor{blue}{$+v^{m, n}_{f_{i, i} e_{m, j}} t_{i}^{f_{i, i}}
    // t_{n}^{\tilde{b}_{n, n}} t_{m, j}^{\tilde{a}_{m, j} e_{m, j}} S^{(m, j)(i, j)}_{\tilde{a}_{m,
    // j} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
    + ((LTOp) V_mn_ef(m, n, f(i, i), e(m, j)) *
       ((LTOp) t1(i, f(i, i)) * (LTOp) t2(m, j, at(m, j), e(m, j))) * (LTOp) t1(n, bt(n, n)) *
       (LTOp) S(m, j, i, j, at(m, j), a(i, j)) * (LTOp) S(n, n, i, j, bt(n, n), b(i, j)))

    // D125. (@ \textcolor{blue}{$+v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}}
    // t_{n}^{\tilde{a}_{n, n}} t_{m, i}^{\tilde{b}_{m, i} e_{m, i}} S^{(n, n)(i, j)}_{\tilde{a}_{n,
    // n} a_{i, j}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)

    + ((LTOp) V_mn_ef(m, n, f(j, j), e(m, i)) *
       ((LTOp) t1(j, f(j, j)) * (LTOp) t2(m, i, bt(m, i), e(m, i))) * (LTOp) t1(n, at(n, n)) *
       (LTOp) S(n, n, i, j, at(n, n), a(i, j)) * (LTOp) S(m, i, i, j, bt(m, i), b(i, j)))

    ;

  i0(a(i, j), b(i, j), i, j).update(op);
  OpExecutor op_exec{sch, symbol_table};
  op_exec.pretty_print_binarized(i0);
  op_exec.execute(i0);

  std::cout << "i0(a(i, j), b(i, j), i, j):"
            << "\n";
  print_tensor(i0);
}

using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

DependencyMap construct_full_dependency_2(const TiledIndexSpace& in_tis1,
                                          const TiledIndexSpace& in_tis2,
                                          const TiledIndexSpace& out_tis) {
  DependencyMap result;
  for(const auto& tile_1: in_tis1) {
    for(const auto& tile_2: in_tis2) { result.insert({{tile_1, tile_2}, out_tis}); }
  }
  return result;
}

template<typename T>
void ccsd_driver_initial(Scheduler& sch) {
  // using DependencyMap = std::map<IndexVector, TiledIndexSpace>;

  TiledIndexSpace LMO{IndexSpace{range(4)}};

  DependencyMap depMO = construct_full_dependency_2(LMO, LMO, LMO);
  // DependencyMap depMO = {
  //   {{1, 1}, {TiledIndexSpace{LMO, IndexVector{0, 1, 4, 5, 8}}}},
  //   {{2, 3}, {TiledIndexSpace{LMO, IndexVector{0, 6, 8}}}},
  //   {{3, 0}, {TiledIndexSpace{LMO, IndexVector{1, 3, 7, 9}}}},
  //   {{4, 1}, {TiledIndexSpace{LMO, IndexVector{2, 4, 7}}}},
  //   {{7, 8}, {TiledIndexSpace{LMO, IndexVector{1, 5, 7}}}}};

  TiledIndexSpace PNO{LMO, {LMO, LMO}, depMO};

  auto [i, j, k, l, m, n] = LMO.labels<6>("all");
  auto [a, b, c, d, e, f] = PNO.labels<6>("all");

  Tensor<T> d_e{};
  Tensor<T> d_r1{a(i, i), i};
  Tensor<T> d_r2{a(i, j), b(i, j), i, j};
  Tensor<T> S{i, j, m, n, b(i, j), a(m, n)};
  Tensor<T> S_1{i, j, b(i, i), a(j, j)};
  Tensor<T> S_2{i, j, k, b(i, j), a(k, k)};
  Tensor<T> S_3{m, i, b(m, i), a(i, i)};
  Tensor<T> S_4{i, m, b(i, m), a(i, i)};
  Tensor<T> S_5{i, j, k, b(i, i), a(j, k)};

  Tensor<T> t1{i, e(i, i)};
  Tensor<T> t2{i, j, a(i, j), e(i, j)};

  // All versions of Fock Matrix
  Tensor<T> F_ij{i, j};
  Tensor<T> F_ai{a(i, i), i};
  Tensor<T> F_ia{i, a(i, i)};
  Tensor<T> F_ab{i, a(i, i), b(i, i)};

  // All versions of V
  Tensor<T> V_mn_ei{m, n, e(m, n), i};
  Tensor<T> V_mn_ef{m, n, e(m, n), f(m, n)};
  Tensor<T> V_mn_ij{m, n, i, j};
  Tensor<T> V_mn_ie{m, n, i, e(i, i)};
  Tensor<T> V_ma_ei{m, a(i, i), e(m, m), i};
  Tensor<T> V_ma_ie{m, a(i, i), i, e(m, m)};
  Tensor<T> V_ma_ef{m, a(m, m), e(m, m), f(m, m)};
  Tensor<T> V_ma_fe{m, a(m, m), f(m, m), e(m, m)};
  Tensor<T> V_ab_cd{i, j, a(i, j), b(i, j), c(i, j), d(i, j)};
  Tensor<T> V_ab_ij{a(i, j), b(i, j), i, j};
  Tensor<T> V_ab_ie{a(i, i), b(i, i), i, e(i, i)};
  Tensor<T> V_am_ef{a(m, m), m, e(m, m), f(m, m)};
  Tensor<T> V_am_ij{a(i, j), m, i, j};
  Tensor<T> V_am_ie{a(i, i), m, i, e(i, i)};
  std::cout << "Tensors declared"
            << "\n";

  sch
    .allocate(d_e, d_r1, d_r2, S, S_1, S_2, S_3, S_4, S_5, t1, t2, F_ij, F_ai, F_ia, F_ab, V_mn_ei,
              V_mn_ef, V_ma_fe, V_ma_ei, V_ma_ie, V_ma_ef, V_ab_cd, V_ab_ij, V_mn_ij, V_mn_ie,
              V_am_ef, V_ab_ie, V_am_ij, V_am_ie)
    .execute();

  std::cout << "Tensors allocated"
            << "\n";
  sch(F_ij() = 1.0)(F_ai() = 1.0)(F_ia() = 1.0)(F_ab() = 1.0)(V_mn_ei() = 1.0)(V_mn_ef() = 1.0)(
    V_mn_ij() = 1.0)(V_ma_ei() = 1.0)(V_ma_ef() = 1.0)(V_ab_cd() = 1.0)(V_ma_fe() = 1.0)(
    V_ma_ie() = 1.0)(V_ab_ij() = 1.0)(V_mn_ie() = 1.0)(V_am_ef() = 1.0)(V_am_ij() = 1.0)(
    V_ab_ie() = 1.0)(V_am_ie() = 1.0)(d_e() = 0.0)(d_r1() = 0.0)(d_r2() = 0.0)(S() = 1.0)(
    S_1() = 1.0)(S_2() = 1.0)(S_3() = 1.0)(S_4() = 1.0)(S_5() = 1.0)(t1() = 1.0)(t2() = 1.0)
    .execute();

  std::cout << "Tensors set initial values"
            << "\n";

  ccsd_e(sch, LMO, PNO, d_e, F_ia, V_mn_ef, t1, t2);
  ccsd_t1(sch, LMO, PNO, d_r1, t1, t2, S_1, S_2, S_3, S_4, F_ai, F_ia, F_ab, F_ij, V_mn_ef, V_mn_ei,
          V_ma_ei, V_ma_ie, V_ma_ef);
  // ccsd_t2(sch, LMO, PNO, d_r2, t1, t2, S, F_ia, F_ab, F_ij, V_mn_ef, V_mn_ei,
  //         V_ma_ie, V_ma_ef, V_ab_ij, V_mn_ij, V_mn_ie, V_ab_cd, V_am_ef,
  //         V_ab_ie, V_am_ij, V_am_ie);
}

template<typename T>
void example_contractions(Scheduler& sch) {
  /**
   * @brief Setup tiled index spaces
   *
   */
  TiledIndexSpace RedundantPAO;
  TiledIndexSpace NonRedundantPAO;
  TiledIndexSpace LMO;
  TiledIndexSpace LMOPairs;
  TiledIndexSpace MOVirtual;
  TiledIndexSpace PNO;
  TiledIndexSpace CoulumbMetric;
  TiledIndexSpace AO;

  /**
   * @brief Setup index labels
   *
   */
  auto [mu, nu]                     = AO.labels<2>("all");
  auto [mutp, nutp, gutp]           = RedundantPAO.labels<3>("all");
  auto [mut, nut, gut]              = NonRedundantPAO.labels<3>("all");
  auto [i, j, k, l, m, n]           = LMO.labels<6>("all");
  auto [ii, ij, ik, ji, jk, kl]     = LMOPairs.labels<6>("all");
  auto [a, b, c, d, e, f]           = MOVirtual.labels<6>("all");
  auto [at, bt, ct, dt, et, ft]     = PNO.labels<6>("all");
  auto [at_ii, at_ij, bt_ij, bt_kl] = PNO.labels<4>("all");
  auto [K, L, M]                    = CoulumbMetric.labels<3>("all");

  /**
   * @brief Input Tensors
   */
  // 3-electron integrals
  Tensor<T> TE1{mutp, nutp, K}; // size RPAO * RPAO * CM
  Tensor<T> TEmix{k, mutp, K};  // size LMO * RPAO * CM
  Tensor<T> TEoo{k, l, K};      // size LMO * LMO * CM
  // Redundant PAOs to non-redundant PAOs
  Tensor<T> X{mutp, mut}; // size RPAO * PAO
  // Transformation matrices
  Tensor<T> P{mut, at_ij}; // size NRPAO * LMOP * PNO
  // Fock matrix in AO basis
  Tensor<T> F{mu, nu}; // size AO * AO
  // overlap matrix in RPAO basis
  Tensor<T> Smunu{mutp, nutp}; // size RPAO * RPAO
  // Transformation matrix from RPAO basis to PNO
  Tensor<T> D{ij, mutp, at_ij}; // size RPAO * LMOP * PNO
  // Transformation matrix from AO to MO occupied
  Tensor<T> LCAO{mu, k}; // size AO * MO_occ (same as AO * LMO)
  // Transformation matrix from AO to NRPAO
  Tensor<T> Y{mu, mut}; // size AO * NRPAO

  sch.allocate(TE1, TEmix, TEoo, X, P, F, Smunu, D, LCAO, Y).execute();

  // precomputed V
  Tensor<T> V{ij, at_ij, bt_ij}; // size LMOP * PNO * PNO
  // Overlapped matrix
  Tensor<T> S{ij, kl, at_ij, bt_kl}; // size LMOP * LMOP * PNO * PNO
  // Fock matrix occupied space
  Tensor<T> Foo{i, j}; // size LMO * LMO. Or is it LMOP?
  // Fock matrix virtual space
  Tensor<T> Faoao{mut, nut}; // size NRPAO * NRPAO
  // Rest of Fock matrix is zero

  sch.allocate(V, S, Foo, Faoao).execute();

  // V(ij, at_ij, bt_ij) =
  //     ∀ (	(i,j) ∈  ij,
  //  	      mutp ∈  DNRPAO(i) ∪ DNRPAO(j),
  //           nutp ∈  DNRPAO(i) ∪ DNRPAO(j),
  //           K ∈  DCM(i) ∪ DCM(j))
  //       TEmix(i, mutp, K) *
  //       TEmix(j, nutp, K) *
  //       d(mutp, at_ij) *
  //       d(nutp, bt_ij);

  // intermediate for V computation
  Tensor<T> int_0_V{i, j, mutp, nutp};
  Tensor<T> int_1_V{ij, mutp, bt_ij};

  sch
    .allocate(int_0_V, int_1_V)(int_0_V(i, j, mutp, nutp) = TEmix(i, mutp, K) * TEmix(j, nutp, K))(
      int_1_V(ij, mutp, bt_ij) = int_0_V(i, j, mutp, nutp) * D(ij, nutp, bt_ij))(
      V(ij, at_ij, bt_ij) = int_1_V(ij, mutp, bt_ij) * D(ij, mutp, at_ij))
    .execute();

  // S(ij, kl, at_ij, bt_kl) =
  //            d(ij, mut, at_ij) *
  //            S(mut,nut) *
  //            d(kl, nut, bt_kl);
  Tensor<T> int_2_S{ij, nut, at_ij};

  sch
    .allocate(int_2_S)(int_2_S(ij, nut, at_ij) = D(ij, mut, at_ij) * Smunu(mut, nut))(
      S(ij, kl, at_ij, bt_kl) = int_2_S(ij, nut, at_ij) * D(kl, nut, bt_kl))
    .execute();

  // Foo(i,j) = F(mu, nu) * LCAO(mu, i) * LCAO(nu, j);
  Tensor<T> int_3_Foo{i, nu};

  sch
    .allocate(int_3_Foo)(int_3_Foo(i, nu) =
                           F(mu, nu) * LCAO(mu, i))(Foo(i, j) = int_3_Foo(i, nu) * LCAO(nu, j))
    .execute();

  // Faoao(mut, nut) = F(mu, nu) * Y(nu, nut) * Y(mu, mut);
  Tensor<T> int_4_Faoao{mu, nut};

  sch
    .allocate(int_4_Faoao)(int_4_Faoao(mu, nut) = F(mu, nu) * Y(nu, nut))(
      Faoao(mut, nut) = int_4_Faoao(mu, nut) * Y(mu, mut))
    .execute();
#if 0
  /**
   * @brief Most expensive contractions 125, 105, 85, 71, 61, 43, 11
   * 
   */
  Tensor<T> i0{at_ij, bt_ij, ij};
  // D125. (@ \textcolor{blue}{$+v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{n}^{\tilde{a}_{n, n}} t_{m, i}^{\tilde{b}_{m, i} e_{m, i}} S^{(n, n)(i, j)}_{\tilde{a}_{n, n} a_{i, j}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
  // +(V(m, n, f(j, j), e(m, i)) * (t1(j, f(j, j)) * t2(m, i, bt(m, i), e(m, i))) *
  //   t1(n, at(n, n)) * S(n, n, i, j, at(n, n), a(i, j)) *
  //   S(m, i, i, j, bt(m, i), b(i, j)))
  
  i0(at_ij, bt_ij, ij) +=
        +V(m, n, f(j, j), e(m, i)) *
        t1(jj, f_jj) * t2(mi, bt_mi, e_mi) * t1(nn, at_nn) *
        S(nn, ij, at_nn, at_ij) * S(mi, ij, bt_mi, bt_ij);

  // D105. (@ \textcolor{blue}{$-\frac{1}{2} v^{m, n}_{f_{j, j} e_{m, i}} t_{j}^{f_{j, j}} t_{n}^{\tilde{b}_{n, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
  // +(-(1.0 / 2.0) * V(m, n, f(j, j), e(m, i)) *
  //   (t1(j, f(j, j)) * t2(m, i, at(m, i), e(m, i))) * t1(n, bt(n, n)) *
  //   S(m, i, i, j, at(m, i), a(i, j)) * S(n, n, i, j, bt(n, n), b(i, j)));
  i0(at_ij, bt_ij, ij) +=;
  -1.0 / 2.0 * V(m, n, f(j, j), e(m, i)) *
    t1(jj, f_jj) * t2(mi, at_mi, e_mi) * t1(nn, bt_nn) *
    S(mi, ij, at_mi, a_ij) * S(nn, ij, bt_nn, b_ij);

  // D85. (@ \textcolor{blue}{$+\frac{1}{2} v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{\tilde{b}_{j, n} f_{j, n}} t_{m, i}^{\tilde{a}_{m, i} e_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}}$} @)
  // +((1.0 / 2.0) * V_mn_ef(m, n, f(j, n), e(m, i)) *
  //   (t2(j, n, bt(j, n), f(j, n)) * t2(m, i, at(m, i), e(m, i))) *
  //   S(m, i, i, j, at(m, i), a(i, j)) * S(j, n, i, j, bt(j, n), b(i, j)));
  i0(at_ij, bt_ij, ij) +=
      1.0 / 2.0 * V_mn_ef(m, n, f(j, n), e(m, i)) *
        (t2(jn, bt_jn, ft_jn) * t2(mi, at_mi, et_mi) *
        S(mi, ij, at_mi, at_ij) * S(jn, ij, bt_jn, b_ij);

  // D71. (@ \textcolor{blue}{$-v^{m, n}_{f_{j, n} e_{m, i}} t_{j, n}^{\tilde{b}_{j, n} f_{j, n}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(j, n)(i, j)}_{\tilde{b}_{j, n} b_{i, j}} $} @)
  // +(2.0 * V_mn_ef(m, n, f(j, n), e(m, i)) *
  //   (t2(j, n, bt(j, n), f(j, n)) * t2(m, i, e(m, i), at(m, i))) *
  //   S(m, i, i, j, at(m, i), a(i, j)) * S(j, n, i, j, bt(j, n), b(i, j)));
  i0(at_ij, bt_ij, ij) +=
  +2.0 * V_mn_ef(m, n, f(j, n), e(m, i)) *
    t2(jn, bt_jn, ft_jn) * t2(mi, et_mi, at_mi) *
    S(mi, ij, at_mi, at_ij) * S(jn, ij, bt_jn, bt_ij);

  // D61. (@ \textcolor{blue}{$-2v^{nm}_{j e_{m, i}} t_{n}^{b_{n, n}} t_{m, i}^{e_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}}$} @)
  // +(-2.0 * V(n, m, j, e(m, i)) * t2(e(m, i), at(m, i), m, i) * t1(n, bt(n, n)) *
  //   S(m, i, i, j, at(m, i), a(i, j)) * S(n, n, i, j, bt(n, n), b(i, j)));
  i0(at_ij, bt_ij, ij) +=
  -2.0 * V(n, m, j, e(m, i)) * t2(et_mi, at_mi, mi) * t1(nn, bt_nn) * 
    S(mi, ij, at_mi, at_ij) * S(nn, ij, bt_nn, b_ij);

  // D43. (@ \textcolor{blue}{$+v^{m, n}_{f_{n, n} e_{j, j}}  t_{j}^{e_{j, j}} t_{n}^{f_{n, n}} t_{m, i}^{\tilde{b}_{m, i} \tilde{a}_{m, i}} S^{(m, i)(i, j)}_{\tilde{a}_{m, i} a_{i, j}} S^{(m, i)(i, j)}_{\tilde{b}_{m, i} b_{i, j}}$} @)
  // +(V(m, n, f(n, n), e(j, j)) * t1(j, e(j, j)) * t1(n, f(n, n)) *
  //   t2(m, i, bt(m, i), at(m, i)) * S(m, i, i, j, at(m, i), a(i, j)) *
  //   S(m, i, i, j, bt(m, i), b(i, j)));
  i0(at_ij, bt_ij, ij) +=
  V(m, n, f(n, n), e(j, j)) * t1(jj, et_jj) * t1(nn, ft_nn) *
    t2(mi, bt_mi, at_mi) * S(mi, ij, at_mi, at_ij) *
    S(mi, ij, bt_mi, bt_ij);

  // D11. (@ \textcolor{blue}{$+v^{m, n}_{e_{i, i} f_{j, j}} t_{i}^{e_{i, i}} t_{j}^{f_{j, j}} t_{m}^{\tilde{a}_{m, m}} t_{n}^{\tilde{b}_{n, n}} S^{(m, m)(i, j)}_{\tilde{a}_{m, m} a_{i, j}} S^{(n, n)(i, j)}_{\tilde{b}_{n, n} b_{i, j}}$} @)
  // +(V(m, n, e(i, i), f(j, j)) * (t1(i, e(i, i)) * t1(j, f(j, j))) *
  //   t1(m, at(m, m)) * t1(n, bt(n, n)) * S(m, m, i, j, at(m, m), a(i, j)) *
  //   S(n, n, i, j, bt(n, n), b(i, j)));
  i0(at_ij, bt_ij, ij) +=
  +V(m, n, e(i, i), f(j, j)) * t1(ii, et_ii) * t1(jj, ft_jj) *
    t1(mm, at_mm) * t1(nn, bt_nn) * S(mm, ij, at_mm, a_ij) *
    S(nn, ij, bt_nn, bt_ij);

  // std::cout << "i0(a_ij, b_ij, ij):"
  //           << "\n";
  // print_tensor(i0);
#endif
}

void test_new_alloc(Scheduler& sch) {
  using T = double;
  TiledIndexSpace TIS{IndexSpace{range(10)}, 10};

  SymbolTable symbol_table;

  Tensor<T> A{TIS, TIS};
  Tensor<T> B{TIS, TIS};
  Tensor<T> C{TIS, TIS};

  Tensor<T> D{TIS, TIS};
  Tensor<T> E{TIS, TIS};

  TAMM_REGISTER_SYMBOLS(symbol_table, A, B, C, D, E);
  OpExecutor op_executor{sch, symbol_table};

  sch.allocate(A, B)(A() = 1.0)(B() = 2.0).execute();

  auto op1 = (new_ops::LTOp) A("i", "j");
  auto op2 = (new_ops::LTOp) A("i", "j") * (new_ops::LTOp) B("j", "k");
  auto op3 =
    (new_ops::LTOp) A("i", "j") * (new_ops::LTOp) B("j", "k") * (new_ops::LTOp) C("k", "l");

  C("i", "j").update(op2);

  // op_executor.execute(C);
  // print_tensor_all(C);

  D("i", "l").update(op3);
  // op_executor.execute(D);
  // print_tensor_all(D);

  E("i", "j").update((new_ops::LTOp) C("i", "l") * (new_ops::LTOp) D("l", "j"));
  op_executor.execute(E);
  print_tensor_all(E);
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  ProcGroup         pg  = ProcGroup::create_world_coll();
  auto              mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW   distribution;
  ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
  Scheduler         sch{*ec};

  // ccsd_driver_initial<double>(sch);
  // example_contractions<double>(sch);
  test_new_alloc(sch);

  tamm::finalize();
  return 0;
}
