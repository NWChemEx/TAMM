#include <tuple>
#include <vector>

#include "mso.h"
#include "ao.h"
#include "tensor.h"
#include "labeled_tensor.h"
#include "scheduler.h"


using namespace tammy;

void four_index_transform(const AO& ao,
                          const MSO& mso,
                          Tensor<double> tC,
                          Tensor<double> tV) {
  IndexLabel f1, f2, f3, f4;
  IndexLabel p, q, r, s, E;
  
  std::tie(f1, f2, f3, f4) = ao.N(1, 2, 3, 4);
  std::tie(p, q, r, s) = mso.N(1, 2, 3, 4);

  Tensor<double> I0 = Tensor<double>::create<TensorImpl<double>>(E | f1 + f2 | f3 + f4/* , perm_s4d */);
  Tensor<double> I1 = Tensor<double>::create<TensorImpl<double>>(E | p + f2 | f3 + f4/* , perm_n2d + perm_s2d */);
  Tensor<double> I2 = Tensor<double>::create<TensorImpl<double>>(E | p + q | f3 + f4/* , perm_a2d + perm_n2d */);
  Tensor<double> I3 = Tensor<double>::create<TensorImpl<double>>(E | p + q | r + f4/* , perm_s2d + perm_n2d */);
  
  //I0(f1, f2, f2, f4) = integral_function()
  I1(p, f2, f3, f4) += tC(f1, p) * I0(f1, f2, f3, f4);
  I2(p, r, f3, f4)  += tC(f2, r) * I1(p, f2, f3, f4);
  I3(p, r, q, f4)   += tC(f3, q) * I2(p, r, f3, f4);
  tV(p, r, q, s)     = tC(f4, s) * I3(p, r, q, f4);
}

void two_index_transform(const AO& ao,
                         const MSO& mso,
                         Tensor<double> tC,
                         Tensor<double> tF_ao,
                         Tensor<double> tF_mso) {
  IndexLabel f1, f2;
  IndexLabel p, q, E;
  
  std::tie(f1, f2) = ao.N(1, 2);
  std::tie(p, q) = mso.N(1, 2);

  Tensor<double> I0 = Tensor<double>::create<TensorImpl<double>>(E | p | f2);
  
  I0(p, f2)    = tC(f1, p) * tF_ao(f1, f2);
  tF_mso(p, q) = tC(f1, q) * I0(f1, p);
}
 
// replicate C horizontally to construct C that has both spins
Tensor<double>
construct_spin_C(const AO& ao,
                 Tensor<double> tC_in) {
  
}


void
compute_two_body_fock(Scheduler& sch,
                      const AO& ao,
                      Tensor<double> tD,
                      Tensor<double> F) {
  IndexLabel a, b, c, d, E;

  std::tie(a, b, c, d) = ao.N(1,2, 3, 4);

  Tensor<double> G = Tensor<double>::create<TensorImpl<double>>(E | a + b | E);

  // Tensor<double> Integral = Tensor<double>::create<TensorImpl<double>>(E | f1 + f2 | f3 + f4);
  Tensor<double> Integrals = Tensor<double>::create<TensorImpl<double>>(E | a + b | c + d);
  
  // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
  //    F(a,b) += (ab|cd) * D(c,d)

  sch(
      G(a,b) = 0,
      G(a,b) += Integrals(a,c,b,d) * tD(c,d),
      //    F(c,d) += (ab|cd) * D(a,b)
      G(c,d) += Integrals(a,c,b,d) * tD(a,b),
      //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
      G(b,d) -= 0.25 * Integrals(a,c,b,d) * tD(a,c),
      //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
      G(b,c) -= 0.25 * Integrals(a,c,b,d) * tD(a,d),
      //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
      G(a,c) -= 0.25 * Integrals(a,c,b,d) * tD(b,d),
      //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
      G(a,d) -= 0.25 * Integrals(a,c,b,d) * tD(b,c),
      F(a,b) += 0.5*G(a,b),
      F(a,b) += 0.5*G(b,a)
      );
}



void
hartree_fock(const AO& ao,
             Tensor<double> C,
             Tensor<double> F) {

  IndexLabel a, b, c, E;
  std::tie(a, b, c) = ao.N(1, 2, 3);

  // compute overlap integrals
  Tensor<double> S/*  = compute_1body_ints(shells, Operator::overlap) */;

  // compute kinetic-energy integrals
  Tensor<double> T/*  = compute_1body_ints(shells, Operator::kinetic) */;

  // compute nuclear-attraction integrals
  Tensor<double> V/*  = compute_1body_ints(shells, Operator::nuclear, atoms) */;
  
  Tensor<double> H = Tensor<double>::create<TensorImpl<double>>(a + b | E | E);

  Scheduler()(
      H()  = T(),
      H() += V())
      .deallocate(T, V)
      .execute();

  Tensor<double> D;
//   eigen_solve(H, S, eps, C);

  Scheduler() (
      D(a, b) = C(a, c) * C(c, b)
               ).execute();
  
  const auto maxiter = 100;
  const auto conv = 1e-12;
  double rmsd = 0.0;
  double ediff = 0.0;
  double ehf = 0.0;
  int iter = 0;

  Tensor<double> EHF = Tensor<double>::create<TensorImpl<double>>(E|E|E);
  Tensor<double> EHF_last = Tensor<double>::create<TensorImpl<double>>(E|E|E);
  Tensor<double> EDIFF = Tensor<double>::create<TensorImpl<double>>(E|E|E);
  Tensor<double> RMSD = Tensor<double>::create<TensorImpl<double>>(E|E|E);

  Tensor<double> TMP = Tensor<double>::create<TensorImpl<double>>(a + b | E | E);
  Tensor<double> TMP1 = Tensor<double>::create<TensorImpl<double>>(a + b | E | E);
  Tensor<double> D_last = Tensor<double>::create<TensorImpl<double>>(a + b | E | E);
  
  do {
    Scheduler sch;
    // Save a copy of the energy and the density
    sch(
        EHF_last() = EHF(),
        D_last(a, b) = D(a, b)
        );

    // build a new Fock matrix
    sch(F() = H());
    compute_two_body_fock(sch, ao, D, F);
    sch.execute();
    
    // solve F C = e S C
    // eigen_solve(F, S, eps, C);
    
    sch(D(a, b) = C(a, c) * C(c, b));

    // compute HF energy
    sch(
        TMP(a,b)  = H(a,b),
        TMP(a,b) += F(a,b),
        EHF()     = D(a, b) * TMP(a, b)
        );

    // compute difference with last iteration
    sch(
        EDIFF()    = EHF() - EHF_last(),
        TMP1(a, b) = D(a, b),
        TMP1(a,b) += -1.0 * D_last(a, b),
        RMSD()     = TMP1(a, b) * TMP1(a, b)
        ).execute();

    ehf = EHF.get();
    ediff = EDIFF.get();
    rmsd = RMSD.get();
    
    iter++;
  } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));
} 