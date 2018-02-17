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

  PermGroup perm_s4 = PermGroup::antisymm(2, 2);
  PermGroup perm_a4 = PermGroup::symm(4);
  PermGroup perm_i2 = perm_a4.remove_index(3);

  Tensor<double> I0 = TensorImpl<double>::create(E | f1 + f2 | f3 + f4, perm_s4);
  Tensor<double> I1 = TensorImpl<double>::create(E | p + f2 | f3 + f4, perm_s4);
  Tensor<double> I2 = TensorImpl<double>::create(E | p + q | f3 + f4, perm_i2);
  Tensor<double> I3 = TensorImpl<double>::create(E | p + q | r + f4, perm_a4);
  
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

  Tensor<double> I0 = TensorImpl<double>::create(E | p | f2);
  
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
                      Tensor<double> tF) {
  IndexLabel a, b, c, d, E;

  std::tie(a, b, c, d) = ao.N(1,2, 3, 4);

  Tensor<double> G = TensorImpl<double>::create(E | a + b | E);

  // Tensor<double> Integral = TensorImpl<double>::create(E | f1 + f2 | f3 + f4);
  Tensor<double> Integrals = TensorImpl<double>::create(E | a + b | c + d);

  sch(
      tF(a, b) += 2.0 * tD(c, d) * Integrals(a, b, c, d),
      tF(a, b) += -1.0 * tD(c, d) * Integrals(a, c, b, d)
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
  
  Tensor<double> H = TensorImpl<double>::create(a + b | E | E);

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

  Tensor<double> EHF, EHF_last, EDIFF, RMSD;
  std::tie(EHF, EHF_last, EDIFF, RMSD) = TensorImpl<double>::create_list<4>(E|E|E);
  
  // Tensor<double> EHF = TensorImpl<double>::create(E|E|E);
  // Tensor<double> EHF_last = TensorImpl<double>::create(E|E|E);
  // Tensor<double> EDIFF = TensorImpl<double>::create(E|E|E);
  // Tensor<double> RMSD = TensorImpl<double>::create(E|E|E);

//   Tensor<double> TMP, TMP1, D_last;
//   std::tie(TMP, TMP1, D_last) = TensorImpl<double>::create_list<3>(a + b | E | E);

  Tensor<double> TMP = TensorImpl<double>::create(a + b | E | E);
  Tensor<double> TMP1 = TensorImpl<double>::create(a + b | E | E);
  Tensor<double> D_last = TensorImpl<double>::create(a + b | E | E);

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

    // ehf = EHF.get();
    // ediff = EDIFF.get();
    // rmsd = RMSD.get();
    
    iter++;
  } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));
} 
