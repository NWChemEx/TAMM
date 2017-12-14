//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
/*
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "stdio.fh"
#include "rtdb.fh"
#include "tce.fh"
#include "tce_main.fh"
#include "tce_diis.fh"
#include "tce_restart.fh"
#include "tensor/fapi.h"
*/
#include <cassert>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "tensor/gmem.h"
#include "tensor/variables.h"
#include "tensor/tensor.h"
// #include "tammx/tammx.h"

// #define maxtrials 20  // argument to be passed from Fortran
// #define nxtrials 10  // argument to be passed from Fortran
#define nroots_reduced 10  // argument to be passed from Fortran
#define thresh 1.0e-6  // argument to be passed from Fortran
#define MA_ERR "ma_error_cpp"

// #define hbard 4  // argument to be passed from Fortran
// #define print_default 20  // print parameter set in Fortran
// #define size_x1 4  // argument to be passed from Fortran
// #define size_x2 4  // argument to be passed from Fortran

/* Arguments that need to be passed to this function
 * rtdb
 * maxtrials
 * nxtrials - from tce/include/tce_diis.fh
 * hbard
 * size_x1
 * size_x2
 * ip_unused_sym - from include/tce.fh
 * eom_solver - from tce/include/tce_diis.fh, assigned in tce_energy.F
 * k_irs - from tce/include/tce_main.fh, integer k_irs(2)
 * nirreps - from tce/include/tce_main.fh
 * nroots_reduced - from tce/include/tce_diis.fh
 * irrep_x
 * irrep_y
 * ccsd_var - from tce/include/tce_main.fh, defined in tce_input.F and passed on through tce_energy.F
 * l_omegax - from tce_energy.F
 * l_omegax - from tce_energy.F
 * ipol - from tce/include/tce_main.fh
 * nocc - from tce/include/tce_main.fh, integer nocc(2)
 * int_mb will be set through set_var_cxx_
 * geom - from tce/include/tce_main.fh
 * symmetry - from tce/include/tce_main.fh, logical symmetry
 * targetsym - from tce/include/tce_main.fh, character*4 targetsym
 * thresh
 *
 */

/* xc1 is converged RHS1 vector from file tce/include/tce_diss.fh
 * xc2 is converged RHS2 vector from file tce/include/tce_diss.fh
 * xp1 is product RHS1 vector from file tce/include/tce_diss.fh
 * xp2 is product RHS2 vector from file tce/include/tce_diss.fh
 */

extern "C" {
int util_print_(const std::string &str, const std::string &print_default);
void tce_eom_init_();
void sym_irrepname_(Fint geom, int irrepp, const std::string irrepname);
void tce_hbarinit_(double *hbar, const int hbard);
void tce_eom_ipxguess_(Fint *rtdb, bool needt1, bool needt2,
                       bool false_1, bool false_2, Fint *size_x1, Fint *size_x2,
                       Fint *dummy_1, Fint *dummy_2, Fint *k_x1_offset,
                       Fint *k_x2_offset, Fint *dummy_3, Fint *dummy_4);
void tce_eom_xdiagon_(bool needt1, bool needt2, bool false_1, bool false_2,
                      Fint *size_x1, Fint *size_x2, Fint *dummy_1,
                      Fint *dummy_2, Fint *k_x1_offset,
                      Fint *k_x2_offset, Fint *dummy_3, Fint *dummy_4,
                      Fint *k_x1_offset_, Fint *k_x2_offset_,
                      Fint *dummy_5, Fint *dummy_6, double *dbl_k_omegax,
                      double *dbl_k_residual, Fint *k_hbar, Fint *iter,
                      bool eaccsd, bool ipccsd);
double util_cpusec_();
double util_wallsec_();
int ma_pop_stack_(Fint *l_residual);
void tce_eom_xtidy_();
void tce_print_ipx1_(Fint *xc1, Fint *k_x1_offset, double DECI,
                     int irrep_x);
void tce_print_ipx2_(Fint *xc2, Fint *k_x2_offset, double DECI,
                     int irrep_x);
void ipccsd_x1_cxx_(Fint *d_f1, Fint *d_i0, Fint *d_t_vo,
                    Fint *d_t_vvoo, Fint *d_v2, Fint *d_x1, Fint *d_x2,
                    Fint *k_f1_offset, Fint *k_i0_offset,
                    Fint *k_t_vo_offset, Fint *k_t_vvoo_offset,
                    Fint *k_v2_offset, Fint *k_x1_offset,
                    Fint *k_x2_offset);
void ipccsd_x2_cxx_(Fint *d_f, Fint *d_i0, Fint *d_t_vo,
                    Fint *d_t_vvoo, Fint *d_v, Fint *d_x_o,
                    Fint *d_x_voo, Fint *k_f_offset, Fint *k_i0_offset,
                    Fint *k_t_vo_offset, Fint *k_t_vvoo_offset,
                    Fint *k_v_offset, Fint *k_x_o_offset,
                    Fint *k_x_voo_offset);

void ip_ccsd123_(tamm::Tensor *f, tamm::Tensor *v, tamm::Tensor *t_vo,
                 tamm::Tensor *t_vvoo, Fint *rtdb, Fint *size_x1, Fint *size_x2,
                 Fint *k_irs, Fint nirreps, bool symmetry, std::string targetsym,
                 int maxtrials, int *nxtrials, double const &cpu, double const &wall);

void ip_ccsd_driver_cxx_(
    // Please note d_e and k_e_offset are not used
    Fint *d_e, Fint *d_f1, Fint *d_v2, Fint *d_t1,
    Fint *d_t2, Fint *k_e_offset, Fint *k_f1_offset,
    Fint *k_v2_offset, Fint *k_t1_offset, Fint *k_t2_offset,
    Fint *rtdb, Fint *size_x1, Fint *size_x2, Fint *k_irs,
    Fint nirreps, bool symmetry, std::string targetsym,
    Fint *maxtrials, Fint *nxtrials,
    double const &cpu, double const &wall) {

  /* Create Tensor objects and attach to objects
   * passed on to this function
   */

  // f[N][N]
  tamm::RangeType list_f[] = {tamm::TN, tamm::TN};
  tamm::Tensor *f = new tamm::Tensor
      (2, 1, 0, list_f, tamm::dist_nw);  // irrep_f
  f->create();
  f->attach(*k_f1_offset, 0, *d_f1);  // d_f1, k_f1_offset

  // v[N,N][N,N]
  tamm::RangeType list_v[] = {tamm::TN, tamm::TN, tamm::TN, tamm::TN};
  tamm::Tensor *v = new tamm::Tensor  // d_v2
      (4, 2, 0, list_v, tamm::dist_nw);  // irrep_v
  v->create();
  v->attach(*k_v2_offset, 0, *d_v2);  // d_v2, k_v2_offset

  // t_vo[V][O]
  tamm::RangeType list_t[] = {tamm::TV, tamm::TO};
  tamm::Tensor *t_vo = new tamm::Tensor  // d_t1
      (2, 1, 0, list_t, tamm::dist_nw);  // irrep_t
  t_vo->create();
  t_vo->attach(*k_t1_offset, 0, *d_t1);  // d_vo, k_t1_offset

  // array t_vvoo[V,V][O,O]
  tamm::RangeType list_t2[] = {tamm::TN, tamm::TN, tamm::TO, tamm::TO};
  tamm::Tensor *t_vvoo = new tamm::Tensor  // d_t2
      (4, 2, 0, list_t2, tamm::dist_nw);  // ittep_t
  t_vvoo->create();
  t_vvoo->attach(*k_t2_offset, 0, *d_t2);  // d_vvoo, k_t2_offset

  /* Call ip_ccsd driver */
  ip_ccsd(f, v, t_vo, t_vvoo,
          rtdb, size_x1, size_x2, k_irs, nirreps, symmetry, targetsym,
          maxtrials, nxtrials, cpu, wall);

  /* detach tensor objects */
  f->detach();
  v->detach();
  t_vo->detach();
  t_vvoo->detach();
}

}  // extern "C"

void tce_eom_ipxguess(double *p_evl_sorted, double const &maxdiff,
                      int nroots, double *rtdb_maxeorb, Fint *maxtrials1, int *offsets,
                      std::vector<tamm::Tensor *> const &x1_tensors,
                      std::vector<tamm::Tensor *> const &x2_tensors);

int ga_nodeid() {
  // using namespace tamm::gmem;
  // return rank();
  return tamm::gmem::rank();
}

static const double au2ev = 27.2113961;
static const double DECI = 0.10;

/*std::ostream& nodezero_print(const std::string& str,
  std::ostream &os = std::cout) {
    if (ga_nodeid() == 0) {
      os << str << std::endl;
  }
  return os;
}*/

std::ostream &nodezero_print(const std::string &str,
                             std::ostream &os = std::cout) {
  if (ga_nodeid() == 0) {
    os << str << std::endl;
  }
  return os;
}

void Expects(bool cond, const std::string &msg) {
  if (!cond) {
    std::cerr << msg << std::endl;
    assert(cond);
  }
}

void tce_eom_ipxguess_cxx_(Fint *rtdb, Fint *size_x1, Fint *size_x2,
                           Fint *k_x1_offset, Fint *k_x2_offset) {
  bool dummy_t = true;
  bool dummy_f = false;
  Fint dummy_int = 0;
  tce_eom_ipxguess_(rtdb, &dummy_t, &dummy_t, &dummy_f, &dummy_f,
                    size_x1, size_x2, &dummy_int, &dummy_int, k_x1_offset,
                    k_x2_offset, &dummy_int, &dummy_int);
}

void tce_eom_xdiagon_cxx_(Fint *size_x1, Fint *size_x2, Fint *k_x1_offset,
                          Fint *k_x2_offset, Fint *d_rx1, Fint *d_rx2, double *dbl_k_omegax,
                          double *dbl_k_residual, Fint *k_hbar, Fint *iter) {
  bool dummy_t = true;
  bool dummy_f = false;
  Fint dummy_int = 0;
  tce_eom_xdiagon_(&dummy_t, &dummy_t, &dummy_f, &dummy_f,
                   size_x1, size_x2, &dummy_int, &dummy_int,
                   k_x1_offset, k_x2_offset, &dummy_int, &dummy_int,
                   &d_rx1, &d_rx2, &dummy_int, &dummy_int,
                   dbl_k_omegax, dbl_k_residual, k_hbar, iter,
                   &dummy_f, &dummy_f);
}

void tce_hbarinit(double *hbar, int hbard) {
  for (int i = 0; i < hbard; i++) {
    for (int j = 0; j < hbard; j++) {
      hbar[i + j * hbard] = 0.0;
    }
    hbar[i + i * hbard] = 1.0e+8;
  }

}

void tce_filename_cxx_(Fint index, const std::string &xc_count) {
}



/*void indexed_tensor_create(Fint index, ) {
  Fint k_a, l_a, size, d_a;
  fn(&l_a, &k_a, &size);
  fname_and_create(&d_a, &size);
  tensor->attach(k_a, l_a, d_a);
}*/


// class Tensor;
using Irrep = int;


std::string sym_irrepname(Irrep irrep_g) {
  std::string list_objects[] = {
      "a", "a2", "b"
  };
  return list_objects[irrep_g + 1];
}

/*static Fint xc1[maxtrials];
static Fint xc2[maxtrials];
static Fint xp1[maxtrials];
static Fint xp2[maxtrials];*/

void ip_ccsd(tamm::Tensor *f, tamm::Tensor *v, tamm::Tensor *t_vo,
             tamm::Tensor *t_vvoo,
    // Fint *d_e, Fint *d_f1, Fint *d_v2, Fint *d_t1,
    // Fint *d_t2, Fint *k_e_offset, Fint *k_f1_offset,
    // Fint *k_v2_offset,   Fint *k_t1_offset, Fint *k_t2_offset,
             Fint *rtdb, Fint *size_x1, Fint *size_x2, Fint *k_irs,
             Fint nirreps, bool symmetry, std::string targetsym,
             Fint *maxtrials, int *nxtrials,
             double const &cpu, double const &wall) {
  Fint x1[*maxtrials];
  Fint x2[*maxtrials];

  // double cpu, wall;
  // double r1, r2;
  // double residual;
  // Fint irrep;            // Symmetry loop index
  Fint *l_hbar;
  Fint *k_hbar;
  Fint l_residual, k_residual;
  Fint *l_omegax;
  Fint *k_omegax;
  Fint *l_x1_offset;
  Fint *k_x1_offset;
  Fint *l_x2_offset;
  Fint *k_x2_offset;
  // Fint ivec, jvec;        // Current trial vector
  // Fint d_rx1;            // RHS residual file
  // Fint d_rx2;            // RHS residual file
  // double au2ev;    // Conversion factor from a.u. to eV

  // std::string filename;

  std::map<std::string, tamm::Tensor> tensors;

  // bool needt1 = true, needt2 = true;
  Fint dummy = 0;

  // Fint ip_unused_spin = 1;
  // Fint ip_unused_sym = 0;
  nodezero_print("\nIPCCSD calculation");

  // @todo eom_solver global variable fix
  // @todo ccsd_var global variable fix


  Fint ipol = 2;  // ipol will be passed on from Fortran
  static Fint nocc[2];  // will be passed on from Fortran
  Fint *int_mb = tamm::Variables::int_mb();
  /* Alternatively set as under
   * Variables:: set_idmb(int_mb, dbl_mb);
   */
//  Fint i, j;
  Irrep irrep_g = 0;          // Ground state symmetry
/*  if (ipol == 2) {
    for (int i = 1; i <= 2; i++) {
      for (int j = 1; i <= nocc[i]; j++) {
        irrep_g = irrep_g ^ int_mb[k_irs[i]+j-1];  // k_irs tce_main
      }
    }
  }*/
  Fint geom = 1;  // geom will be passed on from Fortran
  /* sym_irrepname is defined in src/symmetry/sym_irrepname.F
   */

  // sym_irrepname_(geom, irrep_g+1, irrepname); fortran function
  std::string irrepname = sym_irrepname(irrep_g);
  nodezero_print("\n Ground-state symmetry is " + irrepname);


  // bool symmetry = true;  // symmetry will be passed from Fortran
  // std::string targetsym;  // targetsym will be passed from Fortran
  for (Irrep irrep = 0; irrep <= nirreps - 1; irrep++) {
    // main irreps loop ===================
    // Irrep irrep_x = irrep;
    // Irrep irrep_y = irrep;
    std::string irrepname_iter = sym_irrepname(irrep ^ irrep_g);
    if ((!symmetry) || (targetsym == irrepname_iter)) {  // main
      tce_eom_init_();
      // if (util_print_("eom", print_default)) {
      nodezero_print("=========================================\n" +
                     "Excited-state calculation ( " + irrepname + " symmetry)" +
                     "==========================================\n");
      //}
      const int hbard = 4;
      double *hbar = new double[hbard][hbard];
      // if (!ma_push_get(mt_dbl,hbard*hbard,'hbar',
      //     l_hbar,k_hbar)) errquit('tce_eom_xdiagon: MA problem',0,
      //     MA_ERR)
      // tce_hbarinit_(dbl_mb(k_hbar),hbard);
      tce_hbarinit(hbar, hbard);
      // following block will use malloc instead of ma_push_get

      double *omegax = new double[*maxtrials];
      // if (!ma_push_get(mt_dbl,maxtrials,'omegax',l_omegax,k_omegax))
      // errquit('tce_energy: MA problem',1000,MA_ERR)

      tamm::RangeType list_d_rx1[] = {tamm::TO};
      tamm::Tensor *d_rx1 = new tamm::Tensor
          (1, 0, 0, list_d_rx1, tamm::dist_nw);  // tce_ipx1();
      d_rx1->create();
      // CorFortran(1, &tce_ipx1, tce_ipx1_offset_);
      // tce_ipx1_offset(l_x1_offset,k_x1_offset,size_x1)
      // tce_filename('rx1',filename)
      // createfile(filename,d_rx1,size_x1)

      tamm::RangeType list_d_rx2[] = {tamm::TN, tamm::TO, tamm::TO};
      tamm::Tensor *d_rx2 = new tamm::Tensor
          (3, 1, 0, list_d_rx2, tamm::dist_nw);  // tce_ipx2();
      d_rx2->create();
      // Tensor *d_rx2 = {nullptr};  // tce_ipx2();
      // CorFortran(1, &tce_ipx2, tce_ipx2_offset_);
      // tce_ipx2_offset(l_x2_offset,k_x2_offset,size_x2)
      // tce_filename('rx2',filename)
      // createfile(filename,d_rx2,size_x2)

      //         ------------------------------
      //         Generate initial trial vectors
      //         ------------------------------
      //
      // use fortran function for tce_eom_ipxguess
      // eom_ipxguess(x1, x2);
      double *p_evl_sorted;  // will be passed from Fortran
      double maxdiff;  // will be passed from Fortran
      int nroots;  // will be passed from Fortran
      double *rtdb_maxeorb;  // will be passed from Fortran
      int *offsets;  // will be passed from Fortran
      std::vector<tamm::Tensor *> x1_tensors;
      std::vector<tamm::Tensor *> x2_tensors;

      // tce_eom_ipxguess_cxx_
      //     (rtdb, size_x1, size_x2, k_x1_offset, k_x2_offset);
      tce_eom_ipxguess(p_evl_sorted, maxdiff,
                       nroots, rtdb_maxeorb, maxtrials, offsets,
                       x1_tensors, x2_tensors);

      //
      Expects(*nxtrials >= 0, "tce_ip_ccsdinitial space problems");

/*     Tensor *xc1;
      Tensor *xc2;
      Tensor *xp1;
      Tensor *xp;
      xc1 = new Tensor[nroots_reduced];
      xc2 = new Tensor[nroots_reduced];
      xp1 = new Tensor[nroots_reduced];
      xp2 = new Tensor[nroots_reduced];*/

      tamm::Tensor *xc1[nroots_reduced] = {nullptr};
      tamm::Tensor *xc2[nroots_reduced] = {nullptr};
      tamm::Tensor *xp1[nroots_reduced] = {nullptr};
      tamm::Tensor *xp2[nroots_reduced] = {nullptr};

      // make an indexed createfile equivalent function
      for (int ivec = 1; ivec <= nroots_reduced; ivec++) {
/*
      tce_filenameindexed_(ivec,'xc1',&filename); // in tce/tce_filename.F
      createfile_(filename,xc1(ivec),size_x1); // xc1 in tce/include/tce_diss.fh
      xc1_exist_(ivec) = true; // defined in tce_diis.fh
      tce_filenameindexed_(ivec,'xc2',&filename);
      createfile(filename,xc2(ivec),size_x2); // xc2 in tce/include/tce_diss.fh
      xc2_exist_(ivec) = true; // defined in tce_diis.fh
*/
        // create xc1 related objects
        // xc1[ivec-1] = new Tensor(ivec-1);
        xc1[ivec - 1] = tamm::Tensor::create();  // further define

        // create xc2 related objects
        // xc2[ivec-1] = new Tensor(ivec-1);
        xc2[ivec - 1] = tamm::Tensor::create();  // further define
      }

      bool converged = false;
      while (!converged) {  // loop to check for convergence
        for (Fint iter = 1; iter <= maxiter; iter++) {  // main loop
          if (util_print_("eom", print_default)) {
            nodezero_print("9210 " + std::to_string(iter) +
                           ", " + std::to_string(*nxtrials));
          }
          for (int ivec = 1; ivec <= *nxtrials; ivec++) {  // nxtrials loop
            if (!xp1[ivec - 1]) {  // uuu1
/*          tce_filenameindexed(ivec,'xp1',filename);
            createfile(filename,xp1(ivec),size_x1);
            xp1_exist(ivec) = true;
*/
              xp1[ivec - 1] = tamm::Tensor::create();  // further define
              ipccsd_x1_cxx_(d_f1, &xp1[ivec], d_t1, d_t2, d_v2, &x1[ivec],
                             x2[ivec], k_f1_offset, k_x2_offset, k_t1_offset,
                             k_t2_offset, k_v2_offset, k_x1_offset, k_x2_offset);
            }  // if xp1_exist(ivec)
            // reconcilefile(xp1(ivec),size_x1); not required here
            if (!xp2[ivec - 1]) {  // if xp2_exist
              // tce_filenameindexed(ivec,'xp2',filename);
              // createfile(filename,xp2(ivec),size_x2);
              // xp2_exist(ivec) = true;
              xp2[ivec - 1] = tamm::Tensor::create();

              ipccsd_x2_cxx_(d_f1, &x2[ivec], d_t1, d_t2, d_v2, &x1[ivec],
                             &x2[ivec], k_f1_offset, k_x2_offset, k_t1_offset,
                             k_t2_offset, k_v2_offset, k_x1_offset, k_x2_offset);
            }  // if xp2_exist(ivec)
            // reconcilefile(xp2(ivec),size_x2); not required here
          }  // nxtrials loop
          // if (!ma_push_get(mt_dbl,nxtrials,'residual',
          //     1      l_residual,k_residual))
          //     2      errquit('tce_energy: MA problem',101,MA_ERR)
          double dbl_k_omegax = static_cast <double>(k_omegax);
          double dbl_k_residual = static_cast <double>(k_residual);
          tce_eom_xdiagon_cxx_(size_x1, size_x2, k_x1_offset, k_x2_offset,
                               &d_rx1, &d_rx2, &dbl_k_omegax, &dbl_k_residual, k_hbar,
                               &iter);
/*        tce_eom_xdiagon_(needt1, needt2, false, false,
                 size_x1, size_x2, &dummy, &dummy,
                 k_x1_offset, k_x2_offset, &dummy, &dummy,
                 &d_rx1, &d_rx2, &dummy, &dummy,
                 &dbl_k_omegax, &dbl_k_residual, k_hbar, &iter,
                 false, true);*/

          cpu = cpu + util_cpusec_();
          wall = wall + util_wallsec_();
          converged = true;
          for (int ivec = 1; ivec <= nroots_reduced; ivec++) {
            if (ivec != nroots_reduced) {
              nodezero_print(
                  std::to_string(static_cast <double>(k_residual + ivec - 1))
                  + ", " + std::to_string(static_cast <double>(k_omegax + ivec - 1))
                  + ", "
                  + std::to_string(static_cast <double>((k_omegax + ivec - 1) * au2ev)));
            } else {
              nodezero_print(
                  std::to_string(static_cast <double>(k_residual + ivec - 1))
                  + ", " + std::to_string(static_cast <double>(k_omegax + ivec - 1))
                  + ", "
                  + std::to_string(static_cast <double>((k_omegax + ivec - 1) * au2ev))
                  + ", " + std::to_string(static_cast <double>(cpu))
                  + ", " + std::to_string(static_cast <double>(wall)));
            }
            nodezero_print("\n");  // check util_flush
            if (static_cast <double>(k_residual + ivec - 1) > thresh)
              converged = false;
          }  // nroots_reduced for loop
          cpu = -util_cpusec_();
          wall = -util_wallsec();
          delete[] residual;
          // if (!ma_pop_stack_(&l_residual))
          // errquit_("tce_energy: MA problem",102,MA_ERR);
          // nodezero_print("tce_energy: MA problem" + ", 102, " + MA_ERR);
          if (converged) {
            tce_eom_xtidy_();
            if (nodezero) {
              //  write(LuOut,*)'largest EOMCCSD amplitudes: R1 && R2'
              //     util_flush(LuOut)
              nodezero_print("largest EOMCCSD amplitudes: R1 && R2\n");
            }
            for (jvec = 1; jvec <= nroots_reduced; jvec++) {
              tce_print_ipx1_(&xc1[jvec - 1], k_x1_offset, DECI, irrep_x);
              tce_print_ipx2_(&xc2[jvec - 1], k_x2_offset, DECI, irrep_x);
              nodezero_print("\n");  // check util_flush
            }  // nroots_reduced for loop
          }  // converged
        }  // main loop
      }  // loop to check convergence
      for (int ivec = 1; ivec <= nroots_reduced; ivec++) {
        if (xc1[ivec - 1]) xc1[ivec - 1]->destroy();
        if (xc2[ivec - 1]) xc2[ivec - 1]->tamm::Tensor::destroy();
        if (xp1[ivec - 1]) xp1[ivec - 1]->tamm::Tensor::destroy();
        if (xp2[ivec - 1]) xp2[ivec - 1]->tamm::Tensor::destroy();
      }
      // delete l_x2_offset
      delete[] x2_offset;  // d_rx2
      // if (!ma_pop_stack_(l_x2_offset))
      // errquit_("tce_energy: IP_EA problem 1",36,MA_ERR);
      // nodezero_print("tce_energy: IP_EA problem 1" + ", 36, " + MA_ERR);
      // delete l_x1_offset
      delete[] x1_offset;  // d_rx1
      //if (!ma_pop_stack_(l_x1_offset))
      // errquit_("tce_energy: IP_EA problem 2",36,MA_ERR);
      //  nodezero_print("tce_energy: IP_EA problem 2" + ", 36, " + MA_ERR);
      // delete k_omegax
      delete[] omegax;
      //if (!ma_pop_stack_(l_omegax))
      // errquit("tce_energy: IP_EA problem 3",102,MA_ERR);
      //nodezero_print("tce_energy: IP_EA problem 3" + ", 102, " + MA_ERR);
      // delete hbard
      delete[] hbar;
      //if (!ma_pop_stack_(l_hbar))
      // errquit_('tce_eom_xdiagon: MA problem',12,MA_ERR);
      //nodezero_print("tce_eom_xdiagon: MA problem" + ", 12, " + MA_ERR);
    }  // end of ((!symmetry) || (targetsym == irrepname))
  }  // main irreps loop ===================
  //
  //
  //  9210 format(/,1x,'Iteration ',i3,' using ',i4,' trial vectors')
  //  9230 format(1x,f17.13,f18.13,f11.5,2f8.1)
  //  9250 format(1x,'Ground-state symmetry is ',A4)
  //  9251 format(1x,'Dim. of EOMCC iter. space ',2x,i6)
  //  9200 format(1x,'=========================================',/,
  //     1       1x,'Excited-state calculation ( ',A4,'symmetry)',/,
  //     2       1x,'=========================================')
  //
  //
  //        return
  //        end
}
