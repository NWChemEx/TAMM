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
#include <string>
#include <iostream>
#include <cassert>
#include "tensor/gmem.h"
#include "tensor/variables.h"

#define maxtrials 20  // argument to be passed from Fortran
#define hbard 4  // argument to be passed from Fortran
#define size_x1  4  // argument to be passed from Fortran
#define size_x1  4  // argument to be passed from Fortran

/* Arguments that need to be passed to this function
 * maxtrials
 * hbard
 * size_x1
 * size_x2
 * eom_solver - from tce/include/tce_diis.fh, assigned in tce_energy.F
 * ccsd_var - from tce/include/tce_main.fh, defined in tce_input.F and passed on through tce_energy.F
 * ipol - from tce/include/tce_main.fh
 * nocc - from tce/include/tce_main.fh, integer nocc(2)
 * int_mb will be set through set_var_cxx_
 * geom - from tce/include/tce_main.fh
 * symmetry - from tce/include/tce_main.fh, logical symmetry
 * targetsym - from tce/include/tce_main.fh, character*4 targetsym
 *

 *
 */

/* xc1 is converged RHS1 vector from file tce/include/tce_diss.fh
 * xc2 is converged RHS2 vector from file tce/include/tce_diss.fh
 * xp1 is product RHS1 vector from file tce/include/tce_diss.fh
 * xp2 is product RHS2 vector from file tce/include/tce_diss.fh
 */

using tamm::gmem;
int ga_nodeid() { return gmem::rank();}

static const double au2ev = 27.2113961;

std::ostream nodezero_print(const std::string& str,
  std::ostream &os = std::cout) {
    if (ga_nodeid() == 0) {
      os << str << std::endl;
  }
  return os;
}

void Expects(bool cond, const string& msg) {
  if(!cond) {
    std::cerr<<msg<<endl;
    assert(cond);
  }
}

void tce_filename_cxx_(Fint index, const string& xc_count) {


}

void indexed_tensor_create(Fint index, ) {
  Fint k_a, l_a, size, d_a;
  fn(&l_a, &k_a, &size);
  fname_and_create(&d_a, &size);
  tensor->attach(k_a, l_a, d_a);
}

void tce_ipx1_offset_(F77Integer *l_x1_offset, F77Integer *k_x1_offset,
                         F77Integer *size_x1);

void tce_ipx2_offset_(F77Integer *l_x2_offset, F77Integer *k_x2_offset,
                         F77Integer *size_x2);

class Tensor;
using Irrep = int;

void ip_ccsd_driver_cxx_(Tensor& d_e, Tensor& d_f,
					Tensor& tv2, Tensor& d_1, Tensor& d_t2,
					RTDB rtdb) {

  static Fint xc1[maxtrials];
  static Fint xc2[maxtrials];
  static Fint xp1[maxtrials];
  static Fint xp2[maxtrials];

  double cpu, wall;
  double r1,r2;
  double residual;
  Fint irrep;            // Symmetry loop index
  Fint l_hbar,k_hbar;
  Fint l_residual,k_residual;
  Fint l_omegax,k_omegax;
  Fint l_x2_offset,k_x2_offset,size_x2;
  Fint l_x1_offset,k_x1_offset,size_x1;
  Fint ivec,jvec;        // Current trial vector
  Fint d_rx1;            // RHS residual file
  Fint d_rx2;            // RHS residual file
  Fint dummy;
  double au2ev;    // Conversion factor from a.u. to eV

  bool ipccsd,eaccsd;
  bool converged;
  bool nodezero;

  string filename;

  Tensor *tce_ipx1 = &tensors["tce_ipx1"];
  Tensor *tce_ipx2 = &tensors["tce_ipx2"];

  bool needt1,needt2;
  needt1=true;
  needt2=true;
  dummy=0;

  ip_unused_spin=1;
  ip_unused_sym=0 ;
  nodezero_print("\nIPCCSD calculation");

  Fint eom_solver = 2;  // eom_solver will be passed on from Fortran
  if (eom_solver == 2) {
    eom_solver = 1;
  }

  string ccsd_var = 'ic';  // ccsd_var will be passed on from Fortran
  if (ccsd_var == 'ic') {
    ccsd_var = 'xx';
  }

  Irrep irrep_g = 0;          // Ground state symmetry
  Fint ipol = 2;  // ipol will be passed on from Fortran
  static Fint nocc[2];  // will be passed on from Fortran
  Fint
  Fint *int_mb = Variables::int_mb();
  /* Alternatively set as under
   * Variables:: set_idmb(int_mb, dbl_mb);
   */
  Fint i, j;
  if (ipol == 2) {
    for (i = 1; i <= 2; i++) {
      for (j = 1; i <= nocc[i]; j++) {
        irrep_g = irrep_g ^ int_mb(k_irs(i)+j-1);  // k_irs tce_main
      }
    }
  }
  Fint geom = 1;  // geom will be passed on from Fortran
  /* sym_irrepname is defined in src/symmetry/sym_irrepname.F
   */
  string irrepname;
  sym_irrepname_(geom, irrep_g+1, irrepname);

  if (util_print_('eom',print_default)) { //print_default = print_medium = 20
    nodezero_print("\n" + std::to_string(irrepname));
  }
  bool symmetry = true;  // symmetry will be passed from Fortran
  string targetsym;  // targetsym will be passed from Fortran
  for (Irrep irrep = 0; irrep <= nirreps-1; irrep++) {  // main irreps loop ===================
    irrep_x = irrep;
    irrep_y = irrep;
    sym_irrepname_(geom, (irrep_x ^ irrep_g)+1, irrepname);
    if ((!symmetry) || (targetsym == irrepname)) {  // main
      tce_eom_init();
	  if (util_print('eom',print_default)) {
	    nodezero_print(
	    "=========================================\n"
	    "Excited-state calculation ( "+irrepname+" symmetry)==\n");
	  }
      //
      double *hbar = new double [hbard*hbard];
      // if (!ma_push_get(mt_dbl,hbard*hbard,'hbar',
      // 		       1  l_hbar,k_hbar)) errquit('tce_eom_xdiagon: MA problem',0,
      // 						  2  MA_ERR)
      // tce_hbarinit_(dbl_mb(k_hbar),hbard);
      tce_hbarinit_(hbar,hbard);
      //following block will use malloc instead of ma_push_get
      {
	omegax = new double [maxtrials];
	//	if (!ma_push_get(mt_dbl,maxtrials,'omegax',l_omegax,k_omegax))
	//  errquit('tce_energy: MA problem',1000,MA_ERR)

/* We use the code in equations.cc line 135 to create new tensors
 * for now using CorFortran function
 */
	Tensor tce_ipx1();
  //CorFortran(1, &tce_ipx1, tce_ipx1_offset_);
  // tce_ipx1_offset(l_x1_offset,k_x1_offset,size_x1)
  // tce_filename('rx1',filename)
  // createfile(filename,d_rx1,size_x1)

	Tensor tce_ipx2();
  //CorFortran(1, &tce_ipx2, tce_ipx2_offset_);
  // tce_ipx2_offset(l_x2_offset,k_x2_offset,size_x2)
  // tce_filename('rx2',filename)
  // createfile(filename,d_rx2,size_x2)
	}

  //         ------------------------------
  //         Generate initial trial vectors
  //         ------------------------------
  //
  // use fortran function for tce_eom_ipxguess 
  tce_eom_ipxguess_(rtdb,true,true,false,false,
		size_x1,size_x2,dummy,dummy,
		k_x1_offset,k_x2_offset,dummy,dummy);
  //
  Expects(nxtrials >= 0, "tce_ip_ccsdinitial space problems");
  //

  Tensor *xc1_array;
  Tensor *xc2_array;
  Tensor *xp1_array;
  Tensor *xp_array;

  xc1_array = new Tensor[nroots_reduced];
  xc2_array = new Tensor[nroots_reduced];
  xp1_array = new Tensor[nroots_reduced];
  xp2_array = new Tensor[nroots_reduced];

  const bool xc1_exist[nroots_reduced];
  const bool xc2_exist[nroots_reduced];
  const bool xp1_exist[nroots_reduced];
  const bool xp2_exist[nroots_reduced];

  // make an indexed createfile equivalent function
  for(int ivec=1; ivec<=nroots_reduced; ivec++) {
#if 0
	tce_filenameindexed_(ivec,'xc1',&filename); // in tce/tce_filename.F
  	createfile_(filename,xc1(ivec),size_x1); // xc1 in tce/include/tce_diss.fh
  	xc1_exist_(ivec) = true; // defined in tce_diis.fh
  	tce_filenameindexed_(ivec,'xc2',&filename);
 	  createfile(filename,xc2(ivec),size_x2); // xc2 in tce/include/tce_diss.fh
  	xc2_exist_(ivec) = true; // defined in tce_diis.fh
#else
  	// create xc1 related objects
  	// xc1_array[ivec-1] = new Tensor(ivec-1);
  	xc1_array[ivec-1] -> create();
  	xc1_exist[ivec-1] = true;

  	// create xc2 related objects
  	// xc2_array[ivec-1] = new Tensor(ivec-1);
  	xc2_array[ivec-1] -> create();
  	xc2_exist[ivec-1] = true;
#endif
  }

  converged = false;
  while(!converged) { //loop to check for convergence
	  for (int iter=1; iter<=maxiter; iter++) { //main loop
	    if (util_print('eom',print_default)) {
	      nodezero_print("9210 " + iter,nxtrials);
	    }
	    for (int ivec = 1; ivec<=nxtrials; ivec++) { //nxtrials loop
	      if (!xp1_exist[ivec-1]) { // uuu1
#if 0
	        tce_filenameindexed(ivec,'xp1',filename);
		    createfile(filename,xp1(ivec),size_x1);
		    xp1_exist(ivec) = true;
#else
		    xp1_array[ivec-1] -> create();
	        xp1_exist[ivec-1] = true;
#endif
	        ipccsd_x1_cxx(d_f1,xp1(ivec),d_t1,d_t2,d_v2,x1(ivec),
	  	  		  x2(ivec));
//	  	ipccsd_x1_cxx_(d_f1,xp1(ivec),d_t1,d_t2,d_v2,x1(ivec),
//	  		  x2(ivec),
//	  		  k_f1_offset,k_x1_offset,k_t1_offset,
//	  		  k_t2_offset,k_v2_offset,k_x1_offset,
//	  		  k_x2_offset);
	      } // if xp1_exist(ivec)
	      //
	      //reconcilefile(xp1(ivec),size_x1);
	        // 
	        if (!xp2_exist[ivec-1]) // following block will use c++ utilities
	  	{ //if xp2_exist
#if 0
	  	  tce_filenameindexed(ivec,'xp2',filename);
	  	  createfile(filename,xp2(ivec),size_x2);
	  	  xp2_exist(ivec) = true;
#else
		  xp2_array[ivec-1] -> create();
	      xp2_exist[ivec-1] = true;
#endif
	  	    ipccsd_x2_cxx_(d_f1,xp2(ivec),d_t1,d_t2,d_v2,x1(ivec),
	  		      x2(ivec),
	  		      k_f1_offset,k_x2_offset,k_t1_offset,
	  		      k_t2_offset,k_v2_offset,k_x1_offset,
	  		      k_x2_offset);
	  	    } // if xp2_exist(ivec)
	      //
	      //reconcilefile(xp2(ivec),size_x2) // use c++ utilities of of file creation
	        //
	        } // nxtrials loop
	    //
	    if (!ma_push_get(mt_dbl,nxtrials,'residual',
	  		   1      l_residual,k_residual))
	      2      errquit('tce_energy: MA problem',101,MA_ERR)
	        //
	        tce_eom_xdiagon_(needt1,needt2,false,false,
	  		       size_x1,size_x2,dummy,dummy,
	  		       k_x1_offset,k_x2_offset,dummy,dummy,
	  		       d_rx1,d_rx2,dummy,dummy,
	  		       dbl_mb(k_omegax),dbl_mb(k_residual),k_hbar,iter,
	  		       eaccsd,ipccsd); 
	    //
	    cpu=cpu+util_cpusec();
	    wall=wall+util_wallsec();
	    converged = true;
	    for( ivec = 1; ivec<=nroots_reduced; ivec++) {
	      if (nodezero && (ivec.ne.nroots_reduced))
	        1            write(LuOut,9230) dbl_mb(k_residual+ivec-1),
	  	2            dbl_mb(k_omegax+ivec-1),
	  	3            dbl_mb(k_omegax+ivec-1)*au2ev
	  	if (nodezero && (ivec == nroots_reduced))
	  	  1            write(LuOut,9230) dbl_mb(k_residual+ivec-1),
	  	    2            dbl_mb(k_omegax+ivec-1),
	  	    3            dbl_mb(k_omegax+ivec-1)*au2ev,cpu,wall
	  	    if (nodezero) util_flush_(LuOut); 
	      if (dbl_mb(k_residual+ivec-1) > thresh)
	        1            converged = false; 
	    }
	    cpu=-util_cpusec(); 
	    wall=-util_wallsec();
	    if (!ma_pop_stack(l_residual))
	      errquit_("tce_energy: MA problem",102,MA_ERR);
	    //
	    if (converged) {
	      tce_eom_xtidy
	        if(nodezero) {
	  	write(LuOut,*)'largest EOMCCSD amplitudes: R1 && R2'
	  	  util_flush(LuOut)
	  	  } 
	      for(jvec=1; jvec<=nroots_reduced; jvec++) {
	        tce_print_ipx1(xc1(jvec),k_x1_offset,0.10d0,irrep_x)
	  	tce_print_ipx2_(xc2(jvec),k_x2_offset,0.10d0,irrep_x)
	  	util_flush(LuOut)
	  	}
	      // go to 200 - not required
	    } //converged
	    //
	  } // main loop
	//
  } // loop to check convergence now replaces - 200      continue
  for(ivec=1; ivec<=nroots_reduced; ivec++) {
	if (xc1_exist[ivec-1]) xc1_array[ivec-1] -> destroy();
	if (xc2_exist[ivec-1]) xc2_array[ivec-1] -> destroy();
	if (xp1_exist[ivec-1]) xp1_array[ivec-1] -> destroy();
	if (xp2_exist[ivec-1]) xp2_array[ivec-1] -> destroy();
	}
  }
      // delete l_x2_offset
  if (!ma_pop_stack(l_x2_offset))
	// errquit_("tce_energy: IP_EA problem 1",36,MA_ERR);
    std::cerr<<"tce_energy: IP_EA problem 1"<<"36"<<"MA_ERR"<<std::endl;
      // delete l_x1_offset
  if (!ma_pop_stack(l_x1_offset))
	// errquit_("tce_energy: IP_EA problem 2",36,MA_ERR);
    std::cerr<<"tce_energy: IP_EA problem 2"<<"36"<<"MA_ERR"<<std::endl;
      // delete k_omegax
  if (!ma_pop_stack(l_omegax))
	// errquit("tce_energy: IP_EA problem 3",102,MA_ERR);
    std::cerr<<"tce_energy: IP_EA problem 3"<<"102"<<"MA_ERR"<<std::endl;
      // delete hbard
  if (!ma_pop_stack(l_hbar))
	// errquit_('tce_eom_xdiagon: MA problem',12,MA_ERR);
    std::cerr<<"tce_eom_xdiagon: MA problem"<<"12"<<"MA_ERR"<<std::endl;
      //
    } //no symm or targetsym
  }// main irreps loop ===================
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
