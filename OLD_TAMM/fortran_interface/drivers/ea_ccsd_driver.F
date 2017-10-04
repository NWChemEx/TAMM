      subroutine ea_ccsd_driver(d_e,d_f1,d_v2,d_t1,d_t2,
     1           k_e_offset,k_f1_offset,k_v2_offset,
     2           k_t1_offset,k_t2_offset,
     3           rtdb,eaccsd,ipccsd)
      implicit none
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
c
      integer rtdb ! runtime database
      integer d_f1,d_e,d_t1,d_t2,d_v2
      integer k_f1_offset,k_e_offset,k_t1_offset
      integer k_t2_offset,k_v2_offset
      double precision cpu, wall
      double precision r1,r2
      double precision residual
      integer i,j
      integer irrep_g          ! Ground state symmetry
      integer irrep            ! Symmetry loop index
      integer l_hbar,k_hbar
      integer l_residual,k_residual
      integer l_omegax,k_omegax
      integer l_x2_offset,k_x2_offset,size_x2
      integer l_x1_offset,k_x1_offset,size_x1
      integer ivec,jvec        ! Current trial vector
      integer d_rx1            ! RHS residual file
      integer d_rx2            ! RHS residual file
      integer dummy
      double precision au2ev    ! Conversion factor from a.u. to eV
      logical needt1,needt2
      logical ipccsd,eaccsd
      logical converged
      logical nodezero
      parameter (au2ev=27.2113961d0)
      character*4 irrepname
      character*255 filename
!
      nodezero=(ga_nodeid().eq.0)
!
        irrep_g = 0
        needt1=.true.
        needt2=.true.
        dummy=0
c
       ea_unused_spin=1  ! in tce.fh
       ea_unused_sym=0   ! in tce.fh
       if(nodezero) write(LuOut,*)
       if(nodezero) write(LuOut,*) "EACCSD calculation"
       if(nodezero) call util_flush(LuOut)
!
           if (eom_solver.eq.2) eom_solver=1
           if (ccsd_var.eq.'ic') ccsd_var='xx'
!
        if (ipol.eq.2) then
          do i = 1,2
            do j = 1,nocc(i)
             irrep_g = ieor(irrep_g,int_mb(k_irs(i)+j-1)) !k_irs tce_main 
            enddo
          enddo
        endif
        call sym_irrepname(geom,irrep_g+1,irrepname)
        if (nodezero.and.util_print('eom',print_default)) then
          write(LuOut,*)
          write(LuOut,9250) irrepname
        endif
        do irrep = 0, nirreps-1  ! main irreps loop ===================
          irrep_x = irrep  
          irrep_y = irrep  
          call sym_irrepname(geom,ieor(irrep_x,irrep_g)+1,irrepname)
          if ((.not.symmetry).or.(targetsym.eq.irrepname)) then !main
          call tce_eom_init
          if (nodezero.and.util_print('eom',print_default)) then
            write(LuOut,*)
            write(LuOut,9200) irrepname
          endif
!
       if (.not.ma_push_get(mt_dbl,hbard*hbard,'hbar',
     1  l_hbar,k_hbar)) call errquit('tce_eom_xdiagon: MA problem',0,
     2  MA_ERR)
       call tce_hbarinit(dbl_mb(k_hbar),hbard)
       if (.not.ma_push_get(mt_dbl,maxtrials,'omegax',
     1      l_omegax,k_omegax))
     2      call errquit('tce_energy: MA problem',1000,MA_ERR)
            call tce_eax1_offset(l_x1_offset,k_x1_offset,size_x1)
            call tce_filename('rx1',filename)
            call createfile(filename,d_rx1,size_x1)
             call tce_eax2_offset(l_x2_offset,k_x2_offset,size_x2)
            call tce_filename('rx2',filename)
            call createfile(filename,d_rx2,size_x2)
!
!         ------------------------------
!         Generate initial trial vectors
!         ------------------------------
!
            call tce_eom_eaxguess(rtdb,needt1,needt2,.false.,.false.,
     1        size_x1,size_x2,dummy,dummy,
     2        k_x1_offset,k_x2_offset,dummy,dummy)
!
           if (nxtrials.eq.0) 
     &      call errquit('tce_ip_ccsdinitial space problems',0,MA_ERR)
!
          do ivec=1,nroots_reduced
            call tce_filenameindexed(ivec,'xc1',filename)
            call createfile(filename,xc1(ivec),size_x1)
            xc1_exist(ivec) = .true.
            call tce_filenameindexed(ivec,'xc2',filename)
            call createfile(filename,xc2(ivec),size_x2)
            xc2_exist(ivec) = .true.
          enddo
!
            do iter=1,maxiter !main loop
              if (nodezero.and.util_print('eom',print_default))
     1          write(LuOut,9210) iter,nxtrials
              do ivec = 1,nxtrials !nxtrials loop
                if (.not.xp1_exist(ivec)) then ! uuu1
                  call tce_filenameindexed(ivec,'xp1',filename)
                  call createfile(filename,xp1(ivec),size_x1)
                  xp1_exist(ivec) = .true.
             call eaccsd_x1(d_f1,xp1(ivec),d_t1,d_t2,d_v2,x1(ivec),
     +                      x2(ivec),
     +                      k_f1_offset,k_x1_offset,k_t1_offset,
     +                      k_t2_offset,k_v2_offset,k_x1_offset,
     +                      k_x2_offset)
        endif ! if xp1_exist(ivec)
!
        call reconcilefile(xp1(ivec),size_x1)
!
        if (.not.xp2_exist(ivec)) then !if xp2_exist
         call tce_filenameindexed(ivec,'xp2',filename)
         call createfile(filename,xp2(ivec),size_x2)
         xp2_exist(ivec) = .true.
         call eaccsd_x2(d_f1,xp2(ivec),d_t1,d_t2,d_v2,x1(ivec),
     +                      x2(ivec),
     +                      k_f1_offset,k_x2_offset,k_t1_offset,
     +                      k_t2_offset,k_v2_offset,k_x1_offset,
     +                      k_x2_offset)
        endif ! if xp2_exist(ivec)
!
        call reconcilefile(xp2(ivec),size_x2)
!
        enddo ! nxtrials loop
!
        if (.not.ma_push_get(mt_dbl,nxtrials,'residual',
     1      l_residual,k_residual))
     2      call errquit('tce_energy: MA problem',101,MA_ERR)
!
        call tce_eom_xdiagon(needt1,needt2,.false.,.false.,
     1       size_x1,size_x2,dummy,dummy,
     2       k_x1_offset,k_x2_offset,dummy,dummy,
     3       d_rx1,d_rx2,dummy,dummy,
     4       dbl_mb(k_omegax),dbl_mb(k_residual),k_hbar,iter,
     +       eaccsd,ipccsd)
!
             cpu=cpu+util_cpusec()
             wall=wall+util_wallsec()
             converged = .true.
              do ivec = 1,nroots_reduced
                if (nodezero.and.(ivec.ne.nroots_reduced))
     1            write(LuOut,9230) dbl_mb(k_residual+ivec-1),
     2            dbl_mb(k_omegax+ivec-1),
     3            dbl_mb(k_omegax+ivec-1)*au2ev
                if (nodezero.and.(ivec.eq.nroots_reduced))
     1            write(LuOut,9230) dbl_mb(k_residual+ivec-1),
     2            dbl_mb(k_omegax+ivec-1),
     3            dbl_mb(k_omegax+ivec-1)*au2ev,cpu,wall
                if (nodezero) call util_flush(LuOut)
                if (dbl_mb(k_residual+ivec-1).gt.thresh)
     1            converged = .false.
              enddo
              cpu=-util_cpusec()
              wall=-util_wallsec()
              if (.not.ma_pop_stack(l_residual))
     1          call errquit("tce_energy: MA problem",102,MA_ERR)
!
          if (converged) then
            call tce_eom_xtidy
              if(nodezero) then
               write(LuOut,*)'largest EOMCCSD amplitudes: R1 and R2'
               call util_flush(LuOut)
              end if
             do jvec=1,nroots_reduced
               call tce_print_eax1(xc1(jvec),k_x1_offset,0.10d0,irrep_x)
               call tce_print_eax2(xc2(jvec),k_x2_offset,0.10d0,irrep_x)
               call util_flush(LuOut)
             enddo
             go to 200
           endif !converged
!
          enddo ! main loop
!
 200      continue
c delete l_x2_offset
            if (.not.ma_pop_stack(l_x2_offset))
     1        call errquit("tce_energy: IP_EA problem 1",36,MA_ERR)
c delete l_x1_offset
            if (.not.ma_pop_stack(l_x1_offset))
     1        call errquit("tce_energy: IP_EA problem 2",36,MA_ERR)
c delete k_omegax
          if (.not.ma_pop_stack(l_omegax))
     1      call errquit("tce_energy: IP_EA problem 3",102,MA_ERR)
c delete hbard
          if (.not.ma_pop_stack(l_hbar))
     1      call errquit('tce_eom_xdiagon: MA problem',12,MA_ERR)
!
         endif !no symm or targetsym
        enddo! main irreps loop ===================
!
!
 9210 format(/,1x,'Iteration ',i3,' using ',i4,' trial vectors')
 9230 format(1x,f17.13,f18.13,f11.5,2f8.1)
 9250 format(1x,'Ground-state symmetry is ',A4)
 9251 format(1x,'Dim. of EOMCC iter. space ',2x,i6)
 9200 format(1x,'=========================================',/,
     1       1x,'Excited-state calculation ( ',A4,'symmetry)',/,
     2       1x,'=========================================')
!
!
        return 
        end

















































