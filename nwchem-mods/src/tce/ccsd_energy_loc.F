      subroutine ccsd_energy_loc(d_e,d_f1,d_v2,d_t1,d_t2,
     1           k_e_offset,k_f1_offset,k_v2_offset,
     2           k_t1_offset,k_t2_offset,
     3           size_t1,size_t2,ref,corr,rtdb)
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
      integer d_f1,d_e,d_t1,d_t2,d_v2,d_r1,d_r2
      integer k_f1_offset,k_e_offset,k_t1_offset
      integer k_t2_offset,k_v2_offset
      integer l_t1_local,k_t1_local
      integer d_c2
      integer size_t1,size_t2
      double precision ref,corr
      double precision cpu, wall
      double precision r1,r2
      double precision residual
      logical nodezero
      integer dummy
      integer unitn
      integer fldgts
      character*255 filename
c
      nodezero=(ga_nodeid().eq.0)
      if (skip_t) then
        if (nodezero) write(LuOut,9700)
        return
      endif
c
      call tce_diis_init()
c       if (.not.rtdb_get(rtdb,'tce:maxioprocs',mt_int,1,max_ioprocs_in)) 
c     4      max_ioprocs_in=4
c     number of digits for parallel file prefix (util_file_name0 last arg)
       if (.not.rtdb_get(rtdb,'tce:filedgts',mt_int,1,fldgts)) 
     A      fldgts=5
       if(max_ioprocs_in.gt.10**fldgts) then
          fldgts=log10(dble(max_ioprocs_in))+1
          if(ga_nodeid().eq.0) write(6,*)
     R         ' tce:filedgts redefined to ',fldgts
       endif
      do iter=1,maxiter
        cpu=-util_cpusec()
        wall=-util_wallsec()
        if (nodezero.and.(iter.eq.1)) then
          if(ccsd_var.ne.'ic') write(LuOut,9050) "CCSD"
          if(ccsd_var.eq.'ic') write(LuOut,9052) "CCSD"
        end if
        if (nodezero.and.(iter.eq.1)) then
         if(ccsd_var.eq.'ic') write(LuOut,9051)"NEW TASK SCHEDULING"
        end if
        if (.not.MA_PUSH_GET(mt_dbl,size_t1,'t1_local',
     1                       l_t1_local,k_t1_local))
     1      call errquit('ccsd_energy_loc: t1_local',1,MA_ERR)
        call ma_zero(dbl_mb(k_t1_local),size_t1)
cc        call ga_get(d_t1,1,size_t1,1,1,dbl_mb(k_t1_local),1)
        call get_block(d_t1,dbl_mb(k_t1_local),size_t1,0)
        call tce_filename('e',filename)
        call createfile(filename,d_e,1) ! size_e = 1
#if 0
        call ccsd_e(d_f1,d_e,k_t1_local,d_t2,d_v2,
     1              k_f1_offset,k_e_offset,
     2              k_t1_offset,k_t2_offset,k_v2_offset)
        call reconcilefile(d_e,1)
        call tce_filename('r1',filename)
        call createfile(filename,d_r1,size_t1)
        if(ccsd_var.eq.'cc2') then
c           call cc2_t1(d_f1,d_r1,d_t1,d_t2,d_v2,k_f1_offset,
c     &                 k_t1_offset,k_t1_offset,k_t2_offset,k_v2_offset)
           call  cc2_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,k_f1_offset,
     &       k_t1_offset,k_t1_offset,k_t2_offset,k_v2_offset)
        else
          if(ccsd_var.eq.'ic') then
cxxxxxxxxxxxxxxxxxxxx
          call icsd_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,
     1                 k_f1_offset,k_t1_offset,
     2                 k_t1_offset,k_t2_offset,k_v2_offset,iter)
          else
           call ccsd_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t1_offset,
     2                  k_t1_offset,k_t2_offset,k_v2_offset)
          end if
        end if
        call reconcilefile(d_r1,size_t1)
        call tce_filename('r2',filename)
        call createfile(filename,d_r2,size_t2)
        if(ccsd_var.eq.'cc2') then
c           call cc2_t2(d_f1,d_r2,d_t1,d_t2,d_v2,k_f1_offset,k_t2_offset,
c     &                 k_t1_offset,k_t2_offset,k_v2_offset,size_t2,
c     &                 k_t1_local)
         call cc2_t2(d_f1,d_r2,k_t1_local,
     &     d_t2,d_v2,k_f1_offset,k_t2_offset,
     &     k_t1_offset,k_t2_offset,k_v2_offset,size_t2)
        else
          if(ccsd_var.eq.'ic') then
cxxxxxxxxxxxxxxxxxxxxx
           call tce_filename('c2',filename)
           call createfile(filename,d_c2,size_t2)
           call copyfile(d_t2,d_c2,size_t2)
           call reconcilefile(d_c2,size_t2)
           call c2f_t2_t12(k_t1_local,k_t1_offset,d_c2,k_t2_offset)
           call reconcilefile(d_c2,size_t2)
           call icsd_t2(d_f1,d_r2,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t2_offset,
     2               k_t1_offset,k_t2_offset,k_v2_offset,
     3               size_t1,size_t2,d_c2,iter)
           call deletefile(d_c2)
           else
           call ccsd_t2(rtdb,d_f1,d_r2,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t2_offset,
     2                  k_t1_offset,k_t2_offset,k_v2_offset)
          end if
        end if
        call reconcilefile(d_r2,size_t2)
#else
        call tce_filename('r1',filename)
        call createfile(filename,d_r1,size_t1)
        call tce_filename('r2',filename)
        call createfile(filename,d_r2,size_t2)
        if(ccsd_var.eq.'cc2') then
        call ccsd_e(d_f1,d_e,k_t1_local,d_t2,d_v2,
     1              k_f1_offset,k_e_offset,
     2              k_t1_offset,k_t2_offset,k_v2_offset)
c           call cc2_t1(d_f1,d_r1,d_t1,d_t2,d_v2,k_f1_offset,
c     &                 k_t1_offset,k_t1_offset,k_t2_offset,k_v2_offset)
#if 0
           call  cc2_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,k_f1_offset,
     &       k_t1_offset,k_t1_offset,k_t2_offset,k_v2_offset)
c           call cc2_t2(d_f1,d_r2,d_t1,d_t2,d_v2,k_f1_offset,k_t2_offset,
c     &                 k_t1_offset,k_t2_offset,k_v2_offset,size_t2,
c     &                 k_t1_local)
         call cc2_t2(d_f1,d_r2,k_t1_local,
     &     d_t2,d_v2,k_f1_offset,k_t2_offset,
     &     k_t1_offset,k_t2_offset,k_v2_offset,size_t2)
#else
      call set_var_cxx(noab, nvab, int_mb, dbl_mb,
     &k_range, k_spin, k_sym, intorb, restricted,
     &irrep_v, irrep_t, irrep_f)
      call set_k_alpha_cxx(k_alpha)
           call  cc2_t1_cxx(d_f1,d_r1,k_t1_local,d_t2,d_v2,k_f1_offset,
     &       k_t1_offset,k_t1_offset,k_t2_offset,k_v2_offset)
         call cc2_t2_cxx(d_f1,d_r2,k_t1_local,
     &     d_t2,d_v2,k_f1_offset,k_t2_offset,
     &     k_t1_offset,k_t2_offset,k_v2_offset,size_t2)
#endif
        else
          if(ccsd_var.eq.'ic') then
cxxxxxxxxxxxxxxxxxxxx
           call tce_filename('c2',filename)
           call createfile(filename,d_c2,size_t2)
           call copyfile(d_t2,d_c2,size_t2)
           call reconcilefile(d_c2,size_t2)
           call c2f_t2_t12(k_t1_local,k_t1_offset,d_c2,k_t2_offset)
           call reconcilefile(d_c2,size_t2)
        call ccsd_e(d_f1,d_e,k_t1_local,d_t2,d_v2,
     1              k_f1_offset,k_e_offset,
     2              k_t1_offset,k_t2_offset,k_v2_offset)
#if 1
          call icsd_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,
     1                 k_f1_offset,k_t1_offset,
     2                 k_t1_offset,k_t2_offset,k_v2_offset,iter)
           call icsd_t2(d_f1,d_r2,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t2_offset,
     2               k_t1_offset,k_t2_offset,k_v2_offset,
     3               size_t1,size_t2,d_c2,iter)
#else
      call set_var_cxx(noab, nvab, int_mb, dbl_mb,
     &k_range, k_spin, k_sym, intorb, restricted,
     &irrep_v, irrep_t, irrep_f)
      call set_k_alpha_cxx(k_alpha)
          call icsd_t1_cxx(d_f1,d_r1,k_t1_local,d_t2,d_v2,
     1                 k_f1_offset,k_t1_offset,
     2                 k_t1_offset,k_t2_offset,k_v2_offset)
           call icsd_t2_cxx(d_f1,d_r2,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t2_offset,
     2               k_t1_offset,k_t2_offset,k_v2_offset,
     3               size_t1,size_t2,d_c2,iter)
#endif
           call deletefile(d_c2)
          else
#if 0
        call ccsd_e(d_f1,d_e,k_t1_local,d_t2,d_v2,
     1              k_f1_offset,k_e_offset,
     2              k_t1_offset,k_t2_offset,k_v2_offset)
           call ccsd_t1(d_f1,d_r1,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t1_offset,
     2                  k_t1_offset,k_t2_offset,k_v2_offset)
           call ccsd_t2(rtdb,d_f1,d_r2,k_t1_local,d_t2,d_v2,
     1                  k_f1_offset,k_t2_offset,
     2                  k_t1_offset,k_t2_offset,k_v2_offset)
#else
      call set_var_cxx(noab, nvab, int_mb, dbl_mb,
     &k_range, k_spin, k_sym, intorb, restricted,
     &irrep_v, irrep_t, irrep_f)
      call set_k_alpha_cxx(k_alpha)
c           call ccsd_t12_cxx(d_f1, d_v2, d_r1, d_r2, k_t1_local, d_t2,
c     1          k_f1_offset, k_v2_offset, k_t1_offset, k_t2_offset,
c     2          k_t1_offset, k_t2_offset)
           call ccsd_et12_cxx(d_e,d_f1, d_v2, d_r1, d_r2, k_t1_local,
     1     d_t2, k_e_offset,
     1          k_f1_offset, k_v2_offset, k_t1_offset, k_t2_offset,
     2          k_t1_offset, k_t2_offset)
#endif
          end if
        end if
        call reconcilefile(d_e,1)
        call reconcilefile(d_r1,size_t1)
        call reconcilefile(d_r2,size_t2)
#endif
        call tce_residual_t1(d_r1,k_t1_offset,r1)
        call tce_residual_t2(d_r2,k_t2_offset,r2)
        residual = max(r1,r2)
        call get_block(d_e,corr,1,0)
        if(.not.MA_POP_STACK(l_t1_local))
     &     call errquit('ccsd_energy_loc: l_t1_local',2,MA_ERR)
        cpu=cpu+util_cpusec()
        wall=wall+util_wallsec()
        if (nodezero) then
          if(ccsd_var.eq.'ic') then
          write(LuOut,9102) iter,residual,corr,cpu,wall
          else
          write(LuOut,9100) iter,residual,corr,cpu,wall,cpusecs(37)
c          write(LuOut,9101) cpusecs(37),cpusecs(38)
          end if
        endif
        if (residual .lt. thresh) then
          if (nodezero) then
            write(LuOut,9060)
            if(ccsd_var.eq.'cc2') then
               write(LuOut,9070) "CC2 ",corr
               write(LuOut,9080) "CC2 ",ref + corr
            else
               write(LuOut,9070) "CCSD",corr
               write(LuOut,9080) "CCSD",ref + corr
            end if
          endif
          call deletefile(d_r2)
          call deletefile(d_r1)
          call deletefile(d_e)
          if (ampnorms) then
            call tce_residual_t1(d_t1,k_t1_offset,r1)
            call tce_residual_t2(d_t2,k_t2_offset,r2)
            if (nodezero) then
              write(LuOut,9082) "T singles",r1
              write(LuOut,9082) "T doubles",r2
            endif
          endif
          call tce_print_x1(d_t1,k_t1_offset,printtol,irrep_t)
          call tce_print_x2(d_t2,k_t2_offset,printtol,irrep_t)
          call tce_diis_tidy()
          if (save_t(1)) then
            if(nodezero) then
              write(LuOut,*) 'Saving T1 now...'
            endif
            call x1_restart_save(d_t1,k_t1_offset,size_t1,0,
     1                           handle_t1,irrep_t)
          endif
          if (save_t(2)) then
            if(nodezero) then
              write(LuOut,*) 'Saving T2 now...'
            endif
            call x2_restart_save(d_t2,k_t2_offset,size_t2,0,
     1                           handle_t2,irrep_t)
          endif
          return
        endif
        if (save_t(1).and.(mod(iter,save_interval).eq.0)) then
          if(nodezero) then
            write(LuOut,*) 'Saving T1 now...'
          endif
          call x1_restart_save(d_t1,k_t1_offset,size_t1,0,
     1                         handle_t1,irrep_t)
        endif
        if (save_t(2).and.(mod(iter,save_interval).eq.0)) then
          if(nodezero) then
            write(LuOut,*) 'Saving T2 now...'
          endif
          call x2_restart_save(d_t2,k_t2_offset,size_t2,0,
     1                         handle_t2,irrep_t)
        endif
        call tce_diis(.false.,iter,.true.,.true.,.false.,.false.,
     1                d_r1,d_t1,k_t1_offset,size_t1,
     2                d_r2,d_t2,k_t2_offset,size_t2,
     3                dummy,dummy,dummy,dummy,
     4                dummy,dummy,dummy,dummy)
        call deletefile(d_r2)
        call deletefile(d_r1)
        call deletefile(d_e)
        if (nodezero) call util_flush(LuOut)
      enddo
c
          call util_file_name0('t1amp',.false.,.true.,filename,fldgts)
          unitn=79
          call write_tensor(filename,d_t1,size_t1,unitn)
          call util_file_name0('t2amp',.false.,.true.,filename,fldgts)
          unitn=80
          call write_tensor(filename,d_t2,size_t2,unitn)
          call ga_sync()
c
      call errquit('ccsd_energy_loc: maxiter exceeded',iter,CALC_ERR)
      return
 9020 format(1x,'Cpu & wall time / sec',2f15.1)
 9050 format(/,1x,A,' iterations',/,
     1  1x,'----------------------------',
     2     '-------------------------------------',/
     3  1x,'Iter          Residuum      ',
     4     ' Correlation     Cpu    Wall    V2*C2',/
     5  1x,'----------------------------',
     6     '-------------------------------------')
 9051 format(A)
 9052 format(A,' iterations',/,
     1  1x,'----------------------------',
     2     '-----------------------------',/
     3  1x,'Iter          Residuum      ',
     4     ' Correlation     Cpu    Wall ',/
     5  1x,'----------------------------',
     6     '-----------------------------')
 9060 format(
     1  1x,'----------------------------',
     2     '-------------------------------------',/
     3  1x,'Iterations converged')
 9070 format(1x,A,' correlation energy / hartree = ',f25.15)
 9080 format(1x,A,' total energy / hartree       = ',f25.15)
 9082 format(1x,'amplitude norm of ',A9,' = ',f25.15)
 9100 format(1x,i4,2f18.13,3f8.1)
 9101 format(11x,'ccsd_t2_8 ',20x,2f8.1)
 9102 format(1x,i4,2f18.13,2f8.1)
 9700 format(1x,'Skipping CCSD altogether')
      end

c $Id: ccsd_energy_loc.F 28319 2016-07-01 01:08:54Z edo $
!
      SUBROUTINE name_and_create(d_a, size_a)
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
      INTEGER d_a, size_a
      CHARACTER*255 filename

      CALL TCE_FILENAME('notensor',filename)
      CALL CREATEFILE(filename,d_a,size_a)
      end

      SUBROUTINE destroy(d_a, l_a)
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
#include "util.fh"
#include "errquit.fh"
#include "tce.fh"
      INTEGER d_a, l_a

      CALL DELETEFILE(d_a)
      IF (.not.MA_POP_STACK(l_a))
     1     CALL ERRQUIT('ccsd_t1',-1,MA_ERR)
      END
