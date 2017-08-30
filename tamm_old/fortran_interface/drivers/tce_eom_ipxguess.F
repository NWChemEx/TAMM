      subroutine tce_eom_ipxguess(rtdb,
     1  size_x1,size_x2,k_x1_offset,k_x2_offset)
c
      implicit none
#include "stdio.fh"
#include "global.fh"
#include "mafdecls.fh"
#include "sym.fh"
#include "rtdb.fh"
#include "tce.fh"
#include "tce_main.fh"
#include "tce_diis.fh"
#include "errquit.fh"
#include "util.fh"
      integer rtdb
      integer size_x1
      integer size_x2
      integer k_x1_offset
      integer k_x2_offset
      integer p1b,p2b
      integer h3b,h4b
      integer p1,p2,h3,h4
      integer l_x1,k_x1
      integer l_diff,k_diff
      integer size
      integer ivec, jvec
      double precision nextmaxdiff,maxeorb
      integer i,ibumper
      logical defmeo
      logical nodezero
      character*255 filename
c
c     Determine threshold
c
      nodezero = (ga_nodeid().eq.0)
      ibumper=200000
      if (.not.ma_push_get(mt_dbl,ibumper,'diff',l_diff,k_diff))
     1  call errquit('tce_eom_xguess: MA problem',0,MA_ERR)
      call dfill(ibumper,1.0d99,dbl_mb(k_diff),1)
      ivec = 0
      do p1b = 1,1
        do h3b = 1,noab
          if (ip_unused_spin  .eq. int_mb(k_spin+h3b-1)) then
          if ((.not.restricted).or.(ip_unused_spin 
     1      +int_mb(k_spin+h3b-1).ne.4)) then
          if (ieor(ip_unused_sym ,int_mb(k_sym+h3b-1)).eq.
     1      irrep_x) then
            do h3 = 1,int_mb(k_range+h3b-1)
              ivec = ivec + 1
              dbl_mb(k_diff+ivec-1) = 
     2         -dbl_mb(k_evl_sorted+int_mb(k_offset+h3b-1)+h3-1)
            enddo
          endif
          endif
          endif
        enddo
      enddo
        nroots_reduced = min(ivec, nroots)
      maxdiff = 0.0d0
  100 continue
      nextmaxdiff = 1.0d99
      jvec = 0
      do i = 1,ivec
        if (dbl_mb(k_diff+i-1).lt.maxdiff) jvec = jvec + 1
        if (dbl_mb(k_diff+i-1).ge.maxdiff) then
          if (dbl_mb(k_diff+i-1).lt.nextmaxdiff)
     1      nextmaxdiff = dbl_mb(k_diff+i-1)
        endif
      enddo
      if (jvec.ge.nroots_reduced) goto 110
      maxdiff = nextmaxdiff + 0.001d0
      goto 100
  110 continue
      if (.not.ma_pop_stack(l_diff))
     1  call errquit('tce_eom_xguess: MA problem',1,MA_ERR)
c
       maxeorb=0.0d0
       defmeo=.false.
       if (.not.rtdb_get(rtdb,'tce:maxeorb',mt_dbl,1,maxeorb)) then
         defmeo=.false.
       else
         defmeo=.true.
         maxdiff=maxeorb 
       end if
c     Make trial X1/X2 based on single excitations
      nxtrials = 0
      do p1b = 1,1
        do h3b = 1,noab
          if (ip_unused_spin .eq. int_mb(k_spin+h3b-1)) then
          if ((.not.restricted).or.(ip_unused_spin 
     1      +int_mb(k_spin+h3b-1).ne.4)) then
          if (ieor(ip_unused_sym ,int_mb(k_sym+h3b-1)).eq.
     1      irrep_x) then
          size = 1 * int_mb(k_range+h3b-1)
          if (.not.ma_push_get(mt_dbl,size,'x1',l_x1,k_x1))
     1      call errquit('tce_eom_xguess: MA problem',2,MA_ERR)
          ivec = 0
          do p1 = 1,1
            do h3 = 1,int_mb(k_range+h3b-1)
              ivec = ivec + 1
              if ( 
     1           -dbl_mb(k_evl_sorted+int_mb(k_offset+h3b-1)+h3-1)
     2           .le.maxdiff) then
              nxtrials = nxtrials + 1
              call dfill(size,0.0d0,dbl_mb(k_x1),1)
              dbl_mb(k_x1+ivec-1) = 1.0d0
              if (nxtrials.gt.maxtrials) call errquit
     1          ('tce_eom_xguess: problem too large',nxtrials,
     2          CALC_ERR)
                call tce_filenameindexed(nxtrials,'x1',filename)
                call createfile(filename,x1(nxtrials),size_x1)
                call put_hash_block(x1(nxtrials),dbl_mb(k_x1),size,
     1            int_mb(k_x1_offset),((p1b-noab-1)*noab+h3b-1))
                x1_exist(nxtrials) = .true.
                call tce_filenameindexed(nxtrials,'x2',filename)
                call createfile(filename,x2(nxtrials),size_x2)
                x2_exist(nxtrials) = .true.
              endif
            enddo
          enddo
          if (.not.ma_pop_stack(l_x1))
     1      call errquit('tce_eom_xguess: MA problem',3,MA_ERR)
          endif
          endif
          endif
        enddo
      enddo
c
      if (nxtrials.lt.nroots_reduced)
     1  call errquit('there is a bug in the program',0,UNKNOWN_ERR)
      if (nodezero) then
        write(LuOut,9000) nxtrials
        if (nroots.gt.nroots_reduced) then
          write(LuOut,9010) nroots, nroots_reduced
        endif
      endif
      return
 9000 format(/,1x,'No. of initial right vectors ',i4)
 9010 format(  1x,'No. of roots reduced from    ',i4,' to ',i4)
      end
