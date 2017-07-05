
      program test

      end

      subroutine init_fortran_vars(noa1, nob1, nva1, nvb1,
     1     intorb1, restricted1,
     1     spins, syms, ranges)
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "errquit.fh"
#include "tce.fh"

      integer noa1, nob1, nva1, nvb1
      logical intorb1, restricted1
      integer spins(noa1+nob1+nva1+nvb1)
      integer syms(noa1+nob1+nva1+nvb1)
      integer ranges(noa1+nob1+nva1+nvb1)
      integer i, j
      
      noa = noa1
      nva = nva1
      nob = nob1
      nvb = nvb1
      noab = noa + nob
      nvab = nva + nvb
      intorb = intorb1
      restricted = restricted1

      if(intorb)
     1     call errquit('not tested with intorb', 1,1)
      d_v2orb = 0               !fixme should be set correctly for intorb case
      k_v2spin_offset = 0       !fixme should be set correctly for intorb case
      k_v2_alpha_offset = 0     !fixme should be set correctly for intorb case
      
      if (.not.ma_push_get(mt_int,noa+nob+nva+nvb,"spins ",
     1     l_spin,k_spin)) call errquit("tce_tile: MA problem",9,
     2     MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nob+nva+nvb,"syms ",
     1     l_sym,k_sym)) call errquit("tce_tile: MA problem",9,
     2     MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nob+nva+nvb,"ranges ",
     1     l_range,k_range)) call errquit("tce_tile: MA problem",9,
     2     MA_ERR)
      
      do i = 1, noab+nvab
         int_mb(k_spin+i-1) = spins(i)
         int_mb(k_sym+i-1) = syms(i)
         int_mb(k_range+i-1) = ranges(i)
      enddo

      if(intorb) THEN
      if (.not.ma_push_get(mt_int,noa+nob+nva+nvb,"alpha-izer 2 ",
     1  l_b2am,k_b2am)) call errquit("tce_tile: MA problem",9,
     2  MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nob+nva+nvb,"alpha-izer",
     1  l_alpha,k_alpha)) call errquit("tce_tile: MA problem",8,
     2  MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nva,"test for alpha spins ",
     1  l_spin_alpha,k_spin_alpha)) 
     2  call errquit("tce_tile: MA problem",9,MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nva,"test for alpha syms ",
     1  l_sym_alpha,k_sym_alpha))
     2  call errquit("tce_tile: MA problem",9,MA_ERR) 
      if (.not.ma_push_get(mt_int,noa+nva,"test for alpha range ",
     1  l_range_alpha,k_range_alpha))
     2  call errquit("tce_tile: MA problem",9,MA_ERR)
      if (.not.ma_push_get(mt_int,noa+nva,"test for alpha range ",
     1  l_offset_alpha,k_offset_alpha))
     2  call errquit("tce_tile: MA problem",9,MA_ERR)
      endif

      if (restricted) then
        do i = 1,noa
          int_mb(k_alpha+i-1) = i
        enddo
        do i = noa+1,noa+nob
          int_mb(k_alpha+i-1) = i - noa
        enddo
        do i = noa+nob+1,noa+nob+nva
          int_mb(k_alpha+i-1) = i
        enddo
        do i = noa+nob+nva+1,noa+nob+nva+nvb
          int_mb(k_alpha+i-1) = i - nva
        enddo
      else
        do i = 1,noa+nob+nva+nvb
          int_mb(k_alpha+i-1) = i
        enddo
      endif

      if (intorb) then
         j = 0
         ! @bug @fixme SK: check the following logic with Karol
         do i = 1,noa+nva
            int_mb(k_spin_alpha+i-1) = int_mb(k_spin+i-1)
            int_mb(k_sym_alpha+i-1) = int_mb(k_sym+i-1)
            int_mb(k_range_alpha+i-1) = int_mb(k_range+i-1)
            int_mb(k_offset_alpha+i-1) = j
            j = j + int_mb(k_range+i-1)
         enddo
         ! @bug @fixme Implement b2am logic
c$$$
c$$$      do i = 1,noa+nob+nva+nvb
c$$$         int_mb(k_b2am+i-1) = b2am(i)
c$$$      enddo
      endif
      
      ea_unused_spin = 1
      ea_unused_sym = 0
      ip_unused_spin = 1
      ip_unused_sym = 0

      return
      END

      subroutine finalize_fortran_vars()
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "errquit.fh"
#include "tce.fh"

      if(intorb) THEN
      if (.not.ma_pop_stack(l_offset_alpha))
     1     call errquit("MA problem: finalize_fortran_vars",8,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_range_alpha))
     1     call errquit("MA problem: finalize_fortran_vars",7,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_sym_alpha))
     1     call errquit("MA problem: finalize_fortran_vars",6,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_spin_alpha))
     1     call errquit("MA problem: finalize_fortran_vars",5,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_alpha))
     1     call errquit("MA problem: finalize_fortran_vars",4,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_b2am))
     1     call errquit("MA problem: finalize_fortran_vars",3,
     2     MA_ERR)
      endif

      if (.not.ma_pop_stack(l_range))
     1     call errquit("MA problem: finalize_fortran_vars",2,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_sym))
     1     call errquit("MA problem: finalize_fortran_vars",1,
     2     MA_ERR)
      if (.not.ma_pop_stack(l_spin))
     1     call errquit("MA problem: finalize_fortran_vars",0,
     2     MA_ERR)
      
      return
      end


      subroutine dfill(n,val,a,ia)
C$Id: dfill.f 26932 2015-03-09 23:07:47Z d3y133 $
      implicit none
      integer n, ia
      double precision val, a(*)
      integer i
c
c     initialise double precision array to scalar value
c
         call dcopy(n,val,0,a,ia)
c
      end

      SUBROUTINE ccsd_t1_1(d_a,k_a_offset,irrepa,
     1     d_c,k_c_offset,irrepc)
C     $Id: ccsd_t1.F 23632 2013-02-25 23:01:51Z kowalski $
C     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
C     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
C     i0 ( p2 h1 )_f + = 1 * f ( p2 h1 )_f
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
#include "errquit.fh"
#include "tce.fh"
      INTEGER d_a, d_c
      INTEGER k_a_offset, k_c_offset
      integer irrepa, irrepc
      INTEGER NXTASK, next, nprocs, count
      INTEGER p2b, h1b, p2b_1, h1b_1
      INTEGER dim_common, dima_sort, dima, dimc
      INTEGER k_a, l_a
      EXTERNAL NXTASK
      nprocs = GA_NNODES()
      count = 0
      next = NXTASK(nprocs, 1)
      DO p2b = noab+1,noab+nvab
       DO h1b = 1,noab
         IF ((.not.restricted).or.(int_mb(k_spin+p2b-1)
     1                            +int_mb(k_spin+h1b-1).ne.4)) THEN
          IF (int_mb(k_spin+p2b-1) .eq. int_mb(k_spin+h1b-1)) THEN
           IF (ieor(int_mb(k_sym+p2b-1),int_mb(k_sym+h1b-1)) 
     1         .eq. irrepa) THEN
        IF (next.eq.count) THEN
            dimc = int_mb(k_range+p2b-1) * int_mb(k_range+h1b-1)
            CALL TCE_RESTRICTED_2(p2b,h1b,p2b_1,h1b_1)
            dim_common = 1
            dima_sort = int_mb(k_range+p2b-1) * int_mb(k_range+h1b-1)
            dima = dim_common * dima_sort
            IF (dima .gt. 0) THEN
             IF (.not.MA_PUSH_GET(mt_dbl,dima,'a',l_a,k_a)) 
     1            CALL ERRQUIT('ccsd_t1_1',1,MA_ERR)
             CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,
     1            int_mb(k_a_offset),
     2            (h1b_1 - 1 + (noab+nvab) * (p2b_1 - 1)))
             CALL ADD_HASH_BLOCK(d_c,dbl_mb(k_a),dimc,
     1            int_mb(k_c_offset),
     2            (h1b - 1 + noab * (p2b - noab - 1)))
             IF (.not.MA_POP_STACK(l_a)) 
     1            CALL ERRQUIT('ccsd_t1_1',5,MA_ERR)
            END IF
         next = NXTASK(nprocs, 1)
        END IF
        count = count + 1
           END IF
          END IF
         END IF
       END DO
      END DO
      next = NXTASK(-nprocs, 1)
      call GA_SYNC()
      RETURN
      END

      SUBROUTINE ccsd_t1_2(d_a,k_a_offset,irrepa,
     1     d_b,k_b_offset,irrepb,
     1     d_c,k_c_offset,irrepc)
C     $Id: ccsd_t1.F 23632 2013-02-25 23:01:51Z kowalski $
C     This is a Fortran77 program generated by Tensor Contraction Engine v.1.0
C     Copyright (c) Battelle & Pacific Northwest National Laboratory (2002)
C     i0 ( p2 h1 )_tf + = -1 * Sum ( h7 ) * t ( p2 h7 )_t * i1 ( h7 h1 )_f
      IMPLICIT NONE
#include "global.fh"
#include "mafdecls.fh"
c#include "sym.fh"
#include "errquit.fh"
#include "tce.fh"
      INTEGER d_a,d_b,d_c
      INTEGER k_a_offset,k_b_offset,k_c_offset
      integer irrepa, irrepb, irrepc
      INTEGER NXTASK,next,nprocs,count
      INTEGER p2b,h1b,h7b,p2b_1,h7b_1,h7b_2,h1b_2
      INTEGER dim_common,dima_sort,dimb_sort
      INTEGER dima,dimb,dimc
      INTEGER k_a,l_a,k_b,l_b,k_c,l_c
      INTEGER k_bs,l_bs,k_cs,l_cs
      EXTERNAL NXTASK
      nprocs = GA_NNODES()
      count = 0
      next = NXTASK(nprocs, 1)
      DO p2b = noab+1,noab+nvab
       DO h1b = 1,noab
ccx        IF (next.eq.count) THEN
         IF ((.not.restricted).or.(int_mb(k_spin+p2b-1)
     1                            +int_mb(k_spin+h1b-1).ne.4)) THEN
          IF (int_mb(k_spin+p2b-1) .eq. int_mb(k_spin+h1b-1)) THEN
           IF (ieor(int_mb(k_sym+p2b-1),int_mb(k_sym+h1b-1)).eq.
     1         ieor(irrepa,irrepb)) THEN
        IF (next.eq.count) THEN
            dimc = int_mb(k_range+p2b-1) * int_mb(k_range+h1b-1)
            IF (.not.MA_PUSH_GET(mt_dbl,dimc,'noname',l_cs,k_cs))
     1           CALL ERRQUIT('ccsd_t1_2',0,MA_ERR)
            CALL DFILL(dimc,0.0d0,dbl_mb(k_cs),1)
            DO h7b = 1,noab
             IF (int_mb(k_spin+p2b-1) .eq. int_mb(k_spin+h7b-1)) THEN
              IF (ieor(int_mb(k_sym+p2b-1),int_mb(k_sym+h7b-1)) 
     1            .eq. irrepa) THEN
               CALL TCE_RESTRICTED_2(p2b,h7b,p2b_1,h7b_1)
               CALL TCE_RESTRICTED_2(h7b,h1b,h7b_2,h1b_2)
               dim_common = int_mb(k_range+h7b-1)
               dima_sort = int_mb(k_range+p2b-1)
               dima = dim_common * dima_sort
               dimb_sort = int_mb(k_range+h1b-1)
               dimb = dim_common * dimb_sort
               IF ((dima .gt. 0) .and. (dimb .gt. 0)) THEN
                IF (.not.MA_PUSH_GET(mt_dbl,dima,'noname',l_a,k_a))
     1               CALL ERRQUIT('ccsd_t1_2',2,MA_ERR)
                CALL GET_HASH_BLOCK_MA(dbl_mb(d_a),dbl_mb(k_a),dima,
     1               int_mb(k_a_offset),
     2               (h7b_1 - 1 + noab * (p2b_1 - noab - 1)))
                IF (.not.MA_PUSH_GET(mt_dbl,dimb,'noname',l_bs,k_bs))
     1               CALL ERRQUIT('ccsd_t1_2',4,MA_ERR)
                IF (.not.MA_PUSH_GET(mt_dbl,dimb,'noname',l_b,k_b))
     1               CALL ERRQUIT('ccsd_t1_2',5,MA_ERR)
                CALL GET_HASH_BLOCK(d_b,dbl_mb(k_b),dimb,
     1               int_mb(k_b_offset),
     2               (h1b_2 - 1 + noab * (h7b_2 - 1)))
                CALL TCE_SORT_2(dbl_mb(k_b),dbl_mb(k_bs),
     1               int_mb(k_range+h7b-1),int_mb(k_range+h1b-1),
     2               2,1,1.0d0)
                IF (.not.MA_POP_STACK(l_b))
     1               CALL ERRQUIT('ccsd_t1_2',6,MA_ERR)
                CALL DGEMM('T','N',dima_sort,dimb_sort,dim_common,1.0d0,
     1               dbl_mb(k_a),dim_common,dbl_mb(k_bs),dim_common,
     2               1.0d0,dbl_mb(k_cs),dima_sort)
                IF (.not.MA_POP_STACK(l_bs))
     1               CALL ERRQUIT('ccsd_t1_2',7,MA_ERR)
                IF (.not.MA_POP_STACK(l_a))
     1               CALL ERRQUIT('ccsd_t1_2',8,MA_ERR)
               END IF
              END IF
             END IF
            END DO
            IF (.not.MA_PUSH_GET(mt_dbl,dimc,'noname',l_c,k_c))
     1           CALL ERRQUIT('ccsd_t1_2',9,MA_ERR)
            CALL TCE_SORT_2(dbl_mb(k_cs),dbl_mb(k_c),
     1           int_mb(k_range+h1b-1),int_mb(k_range+p2b-1),
     2           2,1,-1.0d0)
            CALL ADD_HASH_BLOCK(d_c,dbl_mb(k_c),dimc,int_mb(k_c_offset),
     1           (h1b - 1 + noab * (p2b - noab - 1)))
            IF (.not.MA_POP_STACK(l_c))
     1           CALL ERRQUIT('ccsd_t1_2',10,MA_ERR)
            IF (.not.MA_POP_STACK(l_cs))
     1           CALL ERRQUIT('ccsd_t1_2',11,MA_ERR)
         next = NXTASK(nprocs, 1)
        END IF
        count = count + 1
           END IF
          END IF
         END IF
       END DO
      END DO
      next = NXTASK(-nprocs, 1)
      call GA_SYNC()
      RETURN
      END

