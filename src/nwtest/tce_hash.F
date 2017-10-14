      subroutine tce_hash_v2(hash,key,offset)
c
c $Id: tce_hash.F 28220 2016-05-10 23:45:09Z edo $
c
      implicit none
#include "util.fh"
#include "mafdecls.fh"
c#include "stdio.fh"
#include "errquit.fh"
#include "tce.fh"
c#include "tce_main.fh"
      integer hash(*)
      integer key
      integer offset
      integer length,length1
      integer i,j,k,l
      integer i_start,j_start,k_start,l_start
      integer i_stop,j_stop,k_stop,l_stop
      integer pos1,pos2
      integer pos1_l,pos1_u
      integer pos2_l,pos2_u
      integer middle
      integer key_loop
      integer irow,icol,index_pair,index_point
      integer sa_ij,sa_kl
      integer ieo_kl,n
      integer indx_pairij,indx_point
      indx_point(i,j,n)=
     =     (i*(2*n+1-i))/2 -n + j
      indx_pairij(i,j)=j+(i*(i-1))/2
c
      length = hash(1)
      middle=-1
      do i=1,length
        if((hash(i+1).le.key).and.(key.le.hash(i+2))) then
            middle=i
            go to 20
        end if
      end do 
      if(middle.eq.-1) 
     &      call errquit('hash: key not found 1',0,UNKNOWN_ERR) 
   20 continue
c middle is fixed
c
      i_start=hash(2*(length+1)+middle+1)
      j_start=hash(3*(length+1)+middle+1)
      k_start=hash(4*(length+1)+middle+1)
      l_start=hash(5*(length+1)+middle+1)
c
      i_stop=hash(2*(length+1)+(middle+1)+1)
      j_stop=hash(3*(length+1)+(middle+1)+1)
      k_stop=hash(4*(length+1)+(middle+1)+1)
      l_stop=hash(5*(length+1)+(middle+1)+1)
c
      offset=hash((length+1)+middle+1)
c
      pos1_l=indx_point(i_start,j_start,noa+nva)
      pos1_u=indx_point(i_stop,j_stop,noa+nva)
c
      pos2_l=indx_point(k_start,l_start,noa+nva)
      pos2_u=indx_point(k_stop,l_stop,noa+nva)
c
      i=i_start
      j=j_start
      k=k_start
      l=l_start
c
      do pos1=pos1_l,pos1_u  !---------------------------------------
      ieo_kl=ieor(int_mb(k_sym_alpha+k-1),int_mb(k_sym_alpha+l-1))
      sa_kl=int_mb(k_spin_alpha+k-1)+int_mb(k_spin_alpha+l-1)
#if 0
       if(pos1.ne.pos1_l) then
        if(pos1.eq.indx_point(i+1,i+1,noa+nva)) then
          i=i+1
          j=i
        else
          j=j+1
        end if
       end if
#endif
       sa_ij=int_mb(k_spin_alpha+i-1)+int_mb(k_spin_alpha+j-1)
c
  100  continue
c
          IF (sa_ij.eq.sa_kl) then
          IF (ieor(int_mb(k_sym_alpha+i-1),ieor(int_mb(k_sym_alpha+j-1),
     &    ieo_kl)) .eq.
     &    irrep_v) THEN
          irow=indx_pairij(j,i)
          icol=indx_pairij(l,k)
          IF(irow.ge.icol) THEN
           key_loop= l - 1 + (noa+nva) * (k - 1 + (noa+nva) * (j-
     &               1 + (noa+nva) * (i - 1)))
           if(key_loop.eq.key) go to 103
           offset=offset+int_mb(k_range_alpha+i-1)*
     &            int_mb(k_range_alpha+j-1)*int_mb(k_range_alpha+k-1)*
     &            int_mb(k_range_alpha+l-1)
          end if
          end if
          end if
c if i eq j eq k eq l eq - wyskok
          if((i.eq.i_stop).and.(j.eq.j_stop).and.(k.eq.k_stop).and.
     &       (l.eq.l_stop)) go to 102
          if((k.eq.noa+nva).and.(l.eq.noa+nva)) go to 200
          if(l.eq.noa+nva) then
            k=k+1
            l=k
          else
            l=l+1
          end if
          sa_kl=int_mb(k_spin_alpha+k-1)+int_mb(k_spin_alpha+l-1)
          ieo_kl=ieor(int_mb(k_sym_alpha+k-1),int_mb(k_sym_alpha+l-1))
          go to 100
c 
  200 continue
      k=1
      l=1
          sa_kl=int_mb(k_spin_alpha+k-1)+int_mb(k_spin_alpha+l-1)
          ieo_kl=ieor(int_mb(k_sym_alpha+k-1),int_mb(k_sym_alpha+l-1))
        if(pos1+1.eq.indx_point(i+1,i+1,noa+nva)) then
          i=i+1
          j=i
        else
          j=j+1
        end if
      enddo                 !---- pos1 -------------------------------
c
c
  102 continue
      write(*,*)'key=',key
      call errquit('hashv2: key not found 2',0,UNKNOWN_ERR)
  103 continue
      return
      end
c
c
c
c
c$$$      subroutine tce_hash_v2_act(hash,key,offset)
c$$$c
c$$$c $Id: tce_hash.F 28220 2016-05-10 23:45:09Z edo $
c$$$c
c$$$      implicit none
c$$$#include "util.fh"
c$$$#include "mafdecls.fh"
c$$$c#include "stdio.fh"
c$$$#include "errquit.fh"
c$$$#include "tce.fh"
c$$$c#include "tce_main.fh"
c$$$      integer hash(*)
c$$$      integer key
c$$$      integer offset
c$$$      integer length,length1
c$$$      integer i,j,k,l
c$$$      integer i_start,j_start,k_start,l_start
c$$$      integer i_stop,j_stop,k_stop,l_stop
c$$$      integer pos1,pos2
c$$$      integer pos1_l,pos1_u
c$$$      integer pos2_l,pos2_u
c$$$      integer middle
c$$$      integer key_loop
c$$$      integer irow,icol,index_pair,index_point
c$$$c - exascale ---
c$$$      LOGICAL is_active_4_o
c$$$c --------------
c$$$c
c$$$      length = hash(1)
c$$$      middle=-1
c$$$      do i=1,length
c$$$        if((hash(i+1).le.key).and.(key.le.hash(i+2))) then
c$$$            middle=i
c$$$            go to 20
c$$$        end if
c$$$      end do 
c$$$      if(middle.eq.-1) 
c$$$     &      call errquit('hash: key not found 1',0,UNKNOWN_ERR) 
c$$$   20 continue
c$$$c middle is fixed
c$$$c
c$$$      i_start=hash(2*(length+1)+middle+1)
c$$$      j_start=hash(3*(length+1)+middle+1)
c$$$      k_start=hash(4*(length+1)+middle+1)
c$$$      l_start=hash(5*(length+1)+middle+1)
c$$$c
c$$$      i_stop=hash(2*(length+1)+(middle+1)+1)
c$$$      j_stop=hash(3*(length+1)+(middle+1)+1)
c$$$      k_stop=hash(4*(length+1)+(middle+1)+1)
c$$$      l_stop=hash(5*(length+1)+(middle+1)+1)
c$$$c
c$$$      offset=hash((length+1)+middle+1)
c$$$c
c$$$      pos1_l=index_point(i_start,j_start,noa+nva)
c$$$      pos1_u=index_point(i_stop,j_stop,noa+nva)
c$$$c
c$$$      pos2_l=index_point(k_start,l_start,noa+nva)
c$$$      pos2_u=index_point(k_stop,l_stop,noa+nva)
c$$$c
c$$$      i=i_start
c$$$      j=j_start
c$$$      k=k_start
c$$$      l=l_start
c$$$c
c$$$      do pos1=pos1_l,pos1_u  !---------------------------------------
c$$$       if(pos1.ne.pos1_l) then
c$$$        if(pos1.eq.index_point(i+1,i+1,noa+nva)) then
c$$$          i=i+1
c$$$          j=i
c$$$        else
c$$$          j=j+1
c$$$        end if
c$$$       end if
c$$$c
c$$$  100  continue
c$$$c
c$$$c - exascale ---
c$$$      IF(.not.((i.gt.noa).and.(j.gt.noa).and.(k.gt.noa).and.
c$$$     &   (l.gt.noa).and.(.not.is_active_4_o(i,j,k,l)))) THEN
c$$$c ---
c$$$          IF (int_mb(k_spin_alpha+i-1)+int_mb(k_spin_alpha+j-1).eq.
c$$$     &    int_mb(k_spin_alpha+k-1)+int_mb(k_spin_alpha+l-1)) THEN
c$$$          IF (ieor(int_mb(k_sym_alpha+i-1),ieor(int_mb(k_sym_alpha+j-1),
c$$$     &    ieor(int_mb(k_sym_alpha+k-1),int_mb(k_sym_alpha+l-1)))) .eq.
c$$$     &    irrep_v) THEN
c$$$          irow=index_pair(j,i)
c$$$          icol=index_pair(l,k)
c$$$          IF(irow.ge.icol) THEN
c$$$           key_loop= l - 1 + (noa+nva) * (k - 1 + (noa+nva) * (j-
c$$$     &               1 + (noa+nva) * (i - 1)))
c$$$           if(key_loop.eq.key) go to 103
c$$$           offset=offset+int_mb(k_range_alpha+i-1)*
c$$$     &            int_mb(k_range_alpha+j-1)*int_mb(k_range_alpha+k-1)*
c$$$     &            int_mb(k_range_alpha+l-1)
c$$$          end if
c$$$          end if
c$$$          end if
c$$$c - exascale ---
c$$$      END IF
c$$$c ---
c$$$c if i eq j eq k eq l eq - wyskok
c$$$          if((i.eq.i_stop).and.(j.eq.j_stop).and.(k.eq.k_stop).and.
c$$$     &       (l.eq.l_stop)) go to 102
c$$$          if((k.eq.noa+nva).and.(l.eq.noa+nva)) go to 200
c$$$          if(l.eq.noa+nva) then
c$$$            k=k+1
c$$$            l=k
c$$$          else
c$$$            l=l+1
c$$$          end if
c$$$          go to 100
c$$$c
c$$$  200 continue
c$$$      k=1
c$$$      l=1
c$$$      enddo                 !---- pos1 -------------------------------
c$$$c
c$$$c
c$$$c
c$$$c
c$$$  102 continue
c$$$      write(*,*)'key=',key
c$$$      call errquit('hashv2act: key not found 2',0,UNKNOWN_ERR)
c$$$  103 continue
c$$$      return
c$$$      end
c
c
c
c
c
c
      subroutine tce_hash(hash,key,offset)
c
c $Id: tce_hash.F 28220 2016-05-10 23:45:09Z edo $
c
      implicit none
#include "util.fh"
c#include "stdio.fh"
#include "errquit.fh"
      integer hash(*)
      integer key
      integer offset
      integer length
      integer i
      integer less,middle,more
c
      length = hash(1)
      less = 1
      more = length
   10 continue
        if (more-less.le.4) then
          middle = -1
          do i = less, more
            if (hash(i+1).eq.key) middle = i
          enddo
          if (middle.eq.-1) then
            write(*,*) 'key = ',key
            do i = 1,hash(1)
              write(*,*) i,hash(i+1)
            enddo
            call errquit('tce_hash: key not found',0,UNKNOWN_ERR)
          endif
          goto 20
        endif
        middle = (less + more) / 2
        if (hash(middle+1).eq.key) then
          goto 20
        else if (hash(middle+1).gt.key) then
          more = middle
        else
          less = middle
        endif
        goto 10
   20 continue
      offset = hash(length+middle+1)
      return
      end
c
c
c
c
c
c
c
      subroutine tce_hash_n(hash,key,offset)
c
c $Id: tce_hash.F 28220 2016-05-10 23:45:09Z edo $
c
      implicit none
#include "util.fh"
c#include "stdio.fh"
#include "errquit.fh"
#include "mafdecls.fh"
      integer hash(*)
      integer key
      integer offset
      integer length
      integer i
c
cedo      if (.not.ma_verify_allocator_stuff()) stop ' ma ck 2b'
      length = hash(1)
cedo      if (.not.ma_verify_allocator_stuff()) stop ' ma ck 2b'
      if(length.lt.0) then
         write(6,*) key,hash(1),hash(2)
         call errquit('tce_hash_n: negative length',length,UNKNOWN_ERR)
      endif
       offset = -1
       do i = 1,length
        if(key.eq.hash(i+1)) then
         offset=hash(length+i+1)
         goto 10
        end if
       end do
   10  continue
cedo        if (.not.ma_verify_allocator_stuff()) stop ' ma ck 2b'
       if(offset.eq.-1) then
#if 1
      write(6,123) ' hashn: addr ',i,' key ',key
      write(6,*) ' length ',length
      do i=1,length
         write(6,*) ' i ',i,' hash ',hash(i+1)
      enddo
 123  format(a,i12,a,i12)
#endif
         call errquit('tce_hash_n: key not found',key,UNKNOWN_ERR)
       end if
c
      return
      end
