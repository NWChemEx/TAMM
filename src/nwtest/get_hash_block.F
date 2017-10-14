      subroutine get_hash_block(d_file,array,size,hash,key)
c
c $Id: get_hash_block.F 26499 2014-12-10 01:55:15Z jhammond $
c
      implicit none
#include "tce.fh"
c$$$#include "tce_main.fh"
c$$$#include "tce_hetio.fh"
c
      integer d_file
      integer size
      integer hash(*)
      integer key
      integer offset
      integer indexc
      integer iauxo
      double precision array(size)
c
      if (.not.intorb) then
c$$$        if ((d_file.eq.d_v2orb).and.(io_v2.eq.IOSF)) then
c$$$          call tce_hash(hash,key,offset)
c$$$          call get_block_sf(d_file,array,size,offset)
c$$$        else
          call tce_hash(hash,key,offset)
          call get_block(d_file,array,size,offset)
c$$$        endif
      else if (intorb) then
c$$$        if ((d_file.eq.d_v2orb).and.(io_v2.eq.IOSF)) then
c$$$          iauxo=k_v2spin_offset
c$$$          call tce_hash_ind(iauxo,key,indexc)
c$$$          call get_block_ind_sf(d_file,array,size,key,indexc)
c$$$        else if ((d_file.eq.d_v2orb).and.(io_v2.ne.IOSF)) then
         if ((d_file.eq.d_v2orb)) then
          iauxo=k_v2spin_offset
          call tce_hash_ind(iauxo,key,indexc)
          call get_block_ind(d_file,array,size,key,indexc)
        else
          call tce_hash(hash,key,offset)
          call get_block(d_file,array,size,offset)
        endif
      endif
c
c
      return
      end
c
c
      subroutine get_hash_block_i(d_file,array,size,hash,key,
     &                            g2b,g1b,g4b,g3b)
c
c $Id: get_hash_block.F 26499 2014-12-10 01:55:15Z jhammond $
c
      implicit none
#include "tce.fh"
c$$$#include "tce_main.fh"
c$$$#include "tce_hetio.fh"
c
      integer d_file
      integer size
      integer hash(*)
      integer key
      integer offset
      integer indexc
      integer iauxo
      integer g2b,g1b,g4b,g3b
      integer qy2,qy1,qy4,qy3
      double precision array(size)
c
      if (.not.intorb) then
       call tce_hash(hash,key,offset)
       call get_block(d_file,array,size,offset)
      else if (intorb) then
c$$$        if ((d_file.eq.d_v2orb).and.(io_v2.eq.IOGA)) then
        if ((d_file.eq.d_v2orb)) then
c          iauxo=k_v2spin_offset
c          call tce_hash_ind(iauxo,key,indexc)
          qy2=g2b
          qy1=g1b
          qy4=g4b
          qy3=g3b
          call get_block_ind_i(d_file,array,size,key,indexc,
     &                         qy2,qy1,qy4,qy3)
c$$$        else if ((d_file.eq.d_v2orb).and.(io_v2.eq.IOHYBRID)) then
c$$$c          print*,'d_file  = ',d_file
c$$$c          print*,'d_v2orb = ',d_v2orb
c$$$          qy2=g2b
c$$$          qy1=g1b
c$$$          qy4=g4b
c$$$          qy3=g3b
c$$$          if (.not.((g3b.gt.noa).and.(g4b.gt.noa).and.
c$$$     1              (g1b.gt.noa).and.(g2b.gt.noa))) then
c$$$c           This is the GA in-core original
c$$$c            print*,'going to GA'
c$$$c            print*,'d_v2ga  = ',d_v2ga
c$$$            call get_block_ind_i(d_v2ga,array,size,key,indexc,
c$$$     &                           qy2,qy1,qy4,qy3)
c$$$          else
c$$$c           This is the SF out-of-core copy
c$$$c            print*,'going to SF'
c$$$c            print*,'d_v2sf  = ',d_v2sf
c$$$            call get_block_ind_sf_i(d_v2sf,array,size,key,indexc,
c$$$     &                              qy2,qy1,qy4,qy3)
c$$$          endif
c$$$        else if ((d_file.eq.d_v2orb).and.(io_v2.eq.IOSF)) then
c$$$          qy2=g2b
c$$$          qy1=g1b
c$$$          qy4=g4b
c$$$          qy3=g3b
c$$$          call get_block_ind_sf_i(d_file,array,size,key,indexc,
c$$$     &                            qy2,qy1,qy4,qy3)
        else
          call tce_hash(hash,key,offset)
          call get_block(d_file,array,size,offset)
        endif
      end if
c
c
      return
      end
