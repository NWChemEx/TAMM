      subroutine add_block(d_file,array,size,offset)
c
c $Id: add_block.F 19706 2010-10-29 17:52:31Z d3y133 $
c
c     This subroutine reads a section of a file and adds
c     the content of an array to it and write it back to
c     the original place in the file.  Note that two or more
c     different processes can read/add/write to the same 
c     section, the entire read/add/write must be protected
c     from interruption.
c
      implicit none
#include "global.fh"
#include "mafdecls.fh"
#include "sf.fh"
#include "eaf.fh"
#include "util.fh"
c$$$#include "stdio.fh"
#include "errquit.fh"
#include "tce.fh"
c$$$#include "tce_main.fh"
      integer d_file
      integer d_f
      integer size
      integer offset
      double precision array(size)
      integer l_temp,k_temp
      integer request
      integer i,j
      logical used
      double precision element
      character*255 filename
      logical parallel
c
c$$$      cpusecs(4) = cpusecs(4) - util_cpusec()
c$$$      cpusecs(54) = cpusecs(54) - util_wallsec()
c$$$      parallel = (ga_nnodes().gt.1)
c$$$      if (util_print('add_block',print_debug)) then
c$$$         write(LuOut,9000) ga_nodeid(),d_file,size,offset
c$$$      endif
c$$$      if (offset .eq. -1)
c$$$     1  call errquit('add_block: illegal offset',0,UNKNOWN_ERR)
c$$$      if (ioalg.eq.0) then
c$$$        if (parallel) call ga_lock(0)
c$$$        used = .false.
c$$$        do i = 1,nfiles
c$$$          if (filehandles(i) .eq. d_file) then
c$$$            j = i
c$$$            used = .true.
c$$$          endif
c$$$        enddo
c$$$        if (.not.used) call errquit('add_block: invalid handle',d_file,
c$$$     1    UNKNOWN_ERR)
c$$$        if (parallel) open(d_file,file=filenames(j),
c$$$     1    access='direct',recl=bytes)
c$$$        do i = 1,size
c$$$          read(d_file,rec=offset+i) element
c$$$          write(d_file,rec=offset+i) array(i)+element
c$$$        enddo
c$$$        if (parallel) then
c$$$          close(d_file)
c$$$          call ga_unlock(0)
c$$$        endif
c$$$      else if (ioalg.eq.1) then
c$$$        if (.not.ma_push_get(mt_dbl,size,'temporary',
c$$$     1    l_temp,k_temp))
c$$$     2    call errquit('add_block: MA problem',0,MA_ERR)
c$$$        if (parallel) then
c$$$          used = .false.
c$$$          do i = 1,nfiles
c$$$            if (filehandles(i) .eq. d_file) then
c$$$              j = i
c$$$              used = .true.
c$$$            endif
c$$$          enddo
c$$$          filename = filenames(j)
c$$$          call ga_lock(0)
c$$$          if (eaf_open(filename, EAF_RW, d_f).ne.0)
c$$$     1      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          if (eaf_read(d_f,dfloat(bytes)*dfloat(offset),
c$$$     1      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          do i = 1,size
c$$$            dbl_mb(k_temp+i-1) = dbl_mb(k_temp+i-1) + array(i)
c$$$          enddo
c$$$          if (eaf_write(d_f,dfloat(bytes)*dfloat(offset),
c$$$     1      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          if (eaf_close(d_f).ne.0)
c$$$     1      call errquit('add_block: eaf problem',0,DISK_ERR)
c$$$          call ga_unlock(0)
c$$$        else
c$$$          if (eaf_read(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          do i = 1,size
c$$$            dbl_mb(k_temp+i-1) = dbl_mb(k_temp+i-1) + array(i)
c$$$          enddo
c$$$          if (eaf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$        endif
c$$$        if (.not.ma_pop_stack(l_temp))
c$$$     1    call errquit('add_block: MA problem',0,MA_ERR)
c$$$      else if (ioalg.eq.2) then
c$$$        call ga_acc(d_file,offset+1,offset+size,1,1,
c$$$     1    array,1,1.0d0)
c$$$      else if (ioalg.eq.3) then
c$$$        if (.not.ma_push_get(mt_dbl,size,'temporary',
c$$$     1    l_temp,k_temp))
c$$$     2    call errquit('add_block: MA problem',0,MA_ERR)
c$$$        if (parallel) then
c$$$          call ga_lock(0)
c$$$          if (sf_open(d_file).ne.0)
c$$$     1      call errquit('add_block: sf problem',0,DISK_ERR)
c$$$        endif
c$$$        if (sf_read(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dfloat(bytes)*dfloat(size),dbl_mb(k_temp),request).ne.0)
c$$$     2    call errquit('add_block: sf problem',1,DISK_ERR)
c$$$        if (sf_wait(request).ne.0)
c$$$     1    call errquit('add_block: sf problem',2,DISK_ERR)
c$$$        do i = 1,size
c$$$          dbl_mb(k_temp+i-1) = dbl_mb(k_temp+i-1) + array(i)
c$$$        enddo
c$$$        if (sf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dfloat(bytes)*dfloat(size),dbl_mb(k_temp),request).ne.0)
c$$$     2    call errquit('add_block: sf problem',3,DISK_ERR)
c$$$        if (sf_wait(request).ne.0)
c$$$     1    call errquit('add_block: sf problem',4,DISK_ERR)
c$$$        if (parallel) then
c$$$          if (sf_close(d_file).ne.0)
c$$$     1      call errquit('add_block: sf problem',5,DISK_ERR)
c$$$          call ga_unlock(0)
c$$$        endif
c$$$        if (.not.ma_pop_stack(l_temp))
c$$$     1    call errquit('add_block: MA problem',1,MA_ERR)
c$$$      else if (ioalg.eq.4) then
c$$$        if (.not.ma_push_get(mt_dbl,size,'temporary',
c$$$     1    l_temp,k_temp))
c$$$     2    call errquit('add_block: MA problem',0,MA_ERR)
c$$$        if (eaf_read(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2    call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$        do i = 1,size
c$$$          dbl_mb(k_temp+i-1) = dbl_mb(k_temp+i-1) + array(i)
c$$$        enddo
c$$$        if (eaf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dbl_mb(k_temp),bytes*size).ne.0)
c$$$     2    call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$        if (.not.ma_pop_stack(l_temp))
c$$$     1    call errquit('add_block: MA problem',0,MA_ERR)
c$$$      else if (ioalg.eq.5) then
c$$$        if (fileisga(d_file)) then
c$$$          call ga_acc(filehandles(d_file),offset+1,offset+size,1,1,
c$$$     1      array,1,1.0d0)
          call ga_acc(d_file,offset+1,offset+size,1,1,
     1      array,1,1.0d0)
c$$$        else
c$$$          call errquit('add_block: read/write requests to DRA',0,
c$$$     1      UNKNOWN_ERR)
c$$$        endif
c$$$      else if (ioalg.eq.6) then
c$$$        if (fileisga(d_file)) then
c$$$          call ga_acc(filehandles(d_file),offset+1,offset+size,1,1,
c$$$     1      array,1,1.0d0)
c$$$        else
c$$$          if (.not.ma_push_get(mt_dbl,size,'temporary',
c$$$     1      l_temp,k_temp))
c$$$     2      call errquit('add_block: MA problem',0,MA_ERR)
c$$$          if (eaf_read(filehandles(d_file),
c$$$     1      dfloat(bytes)*dfloat(offset),
c$$$     2      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     3      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          do i = 1,size
c$$$            dbl_mb(k_temp+i-1) = dbl_mb(k_temp+i-1) + array(i)
c$$$          enddo
c$$$          if (eaf_write(filehandles(d_file),
c$$$     1      dfloat(bytes)*dfloat(offset),
c$$$     2      dbl_mb(k_temp),bytes*size).ne.0)
c$$$     3      call errquit('add_block: eaf problem',2,DISK_ERR)
c$$$          if (.not.ma_pop_stack(l_temp))
c$$$     1      call errquit('add_block: MA problem',0,MA_ERR)
c$$$        endif
c$$$      endif
c$$$ 9000 format(1x,'node',i3,' add_block request to file:',i10,
c$$$     1  ' size:',i10,' offset:',i10)
c$$$      cpusecs(4) = cpusecs(4) + util_cpusec()
c$$$      cpusecs(54) = cpusecs(54) + util_wallsec()
      return
      end












c$$$      subroutine add_block_nb(d_file,array,size,offset,nbh)
c$$$c
c$$$      implicit none
c$$$#include "global.fh"
c$$$#include "mafdecls.fh"
c$$$#include "stdio.fh"
c$$$#include "errquit.fh"
c$$$#include "util.fh"
c$$$#include "tce.fh"
c$$$#include "tce_main.fh"
c$$$c
c$$$      integer d_file
c$$$      integer size
c$$$      integer offset
c$$$      integer nbh
c$$$      double precision array(size)
c$$$      logical parallel
c$$$c
c$$$      cpusecs(4) = cpusecs(4) - util_cpusec()
c$$$      cpusecs(54) = cpusecs(54) - util_wallsec()
c$$$c
c$$$      parallel = (ga_nnodes().gt.1)
c$$$c
c$$$      if (util_print('add_block',print_debug)) then
c$$$         write(LuOut,9000) ga_nodeid(),d_file,size,offset
c$$$      endif
c$$$c
c$$$      if (offset .eq. -1)
c$$$     1  call errquit('add_block_nb: illegal offset',0,UNKNOWN_ERR)
c$$$c
c$$$      if (ioalg.eq.2) then
c$$$        call ga_nbacc(d_file,offset+1,offset+size,1,1,array,1,1.0d0,nbh)
c$$$      else
c$$$        call errquit('add_block_nb: only for GA!!!',911,MA_ERR)
c$$$      endif
c$$$c
c$$$      cpusecs(4) = cpusecs(4) + util_cpusec()
c$$$      cpusecs(54) = cpusecs(54) + util_wallsec()
c$$$      return
c$$$c
c$$$ 9000 format(1x,'node',i3,' add_block_nb request to file:',i10,
c$$$     1  ' size:',i10,' offset:',i10)
c$$$c
c$$$      end
