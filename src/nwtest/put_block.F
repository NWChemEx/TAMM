      subroutine put_block(d_file,array,size,offset)
c
c $Id: put_block.F 28088 2016-02-16 19:35:28Z edo $
c
      implicit none
#include "global.fh"
#include "mafdecls.fh"
c$$$#include "sf.fh"
c$$$#include "eaf.fh"
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
      integer request
      integer i,j
      logical used
      character*255 filename
      logical parallel
c
c$$$      cpusecs(3) = cpusecs(3) - util_cpusec()
c$$$      cpusecs(53) = cpusecs(53) - util_wallsec()
c$$$      parallel = (ga_nnodes().gt.1)
c$$$      if (util_print('put_block',print_debug)) then
c$$$         write(LuOut,9000) ga_nodeid(),d_file,size,offset
c$$$      endif
      if (offset .eq. -1)
     1  call errquit('put_block: illegal offset',0,UNKNOWN_ERR)
c$$$      if (ioalg.eq.0) then
c$$$        used = .false.
c$$$        do i = 1,nfiles
c$$$          if (filehandles(i) .eq. d_file) then
c$$$            j = i
c$$$            used = .true.
c$$$          endif
c$$$        enddo
c$$$        if (.not.used) call errquit('put_block: invalid handle',d_file,
c$$$     1  UNKNOWN_ERR)
c$$$        if (parallel) then
c$$$          call ga_lock(0)
c$$$          open(d_file,file=filenames(j),access='direct',recl=bytes)
c$$$        endif
c$$$        do i = 1,size
c$$$          write(d_file,rec=offset+i) array(i)
c$$$        enddo
c$$$        if (parallel) then
c$$$          close(d_file)
c$$$          call ga_unlock(0)
c$$$        endif
c$$$      else if (ioalg.eq.1) then
c$$$        if (parallel) then
c$$$          used = .false.
c$$$          do i = 1,nfiles
c$$$            if (filehandles(i) .eq. d_file) then
c$$$              j = i
c$$$              used = .true.
c$$$            endif
c$$$          enddo
c$$$          filename = filenames(j)
c$$$          if (.not.used)
c$$$     1      call errquit('put_block: invalid handle',d_file,UNKNOWN_ERR)
c$$$          call ga_lock(0)
c$$$          if (eaf_open(filename, EAF_RW, d_f).ne.0)
c$$$     1      call errquit('put_block: eaf problem',1,DISK_ERR)
c$$$          if (eaf_write(d_f,dfloat(bytes)*dfloat(offset),
c$$$     1      array,bytes*size).ne.0)
c$$$     2      call errquit('put_block: eaf problem',2,DISK_ERR)
c$$$          if (eaf_close(d_f).ne.0)
c$$$     1      call errquit('put_block: eaf problem',4,DISK_ERR)
c$$$          call ga_unlock(0)
c$$$        else
c$$$          if (eaf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1      array,bytes*size).ne.0)
c$$$     2      call errquit('put_block: eaf problem',5,DISK_ERR)
c$$$        endif
c$$$      else if (ioalg.eq.2) then
c$$$        call ga_put(d_file,offset+1,offset+size,1,1,
c$$$     1    array,size)
c$$$      else if (ioalg.eq.3) then
c$$$        if (parallel) then
c$$$          call ga_lock(0)
c$$$          if (sf_open(d_file).ne.0)
c$$$     1      call errquit('put_block: sf problem',0,DISK_ERR)
c$$$        endif
c$$$        if (sf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dfloat(bytes)*dfloat(size),array,request).ne.0)
c$$$     2    call errquit('put_block: sf problem',1,DISK_ERR)
c$$$        if (sf_wait(request).ne.0)
c$$$     1    call errquit('put_block: sf problem',2,DISK_ERR)
c$$$        if (parallel) then
c$$$          if (sf_close(d_file).ne.0)
c$$$     1      call errquit('put_block: sf problem',3,DISK_ERR)
c$$$          call ga_unlock(0)
c$$$        endif
c$$$      else if (ioalg.eq.4) then
c$$$        if (eaf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    array,bytes*size).ne.0)
c$$$     2    call errquit('put_block: eaf problem',2,DISK_ERR)
c$$$      else if (ioalg.eq.5) then
c$$$        if (fileisga(d_file)) then
c$$$          call ga_put(filehandles(d_file),offset+1,offset+size,1,1,
c$$$     1      array,size)
          call ga_put(d_file,offset+1,offset+size,1,1,
     1      array,size)
c$$$        else
c$$$          call errquit('put_block: a write request to DRA',1,
c$$$     1      UNKNOWN_ERR)
c$$$        endif
c$$$      else if (ioalg.eq.6) then
c$$$        if (fileisga(d_file)) then
c$$$          call ga_put(filehandles(d_file),offset+1,offset+size,1,1,
c$$$     1      array,size)
c$$$        else
c$$$          if (eaf_write(filehandles(d_file),
c$$$     1      dfloat(bytes)*dfloat(offset),array,bytes*size).ne.0)
c$$$     2      call errquit('put_block: eaf problem',2,DISK_ERR)
c$$$        endif
c$$$      endif
c$$$ 9000 format(1x,'node',i3,' put_block request to file:',i10,
c$$$     1  ' size:',i10,' offset:',i10)
c$$$      cpusecs(3) = cpusecs(3) + util_cpusec()
c$$$      cpusecs(53) = cpusecs(53) + util_wallsec()
      return
      end



















c$$$
c$$$      subroutine put_block_sf(d_file,array,size,offset)
c$$$c
c$$$c $Id: put_block.F 28088 2016-02-16 19:35:28Z edo $
c$$$c
c$$$      implicit none
c$$$#include "global.fh"
c$$$#include "mafdecls.fh"
c$$$#include "sf.fh"
c$$$#include "eaf.fh"
c$$$#include "util.fh"
c$$$#include "stdio.fh"
c$$$#include "errquit.fh"
c$$$#include "tce.fh"
c$$$#include "tce_main.fh"
c$$$      integer d_file
c$$$      integer d_f
c$$$      integer size
c$$$      integer offset
c$$$      double precision array(size)
c$$$      integer request
c$$$      integer i,j
c$$$      logical used
c$$$      character*255 filename
c$$$      logical parallel
c$$$c
c$$$      cpusecs(3) = cpusecs(3) - util_cpusec()
c$$$      cpusecs(53) = cpusecs(53) - util_wallsec()
c$$$      parallel = (ga_nnodes().gt.1)
c$$$      if (util_print('put_block',print_debug)) then
c$$$         write(LuOut,9000) ga_nodeid(),d_file,size,offset
c$$$      endif
c$$$      if (offset .eq. -1)
c$$$     1  call errquit('put_block: illegal offset',0,UNKNOWN_ERR)
c$$$c
c$$$c      else if (ioalg.eq.3) then
c$$$c
c$$$        if (parallel) then
c$$$          call ga_lock(0)
c$$$          if (sf_open(d_file).ne.0)
c$$$     1      call errquit('put_block: sf problem',0,DISK_ERR)
c$$$        endif
c$$$        if (sf_write(d_file,dfloat(bytes)*dfloat(offset),
c$$$     1    dfloat(bytes)*dfloat(size),array,request).ne.0)
c$$$     2    call errquit('put_block: sf problem',1,DISK_ERR)
c$$$        if (sf_wait(request).ne.0)
c$$$     1    call errquit('put_block: sf problem',2,DISK_ERR)
c$$$        if (parallel) then
c$$$          if (sf_close(d_file).ne.0)
c$$$     1      call errquit('put_block: sf problem',3,DISK_ERR)
c$$$          call ga_unlock(0)
c$$$        endif
c$$$c
c$$$c      endif
c$$$c
c$$$ 9000 format(1x,'node',i3,' put_block request to file:',i10,
c$$$     1  ' size:',i10,' offset:',i10)
c$$$      cpusecs(3) = cpusecs(3) + util_cpusec()
c$$$      cpusecs(53) = cpusecs(53) + util_wallsec()
c$$$      return
c$$$      end
