      subroutine errquit(string, icode, errcode)
      implicit none
C$Id: errquit.F 27416 2015-08-24 19:00:36Z jhammond $
#include "errquit.fh"
#include "global.fh"
#include "util.fh"
#include "mafdecls.fh"
c$$$#include "stdio.fh"
      character*(*) string
      character*255 man_sec
      integer icode
      integer errcode
c$$$c     
c$$$c     error termination 
c$$$c     
c$$$c     string = error message printed to stdout
c$$$c     icode  = informative value printed to stdout
c$$$c
c$$$c     try to minimize the amount of output if all nodes 
c$$$c     detect an error
c$$$c     
c$$$      call ecce_print_module_entry('errquit')
c$$$      call ecce_print1_char('errquit message',string,1)
c$$$      call ecce_print1('errquit code', mt_int,icode,1)
c$$$      call ecce_print_module_exit('errquit', 'failed')
c$$$c
c$$$      man_sec = ' '
c$$$      if (ga_nodeid() .eq. 0) write(LuOut,1)
c$$$    1 format(1x,72('-'))
c$$$      write(LuOut, 2) string, icode
c$$$    2 format(1x,a,i20)
c$$$    3 format(/)
c$$$      write(LuOut,1)
c$$$      call inp_errout
c$$$      write(LuOut,1)
c$$$      write(LuOut,*) ' current input line : '
c$$$      call inp_outrec
c$$$      write(LuOut,1)
c$$$c
c$$$c       categorization of errors
c$$$c     
c$$$      write(LuOut,1) 
c$$$      if (errcode .eq. UNKNOWN_ERR) then
c$$$         write(LuOut,*) 'This error has not yet been assigned to a ',
c$$$     &        'category'
c$$$         man_sec = 'No section for this category'
c$$$      endif
c$$$c     
c$$$#if 0
c$$$      if (errcode .eq. MEM_ERR) then
c$$$         write(LuOut,*) 'You need to assign at least ', icode, 
c$$$     &        'bytes more memory'
c$$$         write(LuOut,3)
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'Add a memory card of the following format: '
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'memory total <default + ', icode, '>'
c$$$         man_sec = 'NWChem -> Top Level Directives -> Memory'
c$$$      endif
c$$$c
c$$$      if (errcode .eq. STACK_ERR) then
c$$$         write(LuOut,*) 'You need to assign at least ', icode,
c$$$     &        'bytes more stack memory'
c$$$         write(LuOut,3)
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'Add a memory card of the following format: '
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'memory stack <default + ', icode, '>'
c$$$         man_sec = 'NWChem -> Top Level Directives -> Memory'
c$$$      endif
c$$$c     
c$$$      if (errcode .eq. HEAP_ERR) then
c$$$         write(LuOut,*) 'You need to assign at least ', icode,
c$$$     &        'bytes more heap memory'
c$$$         write(LuOut,3)
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'Add a memory card of the following format: '
c$$$         write(LuOut,3)
c$$$         write(LuOut,*) 'memory heap <default + ', icode, '>'
c$$$         man_sec = 'NWChem -> Top Level Directives -> Memory'
c$$$      endif
c$$$#endif
c$$$c     
c$$$      if (errcode .eq. RTDB_ERR) then
c$$$         write(LuOut,*) 'An error occured in the Runtime Database'
c$$$      endif
c$$$c     
c$$$      if (errcode .eq. BASIS_ERR) then
c$$$         write(LuOut,*) 'There is an error in the specified basis set'
c$$$      endif
c$$$c     
c$$$      if (errcode .eq. GEOM_ERR) then
c$$$         write(LuOut,*) 'There is an error related to the specified ',
c$$$     &        'geometry'
c$$$      endif
c$$$c
c$$$      if (errcode .eq. INPUT_ERR) then
c$$$         write(LuOut,*) 'There is an error in the input file'
c$$$      endif
c$$$c
c$$$      if (errcode .eq. CAPMIS_ERR) then
c$$$         write(LuOut,*) 'A feature requested has not yet been ',
c$$$     &        'implemented'
c$$$         write(LuOut,*) 
c$$$      endif
c$$$c     
c$$$#if 0
c$$$      if (errcode .eq. GA_ERR) then
c$$$         write(LuOut,*) 'You need to assign at least ', icode,
c$$$     &        ' bytes more global memory'
c$$$         man_sec = 'NWChem -> Top Level Directives -> Memory'
c$$$      endif
c$$$c     
c$$$      if (errcode .eq. MA_ERR) then
c$$$         write(LuOut,*) 'You need to assign at least ', icode,
c$$$     &        ' bytes more local memory'
c$$$         man_sec = 'NWChem -> Top Level Directives -> Memory'
c$$$      endif
c$$$#endif
c$$$c
c$$$      if (errcode .eq. INT_ERR) then
c$$$         write(LuOut,*) 'An error occured while computing integrals'
c$$$      endif
c$$$c
c$$$      if (errcode .eq. DISK_ERR) then
c$$$         write(LuOut,*) 'An error occured while trying to read or ',
c$$$     &        'write to disk space'
c$$$      endif
c$$$c
c$$$      if (errcode .eq. CALC_ERR) then
c$$$         write(LuOut,*) 'This type of error is most commonly ',
c$$$     &        'associated with calculations not reaching ',
c$$$     &        'convergence criteria'
c$$$      endif 
c$$$c
c$$$      write(LuOut,1) 
c$$$      write(LuOut,*) 'For more information see the NWChem manual at ',
c$$$     &     'http://www.nwchem-sw.org/index.php/NWChem_Documentation'
c$$$      write(LuOut,3)
c$$$      write(LuOut,*) 'For further details see manual section: ',
c$$$     &     man_sec
c$$$
c$$$      call util_flush(LuOut)
      call ga_error(string, icode)
c
      end

      subroutine util_warning(string, icode, errcode)
      implicit none
C$Id: errquit.F 27416 2015-08-24 19:00:36Z jhammond $
#include "errquit.fh"
#include "global.fh"
#include "util.fh"
#include "mafdecls.fh"
c$$$#include "stdio.fh"
      character*(*) string
      character*255 man_sec
      integer icode
      integer errcode
c$$$
c$$$      if (ga_nodeid() .eq. 0) 
c$$$     >    write(LuOut,*) string
c$$$      call util_flush(LuOut)
c
      end
