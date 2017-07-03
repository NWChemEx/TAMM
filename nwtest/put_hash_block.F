      subroutine put_hash_block(d_file,array,size,hash,key)
c
c $Id: put_hash_block.F 19706 2010-10-29 17:52:31Z d3y133 $
c
      implicit none
      integer d_file
      integer size
      integer hash(*)
      integer key
      integer offset
      double precision array(size)
c
      call tce_hash(hash,key,offset)
      call put_block(d_file,array,size,offset)
      return
      end






c$$$      subroutine put_hash_block_sf(d_file,array,size,hash,key)
c$$$c
c$$$c $Id: put_hash_block.F 19706 2010-10-29 17:52:31Z d3y133 $
c$$$c
c$$$      implicit none
c$$$      integer d_file
c$$$      integer size
c$$$      integer hash(*)
c$$$      integer key
c$$$      integer offset
c$$$      double precision array(size)
c$$$c
c$$$      call tce_hash(hash,key,offset)
c$$$      call put_block_sf(d_file,array,size,offset)
c$$$      return
c$$$      end
