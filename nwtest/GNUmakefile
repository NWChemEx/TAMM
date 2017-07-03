
MPICC = mpicc
MPIFC = mpif90
LD = mpif90

GA_CONFIG = /Users/sriram/code/ga/build/bin/ga-config

FFLAGS = -I. $(shell $(GA_CONFIG) --cppflags --fflags)
CFLAGS = -I. $(shell $(GA_CONFIG) --cppflags --cflags)
LDFLAGS = $(shell $(GA_CONFIG) --ldflags)
LIBS =  $(shell $(GA_CONFIG) --libs)

FSRCS = \
put_hash_block.F \
add_block.F \
get_block.F \
get_block_ind.F \
add_hash_block.F \
get_hash_block.F \
nxtask.F \
nxtval_ga.F \
errquit.F \
put_block.F \
tce_hash.F \
tce_hash_ind.F \
tce_restricted.F \
tce_sort.F \
tce_sortacc.F \
misc.F


CSRCS = \
util_gnxtval.c

OBJS = $(FSRCS:.F=.o) $(CSRCS:.c=.o)

default: test


test: test.o $(OBJS)
	$(LD) -o test test.o $(OBJS) $(LDFLAGS) $(LIBS)

clean:
	rm -f *~ *.o test

.SUFFIXES: .c .o .F


.F.o:
	$(MPIFC) -c $(FFLAGS) $*.F


.c.o:
	$(MPICC) -c $(CFLAGS) $*.c
