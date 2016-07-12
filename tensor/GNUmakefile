
  SET_CPP_USAGE = 1

  GA_CONFIG = $(NWCHEM_TOP)/src/tools/install/bin/ga-config

	CC = $(shell $(GA_CONFIG) --cc)
	#CC = icc
	CXX = mpicxx
	FC = $(shell $(GA_CONFIG) --f77)
	#FC = ifort
	CFLAGS = $(shell $(GA_CONFIG) --cppflags --cflags)
	CXXINCLUDES = $(shell $(GA_CONFIG) --cppflags --cflags)
	FINCLUDES = $(shell $(GA_CONFIG) --fflags) -I$(NWCHEM_TOP)/src/tce/include 

default: libctce.a

libctce.a: variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o \
ccsd_t1.o \
ccsd_t2_12.o ccsd_t2_45.o ccsd_t2_68.o ccsd_t2_7.o \
ccsd_t.o ipccsd_x1.o input.o \
 gt.o fapi.o
	ar -cr -o libctce.a variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o \
ccsd_t1.o \
ccsd_t2_12.o ccsd_t2_45.o ccsd_t2_68.o ccsd_t2_7.o \
ccsd_t.o ipccsd_x1.o input.o \
gt.o fapi.o


clean:
	rm -f *~ *.o *.x *.a *.err *.out 

.SUFFIXES: .cc .F .c

.cc.o:
	$(CC) $(CXXINCLUDES) -DEXT_INT -DGFORTRAN -c -g $*.cc

.F.o:
	$(FC) $(FINCLUDES) -c -g $*.F

.c.o:
	$(CC) $(CFLAGS) -c -g $*.c

