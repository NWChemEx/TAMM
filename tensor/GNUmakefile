
  SET_CPP_USAGE = 1

  GA_CONFIG = $(NWCHEM_TOP)/src/tools/install/bin/ga-config

	CC = $(shell $(GA_CONFIG) --cc)
	#CC = icc
	CXX = mpicxx
	FC = $(shell $(GA_CONFIG) --f77)
	#FC = ifort
	CFLAGS = $(shell $(GA_CONFIG) --cppflags --cflags)
	CXXINCLUDES = $(shell $(GA_CONFIG) --cppflags --cflags) -I../ctce_parser
	FINCLUDES = $(shell $(GA_CONFIG) --fflags) -I$(NWCHEM_TOP)/src/tce/include 
	LDFLAGS = $(shell $(GA_CONFIG) --ldflags)
	LIBS = $(shell $(GA_CONFIG) --libs)

default: libctce.a

libctce.a: variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o \
 \
equations.o ccsd_t1_eqs.o ccsd_t2_eqs.o corf.o ccsd_t1_ref.o ccsd_t2_ref.o ccsd_e_ref.o \
 gt.o fapi.o schedulers.o ccsd_t12.o ccsd_e_eqs.o tensors_and_ops.o stats.o index_sort.o
	ar -cr -o libctce.a variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o\
 \
equations.o ccsd_t1_eqs.o ccsd_t2_eqs.o ccsd_t2_ref.o corf.o ccsd_t1_ref.o ccsd_e_ref.o \
gt.o fapi.o schedulers.o ccsd_t12.o ccsd_e_eqs.o tensors_and_ops.o stats.o index_sort.o


main: ccsd_t1_eqs.o ccsd_t2_eqs.o equations.o main.o print.o ../ctce_parser/libctceparser.a
	$(CXX) -o main equations.o ccsd_t2_eqs.o ccsd_t1_eqs.o main.o print.o ../ctce_parser/libctceparser.a $(LDFLAGS) $(LIBS)

clean:
	rm -f *~ *.o *.x *.a *.err *.out 

.SUFFIXES: .cc .F .c

.cc.o:
	$(CXX) $(CXXINCLUDES) -DEXT_INT -DGFORTRAN -c -g $*.cc

.F.o:
	$(FC) $(FINCLUDES) -c -g $*.F

.c.o:
	$(CC) $(CFLAGS) -c -g $*.c

