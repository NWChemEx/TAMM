
  #GA_CONFIG = /people/gawa722/stuff/nwchem6dot4_dev/Nwchem-6.5/src/tools/install/bin/ga-config
  GA_CONFIG = /people/gawa722/stuff/work_nwchem/Nwchem-6.5r/src/tools/install/bin/ga-config

	CC = $(shell $(GA_CONFIG) --cc)
	CXX = mpicxx
	FC = $(shell $(GA_CONFIG) --f77)

	#GA_HOME = /home/laip/nwchem-6.3/src/tools
#	GA_HOME = /people/gawa722/stuff/nwchem6dot4_dev/Nwchem-6.5/src/tools/install
	#GA_INCLUDE = -I$(GA_HOME)/include -I../src/tce/include
#	TCE_SRC = $(NWCHEM_TOP)/src
#	GA_INCLUDE = -I$(GA_HOME)/include -I$(TCE_SRC)/tce
#	GA_LIBS = -L$(GA_HOME)/lib

	CXXINCLUDES = $(shell $(GA_CONFIG) --cppflags --cflags)
	#FINCLUDES = $(GA_INCLUDE) -I../src/tce/include 
	FINCLUDES = $(shell $(GA_CONFIG) --fflags) -I$(NWCHEM_TOP)/src/tce/include

default: libctce.a

libctce.a: variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o \
ccsd_t1.o \
ccsd_t2_12.o ccsd_t2_45.o ccsd_t2_68.o ccsd_t2_7.o \
ccsd_t.o
	ar cr libctce.a variables.o dummy.o func.o capi.o preprocess.o expression.o t_mult.o t_assign.o tensor.o \
ccsd_t1.o \
ccsd_t2_12.o ccsd_t2_45.o ccsd_t2_68.o ccsd_t2_7.o \
ccsd_t.o

clean:
	rm -f *~ *.o *.x *.a *.err *.out 

.SUFFIXES: .cc .F

.cc.o:
	$(CC) $(CXXINCLUDES) -DEXT_INT -DGFORTRAN -c $*.cc

.F.o:
	$(FC) $(FINCLUDES) -c $*.F
