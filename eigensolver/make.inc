MKLROOT = /home/cyang/MKL/mkl
MKL = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

CXX = mpic++
CXXFLAGS = -g -DEIGEN_USE_MKL_ALL -DUSE_MPI -I ../include -I /home/cyang/eigen -I${MKLROOT}/include -fopenmp

.cpp.o: 
	${CXX} ${CXXFLAGS} -c $< 
