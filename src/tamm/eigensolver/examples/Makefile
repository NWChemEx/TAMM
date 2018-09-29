include ../make.inc

runsisubit: runsisubit.o
	${CXX} ${CXXFLAGS} -o runsisubit runsisubit.o

runslicing: runslicing.o ../src/libslice.a
	${CXX} ${CXXFLAGS} -o runslicing runslicing.o ../src/libslice.a $(MKL)

runscf1: runscf1.o ../src/libslice.a 
	$(CXX) ${CXXFLAGS} -o runscf1 runscf1.o ../src/libslice.a $(MKL)

runmockscf: runmockscf.o ../src/libslice.a 
	$(CXX) ${CXXFLAGS} -o runmockscf runmockscf.o ../src/libslice.a $(MKL)

runsqrt: runsqrt.o ../src/libslice.a 
	$(CXX) ${CXXFLAGS} -o runsqrt runsqrt.o ../src/libslice.a $(MKL)

clean:
	rm -f runsisubit runslicing *.o *.a runscf1 runmockscf runsqrt
