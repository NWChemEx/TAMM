include ../make.inc

OBJS = cholQR.o sisubit.o RayleighRitz.o GetEigvals.o GetResiduals.o itereig.o evalidate.o HSLanczos.o getevbnd.o parseinput.o utilities.o getshifts.o slicing.o collectevs.o sync_slices.o hsdiag.o kmeans.o

lib: $(OBJS) 
	ar rv libslice.a $(OBJS) 
	ranlib libslice.a

clean:
	rm -f *.o *.a 
