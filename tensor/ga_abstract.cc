#include "ga_abstract.h"

using namespace std;

namespace ctce {

extern "C" {
	
  int nga_Create(int type, int ndim, int dims[], char *array_name, int chunk[])
	{
    return NGA_Create( type, ndim, dims, array_name, chunk);
	}

	long nga_Read_inc(int id, int subscript[], long inc)
	{
		return NGA_Read_inc( id, subscript, inc);
	}
	
	void nga_Zero(int id)
	{
		NGA_Zero(id);
	}
	
	void nga_Destroy(int id)
	{
		NGA_Destroy(id);
	}
	
	void nga_Acc(int id, int lo[], int hi[], void* buf, int ld[], void* alpha)
	{
		NGA_Acc( id, lo, hi, buf, ld, alpha);
	}
	
	void nga_Get(int id, int lo[], int hi[], void* buf, int ld[])
	{
	    NGA_Get(id, lo, hi, buf, ld);
	}
	
	void nga_NbGet(int id, int lo[], int hi[], void* buf, int ld[], ga_nbhdl_t* nbhandle)
	{
		NGA_NbGet(id,lo, hi, buf, ld, nbhandle);
	}
	
	void nga_NbWait(ga_nbhdl_t* nbhandle)
	{
		NGA_NbWait(nbhandle);
	}
	
	int ga_Nnodes()
	{
		return GA_Nnodes();
	}
	void ga_Dgop(double x[], int n, char *op)
	{
		GA_Dgop(x, n, op);
	}
	void ga_Zero(int id)
	{
		GA_Zero(id);
	}
	void ga_Sync()
	{
		GA_Sync();
	}
	void ga_Destroy(int id)
	{
		GA_Destroy(id);
	}
}

}

