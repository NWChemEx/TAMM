#ifndef __ctce_ga_abstract_h__
#define __ctce_ga_abstract_h__

#include "ga.h"

using namespace std;

namespace ctce {
extern "C" {
	int nga_Create(int type, int ndim, int dims[], char *array_name, int chunk[]);
	long nga_Read_inc(int id, int subscript[], long inc);
	void nga_Zero(int id);
	void nga_Destroy(int id);
	void nga_Acc(int id, int lo[], int hi[], void* buf, int ld[], void* alpha);
	void nga_Get(int id, int lo[], int hi[], void* buf, int ld[]);
	void nga_NbGet(int id, int lo[], int hi[], void* buf, int ld[], ga_nbhdl_t* nbhandle);
	void nga_NbWait(ga_nbhdl_t* nbhandle);
	int ga_Nnodes();
	void ga_Dgop(double x[], int n, char *op);
	void ga_Zero(int id);
	void ga_Sync();
	void ga_Destroy(int id);
}
}

#endif
