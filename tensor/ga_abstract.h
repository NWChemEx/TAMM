#ifndef __ctce_ga_abstract_h__
#define __ctce_ga_abstract_h__

#include <stdint.h>
#include <assert.h>
#include <stdio.h>

using namespace std;

namespace ctce
{
  namespace gmem
  {
    struct Handle
    {
      Handle(){}
      bool valid();
      
      //FIXME: Make explicit for c++11
      Handle(uint64_t conversion);
      operator uint64_t () const;

      uint64_t value;
    };

    struct Wait_Handle
    {
      uint64_t value;
    };

    extern Handle NULL_HANDLE;

    enum Types
    {
      Int,
      Double
    };

    enum Operation
    {
      Plus=0,
      Multiply,
      Max,
      Min,
      AbsMax,
      AbsMin
    };

    Handle create(Types type, int size, char * name );
    uint64_t ranks();
    int64_t atomic_fetch_add(Handle handle, int pos, long amount);
    void zero(Handle handle);
    void sync();
    void destroy(Handle handle);
    void get(Handle handle, void * buf, int start, int stop );
    void get(Handle handle, void * buf, int start, int stop,  Wait_Handle & wait  );
    void acc(Handle handle, void * buf, int start, int stop);
    void op(double x[], int n, Operation op);
    void wait(Wait_Handle & wait);
    void destroy(Handle handle);
  };
}

#endif
