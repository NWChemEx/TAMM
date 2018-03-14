// #include "ao.h"
// #include "boundvec.h"
// #include "distribution.h"
// #include "index_space.h"
// #include "labeled_tensor.h"
// #include "memory_manager.h"
// #include "memory_manager_local.h"
// #include "mso.h"
// #include "ops.h"
// #include "perm_symmetry.h"
// #include "proc_group.h"
// #include "scheduler.h"
// #include "strong_num.h"
// #include "sub_ao_mso.h"
// #include "tensor.h"
// #include "tensor_base.h"
// #include "types.h"

#include "index_space_fragment.h"

using namespace tammy;

int main() {

  IndexSpaceFragment isf_occ(0, 99);
  IndexSpaceFragment isf_vir(100, 199);

  IndexSpaceX mso = {isf_occ, isf_vir};

  TiledIndexSpace t_mso(mso, 10);

  const TiledIndexRange& t_mso_occ = t_mso.range("occ");


  return 1;
}