extern "C" {
#include "ctce_parser.h"
};


#include "variables.h"
#include <iostream>
#include "tensor.h"
#include "t_mult.h"
#include "t_assign.h"
#include "input.h"
#include "corf.h"
#include "equations.h"


namespace ctce {

#define CTCE_EQ_PATH "/home/sriram/code/ctce/tensor/eqs/"

  void cc2_t1_equations(ctce::Equations &eqs) {
    ::Equations peqs;
    ctce_parser("/home/gawa722/Cascade/nwchem/repo_code/ctce/ctce_parser/transform_input/generated/cc2_t1.eq.lvl", &peqs);
    parser_eqs_to_ctce_eqs(&peqs, eqs);
  }

}; /*ctce*/

