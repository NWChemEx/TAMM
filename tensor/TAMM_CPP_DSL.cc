#include "TAMM_CPP_DSL.h"
#include <vector>

using Op = tamm_cpp_dsl::Operation;
using Tensor = tamm_cpp_dsl::Tensor;


int main(){
    
    std::string i,j,k,l,m;
    std::string a,b,c,d,e;
    // double threshold = 1.0E-6;
    // double energy = 0;
    
    Tensor D,F;

    for (int l1=0;l1<20;l1++) {
        Tensor X,T,R,Z, delta;

        for (int l2 = 0;l2< 20;l2++){
          Tensor tmp1, tmp2, tmp3;

          R({i,a}) += 1.0 * F({i,a});
          R({i,a}) += -1.0 * (F({i,b}) * X({b,a}));
          R({i,a}) += -1.0 * (F({i,m}) * X({m,a}));
          R({i,a}) += X({i,j}) * F({j,a});
          R({i,a}) += -1.0 * (X({i,j}) * F({j,b}));
          R({i,a}) += 1.0 * X({b,a});
          
          tmp1({j,a}) += 1.0 * (F({j,m}) * X({m,a}));
          R({i,a}) += -1.0 * (X({i,j}) * tmp1({j,a}));
          R({i,a}) += 1.0 * (X({i,b}) * F({b,a}));
          
          tmp2({b,a}) += 1.0 * (F({b,c}) * X({c,a}));
          R({i,a}) += -1.0 * (X({i,b}) * tmp2({b,a}));
          tmp3({b,a}) += 1.0 * (F({b,m}) * X({m,a}));
          R({i,a}) += -1.0 * (X({i,b}) * tmp3({b,a}));


          T({i,a}) += Z({i,a}) * R({i,a});  // div (f_aa(i) - f_ii(i))  
        }

        Z({i,j}) += -1.0 * (T({i,e}) * T({j,e}));

        for (int l3=0;l3<10;l3++){
           //r_ij = D_ij - delta_ij - D_im * Z_mj = 0
		   D({i,j}) += 1.0 * delta({i,j});
           D({i,j}) += 1.0 * (D({i,m}) * Z({m,j})); 
		   // D_ij += r_ij / (delta_ij=1 + Z_jj)
        }

        X({i,a}) += 1.0 * (D({i,m}) * T({m,a}));

        X({a,b}) += 1.0 * (X({m,a}) * T({m,b}));

        X({i,j}) += -1.0 * (T({i,e}) * X({j,e}));

        //X({i,a}) = X({a,i})

    // #Construct new fock matrix F(D(i+1)) from Density matrix D
	// #X_ia,X_ab,X_ij (i+1)

	// full D(i+1) = [ l_oo + x_oo | X_ov ]
    //               [ X_vo        | X_vv ]

    // #D_ij = [delta_ij + X_ij ]

    }
    

    return 0;
}