#include "TAMM_CPP_DSL.h"
#include <vector>

using Op = tamm_cpp_dsl::Operation;
using Tensor = tamm_cpp_dsl::Tensor;


int main(){
    
    std::string i,j,k,l,m;
    std::string a,b,c,d,e;
    double threshold = 1.0E-6;
    double energy = 0;
    
    Tensor tc,ta,tb;
    //tc({i,j}) = ta({i,k}) * tb({k,j});
    // tc({i,j}) += 1.0 * (ta({i,k}) * tb({k,j}));
    // tc({i,j}) += 1.0 * tb({k,j});

    Tensor D,F;

    for (int l1=0;l1<20;l1++) {
        Tensor X,T,R,Z;

        for (int l2 = 0;l2< 20;l2++){
    //         R_ia = F_ia - F_ib * X_ba - F_im * X_ma
    //    + X_ij*F_ja - X_ij * F_jb * X_ba - X_ij * F_jm * X_ma
    //    + X_ib * F_ba - X_ib * F_bc * X_ca - X_ib * F_bm * X_ma

              T({i,a}) += Z({i,a}) * R({i,a});
        }

        //Tensor Z_ij, T_ie, T_je;
        Z({i,j}) += -1.0 * (T({i,e}) * T({j,e}));

        for (int l3=0;l3<10;l3++){
		// D_ij = delta_ij + D_im * Z_mj #r_ij = D_ij - delta_ij - D_im * Z_mj = 0
		// D_ij += r_ij / (delta_ij=1 + Z_jj)
        }

        X({i,a}) += 1.0 * (D({i,m}) * T({m,a}));

        X({a,b}) += 1.0 * (X({m,a}) * T({m,b}));

        X({i,j}) += -1.0 * (T({i,e}) * X({j,e}));

        //X({i,a}) = X({a,i})

    //     	#Construct new fock matrix F(D(i+1)) from Density matrix D
	// #X_ia,X_ab,X_ij (i+1)

	// full D(i+1) = [ l_oo + x_oo | X_ov ]
    //               [ X_vo        | X_vv ]

    // #D_ij = [delta_ij + X_ij ]

    }
    

    return 0;
}