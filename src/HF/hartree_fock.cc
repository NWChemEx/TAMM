#include "tamm/tamm.h"

using namespace tamm;

template<typename T>
void hartree_fock(ExecutionContext& ec,const TiledIndexSpace& AO,
const Tensor<T>& C, Tensor<T>& F){

    Scheduler sch{ec};

    const TiledIndexSpace& N = AO("all");
    const TiledIndexSpace& O = AO("occ");

    TiledIndexLabel i,j,k;
    TiledIndexLabel io,jo,ko;

    std::tie(i,j,k)=AO.labels<3>("all");
    std::tie(io,jo,ko)=AO.labels<3>("occ");

    Tensor<T> S{N,N, one_body_overlap_integral_lambda};
    Tensor<T> T{N,N, one_body_kinetic_integral_lambda};
    Tensor<T> V{N,N, one_body_nuclear_integral_lambda};

    Tensor<T> H{N,N};

    sch.allocate(H,T,V)
    (H(i,j) = T(i,j))
    (H(i,j) += V(i,j))
    .execute();

    Tensor<T> D{N,N};

    //sch?
    compute_soad(atoms,D);

    const int maxiter{100};
    const double conv{1e-12};
    int iter{0};

    Tensor<T> ehf, ediff, rmsd;
    Tensor<T> eps{N,N};

    do{
        ++iter;
        //Save a copy of energy and the density
        Tensor<T> ehf_last;
        Tensor<T> D_last{N,N};

        sch.allocate(ehf_last,D_last)
        (ehf_last() = ehf)
        (D_last(i,j) = D(i,j)).execute();

        // build a new Fock matrix
        sch
        (F(i,j) = H(i,j)).execute();

        compute_2body_fock(shells,D,F); //accumulate into F

        // solve F C = e S C
        //Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
        //eps = gen_eig_solver(F,S).eigenvalues();
        //C = gen_eig_solver(F,S).eigenvectors();
        std::tie(C, eps) = eigen_solve(F, S);
        // compute density, D = C(occ) . C(occ)T
        //C_occ(ao,bo) = C(ao,bo); //C.leftCols(ndocc);
        //C_occ_transpose(ao,bo) = C_occ(bo,ao);
        
        sch
        (D(io, jo) = C(io, ko) * C(ko, jo)).execute();

        Tensor<T> tmp1{N,N}, tmp2{N,N};
        // compute HF energy
        //ehf+= D(i,j) * (H(i,j)+F(i,j))

        sch.allocate(tmp1,tmp2)
        (ehf() = 0.0)
        (tmp1(i,j) = H(i,j))
        (tmp1(i,j) += F(i,j))
        (ehf() = D(i,j) * tmp1(i,j))
        //compute difference with last iteration
        (ediff() = ehf())
        (ediff() = -1.0 * ehf_last())
        (tmp2(i,j) = D(i,j))
        (tmp2(i,j) += -1.0 * D_last(i,j))
        .execute();
        //??
        norm(tmp2,rmsd); //rmsd() = tmp2(a,b).norm();
        
        sch(rmsd() = tmp2(i,j) * tmp2(i,j))
        .execute();
	//e.g.:Tensor<T> rmsd_local{AllocationModel::replicated};
	//e.g.:rmsd_local(a) = rmsd(a);
	//e.g.: rmsd(a) +=  rmsd_local(a);
	//TODO: only put rmsd_local in process 0 to rmsd
  } while (((fabs(get_scalar(ediff) > conv) || (fabs(get_scalar(rmsd)) > conv)) && (iter < maxiter));


}