#ifndef TAMM_TESTS_NK_HPP_
#define TAMM_TESTS_NK_HPP_

#include <Eigen/Dense>
#include "tamm/tamm.hpp"

using namespace tamm;

using EMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static const int NUM_NK_ITER = 6;

template<typename T>
void fn(ExecutionContext *ec, TiledIndexSpace TIS) {
    // IndexSpace IS{range(0, 10),
    //               {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};
    // TiledIndexSpace TIS{IS, 1};

    // 1: k = 0;
    size_t k = 0;

    TiledIndexLabel a, b, i, j, n;
    std::tie(a, b) = TIS.labels<2>("virt");
    std::tie(i, j) = TIS.labels<2>("occ");
    std::tie(n)    = TIS.labels<1>("all");

    // 2: while ||R(T(k))|| > tol do
    double tol;
    Tensor<T> F1{n, n}, V2{n, n, n, n};
    Tensor<T> T1{a, i}, T2{a, b, i, j};
    Tensor<T> R1{a, i}, R2{a, b, i, j};

    Tensor<T>::allocate(ec,F1,T1,V2,T2);

    Tensor<T> X1[NUM_NK_ITER], X2[NUM_NK_ITER];

    for(size_t i = 0; i < NUM_NK_ITER; i++) {
        X1[i] = Tensor<T>{a, i};
        X2[i] = Tensor<T>{a, b, i, j};
        Tensor<T>::allocate(ec,X1[i],X2[i]);
    }

    Tensor<T> res{}, res1{}, res2{}, res1_tmp{};
    R1 = ccsd_t1(T1, T2, F1, V2);
    R2 = ccsd_t2(T1, T2, F1, V2);

    Tensor<T>::allocate(ec,res,res1,res2,res1_tmp);

    // res_loc = ||R(T)||;
    Scheduler{ec}
        (res1() = 0)
        (res2() = 0)
        (res1() += R1() * R1())
        (res2() += R2() * R2())
        (res() += res1())
        (res() += res2())
      .execute();

    double res_local;
    res.get({}, span<T>{&res_local, 1});
    res_local = sqrt(res_local);
    res.put({}, span<T>{&res_local, 1});
    
    Scheduler sch{ec};
    while(res_local > tol) {

        // 4: X(:;0) = R(T)/res_loc;
        sch
            (R1() /= res())
            (R2() /= res())
            (X1[0]() = R1())
            (X2[0]() = R2())
            .execute();

        const double h = 1.0e-2;
        Tensor<T> Htmp[NUM_NK_ITER+1][NUM_NK_ITER]; 

        // 5: for j = 1, 2,...maxgmiter do
        for(size_t j = 0; j < NUM_NK_ITER; j++) {
            // 6: R_tmp = R[T + h*X(:,)];
            Tensor<T> T1_tmp{a, i}, T2_tmp{a, b, i, j};
            Tensor<T> R1_tmp{a, i}, R2_tmp{a, b, i, j};
            Scheduler{}(T1_tmp() = h * X1[j] + T1())(T2_tmp() =
                                                       h * X2[j] + T2())
              .execute();
            R1_tmp = ccsd_t1(T1_tmp, T2_tmp, F1, V2);
            R2_tmp = ccsd_t2(T1_tmp, T2_tmp, F1, V2);
            // 7: R_tmp = (R_tmp - R(T))/h;
            sch
                (R1_tmp() -= R1())
                (R2_tmp() -= R2())
                (R1_tmp() /= h)
                (R2_tmp() /= h)
                .execute();
            // 8: H(:; 1 : j) = transpose(X (:;1 :j))* W;
            // 9: R_tmp =  R_tmp - X (:;1 :j)*H(:,1:j);
            
            for(size_t j1=0; j1 < j; j1++) {
                sch
                (Htmp[j1][j] = X1[j1]() * R1_tmp())
                (Htmp[j1][j] += X2[j1]() * R2_tmp())
                (R1_tmp() -= Htmp[j1][j]() * X1[j1])
                (R2_tmp() -= Htmp[j1][j]() * X2[j1]);
            }
            sch.execute();
            // 10: H(j + 1; j) = ||W||;
            sch
            (res1_tmp() = R1_tmp() * R1_tmp())
            (res1_tmp() += R2_tmp() * R2_tmp())
            .execute();
            //res1_tmp() = sqrt(res1_tmp)
            res1_tmp.get({}, span<T>{&res_local, 1});
            res_local = sqrt(res_local);
            res1_tmp.put({}, span<T>{&res_local, 1});

            // 11: X(:;j+1) = R_tmp/H(j + 1; j);
            sch
                (Htmp[j+1][j]() = res1_tmp())
                // if (j < NUM_NK_ITER) {
                (X1[j+1]() = R1_tmp() / res1_tmp())
                (X2[j+1]() = R2_tmp() / res1_tmp())
                //}
                .execute();
            // 12: end for
        }
        // @todo 13: Solve the the projected linear least squares problem mins kHs 
        // e1 k 
        EMatrix Htmp_eigen(NUM_NK_ITER+1,NUM_NK_ITER);
        EMatrix s(NUM_NK_ITER+1);
        Htmp_eigen.setZero();
        //@todo copy Htmp to Htmp_eigen
        //@call the eigen least squares solver with the result in s

        //14: T = X*s; 
        {
            Scheduler sch;
            for(size_t j=0; j<NUM_NK_ITER; j++) {
                sch
                 (T1() += s[j] * X1[j])
                 (T2() += s[j] * X2[j]);
            }
            sch.execute();
        }
        //15: k   k + 1;  --ignore
        //16: Evaluate R(T(k)); 
        R1 = ccsd_t1(T1, T2, F1, V2);
        R2 = ccsd_t2(T1, T2, F1, V2);
    
        Scheduler{ec}
            (res1() = 0)
            (res2() = 0)
            (res1() += R1() * R1())
            (res2() += R2() * R2())
            (res() += res1())
            (res() += res2())
        .execute();
    
        res.get({}, span<T>{&res_local, 1});
        res_local = sqrt(res_local);
        res.put({}, span<T>{&res_local, 1});
        //17: end while
    } //while

    Tensor<T>::deallocate(F1,T1,V2,T2);
    Tensor<T>::allocate(res,res1,res2,res1_tmp);

    for(size_t i = 0; i < NUM_NK_ITER; i++) {
        Tensor<T>::deallocate(X1[i],X2[i]);
    }
}

#endif
