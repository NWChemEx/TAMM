#include "tamm/tamm.hpp"

using namespace tamm;

static const int NUM_NK_ITER = 6;

template<typename T>
void fn() {

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

    IndexSpace IS{range(0, 10),
                  {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};

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

    Scheduler{ec}
        (res1() = 0)
        (res2() = 0)
        (res1() += R1() * R1())
        (res2() += R2() * R2())
        (res() += res1())
        (res() += res2())
      .execute();

    double res_local;
    //@todo apply sqrt() to res()
    res.get({}, &res_local);
    res_local = sqrt(res_local);


    
    Scheduler sch{ec};
    // res.get({}, &res_local);
    while(res_local > tol) {
        // 3: = ||R(T(k))||; -- ignore

        // 4: V (:; 1)   R(T(k))= ;

        sch
            (R1() /= res())
            (R2() /= res())
            (X1[0]() = R1())
            (X2[0]() = R2())
            .execute();

        const double h = 1.0e-2;
        // 5: for j = 1, 2,...maxgmiter do
        for(size_t j = 0; j < NUM_NK_ITER; j++) {
            Tensor<T> T1_tmp{a, i}, T2_tmp{a, b, i, j};
            Tensor<T> R1_tmp{a, i}, R2_tmp{a, b, i, j};
            Scheduler{}(T1_tmp() = h * X1[j] + T1())(T2_tmp() =
                                                       h * X2[j] + T2())
              .execute();
            // 6: W   R[T(k) + hV (:; j)];
            R1_tmp = ccsd_t1(T1_tmp, T2_tmp, F1, V2);
            R2_tmp = ccsd_t2(T1_tmp, T2_tmp, F1, V2);
            // 7: W   (W 􀀀 R(T(k)))=h;
            sch
                (R1_tmp() -= R1())
                (R2_tmp() -= R2())
                (R1_tmp() /= h)
                (R2_tmp() /= h)
                .execute();
            // 8: H(:; 1 : j)   V (:; 1 : j)TW;
            // 9: W   W 􀀀 V (:; 1 : j)h;
            Tensor<T> Htmp[NUM_NK_ITER][NUM_NK_ITER]; //all scalars
            
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
            //@todo do res1_tmp() = sqrt(res1_tmp)

            // 11: V (:; j + 1) = W=H(j + 1; j);
            sch
                (Htmp[j+1][j]() = res1_tmp())
                (X1[j+1]() = R1_tmp() / res1_tmp())
                (X2[j+1]() = R2_tmp() / res1_tmp())
                .execute();
            // 12: end for
        }
        // 13: Solve the the projected linear least squares problem mins kHs 􀀀
        // e1 k 
        
        //14: T(k+1) = V s; 
        
        //15: k   k + 1; 
        
        //16: Evaluate R(T(k)); 
        
        //17: end while
    } //while

    Tensor<T>::deallocate(F1,T1,V2,T2);
    Tensor<T>::allocate(res,res1,res2,res1_tmp);

    for(size_t i = 0; i < NUM_NK_ITER; i++) {
        Tensor<T>::deallocate(X1[i],X2[i]);
    }
}
