## Local HF - Interface

```c++
TEST_CASE("Sample code for Local HF") {
    
    // TAMM Scheduler construction
    auto ec = tamm::make_execution_context();
    Scheduler sch{ec};

    // Dummy TiledIndexSpaces
    TiledIndexSpace TAO{IndexSpace{range(10)}};
    TiledIndexSpace TMO{IndexSpace{range(10)}};

    // Local SCF TAMM Pseudo-code
    
    // Input dense C tensor
    Tensor LMO{TAO, TMO};  //dense

    //LMO_domain(): chooses AOs i -> mu 
    auto lmo_dep_map = LMO_domain();

    // TiledIndexSpace lmo_domain{mu(i)}; //construct using explicit loop
    TiledIndexSpace lmo_domain{TAO, {TMO}, lmo_dep_map}; //construct using explicit loop
    
    //LMO_renormalize() {
        auto [i] = TMO.labels<1>("all");
        auto [mu, nu] = lmo_domain.labels<2>("all");
        auto [mu_p] = TAO.labels<1>("all");

        Tensor S_A{i, mu(i), mu(i)};
        Tensor S_v{i, mu_p, mu(i)};
        Tensor C{i, mu_p};   //column of LMO

        //solved using Eigen

        // Sparsified LMO 
        Tensor LMO_renorm{mu(i), i}; //sparsified LMO
        
        sch
        .allocate(LMO_renorm)
            (LMO_renorm(mu(i), i) = LMO(mu(i), i))
        .execute();
    // }


    //AO_domain(): constructs ao->ao index space
    auto ao_screen_dep_map = AO_domain();

    // TiledIndexSpace ao_int_screening{nu(mu)}; //ao->ao
    TiledIndexSpace ao_int_screening{TAO, {TAO}, ao_screen_dep_map};

    //chain_maps(): compose lmo->ao and ao->ao
    auto [nu_p] = ao_int_screening.labels<1>("all");

    // TiledIndexSpace ao_domain{nu(i)}; //mo->ao
    // compose using labels
    auto ao_domain = compose_tis(mu(i), nu_p(mu)); // nu(i)
    // compose using TiledIndexSpaces
    // auto ao_domain = compose_tis(lmo_domain, ao_int_screening);

    //fitting domain
    // IndexSpace fb; //fitting basis. this is already available and used as input

    auto lmo_to_fit_dep_map = fitting_domain();

    // Output:
    // TiledIndexSpace lmo_to_fit{A(i)}; // mo-> fitting basis
    TiledIndexSpace lmo_to_fit{TAO, {TMO}, lmo_to_fit_dep_map}; //mo->fitting basis

    //continuing with build_K. first contraction “transformation step”

    // TiledIndexSpace ao_to_lmo{i(mu)}; // 
    // invert using labels
    auto ao_to_lmo= invert_tis(mu(i)); // i(mu)
    // invert using TiledIndexSpaces
    // auto ao_to_lmo= invert_tis(lmo_domain);

    // IndexLabel i(mu);//ao_to_lmo
    auto [A, B] = lmo_to_fit.labels<2>("all");

    // TiledIndexSpace ops    
    auto fit_to_lmo = invert_tis(A(i));               // i(A)
    auto fit_to_ao  = compose_tis(fit_to_lmo, mu(i)); // mu(A)
    auto fit_to_fit = compose_tis(fit_to_lmo, A(i));  // B(A)

    auto [B_p] = fit_to_fit.labels<1>("all");

    // Input X (tensor with lamda function that calls libint)
    Tensor X{A(i), mu(i), nu(i)} // internally project on i ?
    // input J
    Tensor J{A, B_p(A)};

    // results
    Tensor Q{A(i), mu(i), i};
    Tensor QB{B(i), mu(i), i};
    Tensor K{mu(i), nu(i)};

    sch.allocate(Q, QB, K);
    // foreach i value:
    for(Index i_val : TMO){
        Tensor J_i{A(i_val), B(i_val)};
        Tensor G_i_inv{A(i_val), B(i_val)};
        sch
        .allocate(J_i, G_i_inv)         // Q: how to allocate within a loop?
            (Q(A(i_val), mu(i_val), i_val) = X(A(i_val), mu(i_val), nu(i_val)) * C(nu(i_val), i_val))
            (J_i(A(i_val), B(i_val)) = J(A(i_val), B(i_val)))
        .execute();

        G_i_inv = invert_tensor(cholesky(J_i));
        
        sch
            (QB(B(i_val), mu(i_val), i_val) += G_i_inv(B(i_val), A(i_val)) * Q(A(i_val), mu(i_val), i_val))
            // (K(mu, nu(mu)) += QB(A(i), mu(i), i) * QB(A(i), nu(i), i)) //nu(mu) is a dependent representation of the sparsity
            (K(mu(i_val), nu(i_val)) += QB(A(i_val), mu(i_val), i_val) * QB(A(i_val), nu(i_val), i_val))
        .deallocate(J_i, G_i_inv)
        .execute();
    }

}
```