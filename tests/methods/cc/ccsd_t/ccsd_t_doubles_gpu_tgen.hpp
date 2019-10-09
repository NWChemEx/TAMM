
#ifndef CCSD_T_DOUBLES_GPU_TGEN_HPP_
#define CCSD_T_DOUBLES_GPU_TGEN_HPP_

#include <algorithm>

// template <typename Arg, typename... Args>
// void dprint(Arg&& arg, Args&&... args)
// {
//     cout << std::forward<Arg>(arg);
//     ((cout << ',' << std::forward<Args>(args)), ...);
//     cout << "\n";
// }

void sd_t_d1_all_cuda_tgen(size_t *sizes,
				double* t3, 
				double* t2_all, size_t size_t2_all,
				double* v2_all, size_t size_v2_all,
				std::vector<bool> &p_kernel,
			size_t opt_register_transpose);

void sd_t_d2_all_cuda_tgen(size_t *sizes,
				double* t3, 
				double* t2_all, size_t size_t2_all,
				double* v2_all, size_t size_v2_all,
				std::vector<bool> &p_kernel,
			size_t opt_register_transpose);

template<typename T>
void ccsd_t_doubles_gpu_tgen(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const Index noab, const Index nvab,
                   std::vector<int>& k_spin,
                   std::vector<T>& a_c,
                   Tensor<T>& d_t2, //d_a
                   Tensor<T>& d_v2, //d_b
                   std::vector<T>& p_evl_sorted,
                   std::vector<size_t>& k_range,
                   size_t t_h1b, size_t t_h2b, size_t t_h3b,
                   size_t t_p4b, size_t t_p5b, size_t t_p6b,
                   std::vector<T>& k_abuf1, std::vector<T>& k_bbuf1,
                   std::vector<T>& k_abuf2, std::vector<T>& k_bbuf2,
                   int usedevice,size_t& kcalls, size_t& kcalls_fused, size_t& kcalls_pfused) {

    // initmemmodule();

    size_t abuf_size1 = k_abuf1.size();
    size_t bbuf_size1 = k_bbuf1.size();
    size_t abuf_size2 = k_abuf2.size();
    size_t bbuf_size2 = k_bbuf2.size();

    Eigen::Matrix<size_t, 9,6, Eigen::RowMajor> a3;
    a3.setZero();

    a3(0,0)=t_p4b;
    a3(0,1)=t_p5b;
    a3(0,2)=t_p6b;
    a3(0,3)=t_h1b;
    a3(0,4)=t_h2b;
    a3(0,5)=t_h3b;

    a3(1,0)=t_p4b;
    a3(1,1)=t_p5b;
    a3(1,2)=t_p6b;
    a3(1,3)=t_h2b;
    a3(1,4)=t_h1b;
    a3(1,5)=t_h3b;

    a3(2,0)=t_p4b;
    a3(2,1)=t_p5b;
    a3(2,2)=t_p6b;
    a3(2,3)=t_h3b;
    a3(2,4)=t_h1b;
    a3(2,5)=t_h2b;

    a3(3,0)=t_p5b;
    a3(3,1)=t_p6b;
    a3(3,2)=t_p4b;
    a3(3,3)=t_h1b;
    a3(3,4)=t_h2b;
    a3(3,5)=t_h3b;

    a3(4,0)=t_p5b;
    a3(4,1)=t_p6b;
    a3(4,2)=t_p4b;
    a3(4,3)=t_h2b;
    a3(4,4)=t_h1b;
    a3(4,5)=t_h3b;

    a3(5,0)=t_p5b;
    a3(5,1)=t_p6b;
    a3(5,2)=t_p4b;
    a3(5,3)=t_h3b;
    a3(5,4)=t_h1b;
    a3(5,5)=t_h2b;

    a3(6,0)=t_p4b;
    a3(6,1)=t_p6b;
    a3(6,2)=t_p5b;
    a3(6,3)=t_h1b;
    a3(6,4)=t_h2b;
    a3(6,5)=t_h3b;

    a3(7,0)=t_p4b;
    a3(7,1)=t_p6b;
    a3(7,2)=t_p5b;
    a3(7,3)=t_h2b;
    a3(7,4)=t_h1b;
    a3(7,5)=t_h3b;

    a3(8,0)=t_p4b;
    a3(8,1)=t_p6b;
    a3(8,2)=t_p5b;
    a3(8,3)=t_h3b;
    a3(8,4)=t_h1b;
    a3(8,5)=t_h2b;

    // auto notset=1;

    for (auto ia6=0; ia6<8; ia6++){
      if(a3(ia6,0) != 0) {
      for (auto ja6=ia6+1;ja6<9;ja6++) { //TODO: ja6 start ?
      if((a3(ia6,0) == a3(ja6,0)) && (a3(ia6,1) == a3(ja6,1))
       && (a3(ia6,2) == a3(ja6,2)) && (a3(ia6,3) == a3(ja6,3))
       && (a3(ia6,4) == a3(ja6,4)) && (a3(ia6,5) == a3(ja6,5)))
       {
        a3(ja6,0)=0;
        a3(ja6,1)=0;
        a3(ja6,2)=0;
        a3(ja6,3)=0;
        a3(ja6,4)=0;
        a3(ja6,5)=0;
      }
      } 
      }
    }

    Index p4b,p5b,p6b,h1b,h2b,h3b;

    size_t max_dima = abuf_size1 / 9;
    size_t max_dimb = bbuf_size1 / 9;
    

  for (Index h7b=0;h7b<noab;h7b++){

    std::vector<bool> sd_t_d1_exec(9,false);
    std::vector<size_t> sd_t_d1_args(63);

    for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    


    if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0){ 
      if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12){
         if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         == k_spin[h1b]+k_spin[h2b]+k_spin[h3b]) {

          // for (Index h7b=0;h7b<noab;h7b++){

            if(k_spin[p4b]+k_spin[p5b]
              == k_spin[h1b]+k_spin[h7b]) {

                size_t dim_common = k_range[h7b];
                size_t dima_sort = k_range[p4b]*k_range[p5b]*k_range[h1b];
                size_t dima = dim_common*dima_sort;
                size_t dimb_sort = k_range[p6b]*k_range[h2b]*k_range[h3b];
                size_t dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);


                  if(h7b<h1b) {

                    d_t2.get({p4b-noab,p5b-noab,h7b,h1b},k_a); //h1b,h7b,p5b-noab,p4b-noab
                    int perm[4]={3,1,0,2}; //3,1,0,2
                    int size[4]={k_range[p4b],k_range[p5b],k_range[h7b],k_range[h1b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(h1b<=h7b){

                    d_t2.get({p4b-noab,p5b-noab,h1b,h7b},k_a); //h7b,h1b,p5b-noab,p4b-noab
                    int perm[4]={2,1,0,3}; //2,1,0,3
                    // int size[4]={k_range[p4b],k_range[p5b],k_range[h1b],k_range[h7b]};
                    int size[4]={k_range[p4b],k_range[p5b],k_range[h1b],k_range[h7b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }


    std::vector<T> k_b_sort(dimb);
    if(h7b <= p6b){
      d_v2.get({h7b,p6b,h2b,h3b},k_b_sort); //h3b,h2b,p6b,h7b

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
     {
       sd_t_d1_args[0] = k_range[h1b];
       sd_t_d1_args[1] = k_range[h2b];
       sd_t_d1_args[2] = k_range[h3b];
       sd_t_d1_args[3] = k_range[h7b];
       sd_t_d1_args[4] = k_range[p4b];
       sd_t_d1_args[5] = k_range[p5b];
       sd_t_d1_args[6] = k_range[p6b];
       
       sd_t_d1_exec[0] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin());
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin());
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
       //dprint(2);
       sd_t_d1_args[7] = k_range[h1b];
       sd_t_d1_args[8] = k_range[h2b];
       sd_t_d1_args[9] = k_range[h3b];
       sd_t_d1_args[10] = k_range[h7b];
       sd_t_d1_args[11] = k_range[p4b];
       sd_t_d1_args[12] = k_range[p5b];
       sd_t_d1_args[13] = k_range[p6b];
       
       sd_t_d1_exec[1] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 1*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 1*max_dimb);
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
     {
      // //dprint(3);
       sd_t_d1_args[14] = k_range[h1b];
       sd_t_d1_args[15] = k_range[h2b];
       sd_t_d1_args[16] = k_range[h3b];
       sd_t_d1_args[17] = k_range[h7b];
       sd_t_d1_args[18] = k_range[p4b];
       sd_t_d1_args[19] = k_range[p5b];
       sd_t_d1_args[20] = k_range[p6b];
       
       sd_t_d1_exec[2] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 2*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 2*max_dimb);
     }

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////dprint(4);
       sd_t_d1_args[21] = k_range[h1b];
       sd_t_d1_args[22] = k_range[h2b];
       sd_t_d1_args[23] = k_range[h3b];
       sd_t_d1_args[24] = k_range[h7b];
       sd_t_d1_args[25] = k_range[p4b];
       sd_t_d1_args[26] = k_range[p5b];
       sd_t_d1_args[27] = k_range[p6b];
       
       sd_t_d1_exec[3] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 3*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 3*max_dimb);
      }

    if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////dprint(5);
       sd_t_d1_args[28] = k_range[h1b];
       sd_t_d1_args[29] = k_range[h2b];
       sd_t_d1_args[30] = k_range[h3b];
       sd_t_d1_args[31] = k_range[h7b];
       sd_t_d1_args[32] = k_range[p4b];
       sd_t_d1_args[33] = k_range[p5b];
       sd_t_d1_args[34] = k_range[p6b];
       
       sd_t_d1_exec[4] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 4*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 4*max_dimb);
     }
  
   if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
      // //dprint(6);
       sd_t_d1_args[35] = k_range[h1b];
       sd_t_d1_args[36] = k_range[h2b];
       sd_t_d1_args[37] = k_range[h3b];
       sd_t_d1_args[38] = k_range[h7b];
       sd_t_d1_args[39] = k_range[p4b];
       sd_t_d1_args[40] = k_range[p5b];
       sd_t_d1_args[41] = k_range[p6b];
       
       sd_t_d1_exec[5] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 5*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 5*max_dimb);
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        ////dprint(7);
       sd_t_d1_args[42] = k_range[h1b];
       sd_t_d1_args[43] = k_range[h2b];
       sd_t_d1_args[44] = k_range[h3b];
       sd_t_d1_args[45] = k_range[h7b];
       sd_t_d1_args[46] = k_range[p4b];
       sd_t_d1_args[47] = k_range[p5b];
       sd_t_d1_args[48] = k_range[p6b];
       
       sd_t_d1_exec[6] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 6*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 6*max_dimb);
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        ////dprint(8);
       sd_t_d1_args[49] = k_range[h1b];
       sd_t_d1_args[50] = k_range[h2b];
       sd_t_d1_args[51] = k_range[h3b];
       sd_t_d1_args[52] = k_range[h7b];
       sd_t_d1_args[53] = k_range[p4b];
       sd_t_d1_args[54] = k_range[p5b];
       sd_t_d1_args[55] = k_range[p6b];
       
       sd_t_d1_exec[7] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 7*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 7*max_dimb);
     }

    if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
        ////dprint(9);
       sd_t_d1_args[56] = k_range[h1b];
       sd_t_d1_args[57] = k_range[h2b];
       sd_t_d1_args[58] = k_range[h3b];
       sd_t_d1_args[59] = k_range[h7b];
       sd_t_d1_args[60] = k_range[p4b];
       sd_t_d1_args[61] = k_range[p5b];
       sd_t_d1_args[62] = k_range[p6b];
       
       sd_t_d1_exec[8] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf1.begin() + 8*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf1.begin() + 8*max_dimb);
     }
    }
     }}}
    }}
    
    } //end ia6

    if(std::all_of(sd_t_d1_exec.begin(),sd_t_d1_exec.end(), [](bool x){return x;})) kcalls_fused++;
    int npfused = std::count_if(sd_t_d1_exec.begin(), sd_t_d1_exec.end(), [](bool x){ return x; });
    if(npfused > 1 && npfused < 9) kcalls_pfused++;

    if(std::any_of(sd_t_d1_exec.begin(),sd_t_d1_exec.end(), [](bool x){return x;})) {
            kcalls++;
            sd_t_d1_all_cuda_tgen(&sd_t_d1_args[0], &a_c[0],
                      &k_abuf1[0], abuf_size1,
                      &k_bbuf1[0], bbuf_size1,
                      sd_t_d1_exec, 1);
    }


    } //end h7b

    //--------------------------------BEGIN DOUBLES-2----------------------------//

  #if 1
    a3(0,0)=t_p4b;
    a3(0,1)=t_p5b;
    a3(0,2)=t_p6b;
    a3(0,3)=t_h1b;
    a3(0,4)=t_h2b;
    a3(0,5)=t_h3b;

    a3(1,0)=t_p4b;
    a3(1,1)=t_p5b;
    a3(1,2)=t_p6b;
    a3(1,3)=t_h2b;
    a3(1,4)=t_h3b;
    a3(1,5)=t_h1b;

    a3(2,0)=t_p4b;
    a3(2,1)=t_p5b;
    a3(2,2)=t_p6b;
    a3(2,3)=t_h1b;
    a3(2,4)=t_h3b;
    a3(2,5)=t_h2b;

    a3(3,0)=t_p5b;
    a3(3,1)=t_p4b;
    a3(3,2)=t_p6b;
    a3(3,3)=t_h1b;
    a3(3,4)=t_h2b;
    a3(3,5)=t_h3b;

    a3(4,0)=t_p5b;
    a3(4,1)=t_p4b;
    a3(4,2)=t_p6b;
    a3(4,3)=t_h2b;
    a3(4,4)=t_h3b;
    a3(4,5)=t_h1b;

    a3(5,0)=t_p5b;
    a3(5,1)=t_p4b;
    a3(5,2)=t_p6b;
    a3(5,3)=t_h1b;
    a3(5,4)=t_h3b;
    a3(5,5)=t_h2b;

    a3(6,0)=t_p6b;
    a3(6,1)=t_p4b;
    a3(6,2)=t_p5b;
    a3(6,3)=t_h1b;
    a3(6,4)=t_h2b;
    a3(6,5)=t_h3b;

    a3(7,0)=t_p6b;
    a3(7,1)=t_p4b;
    a3(7,2)=t_p5b;
    a3(7,3)=t_h2b;
    a3(7,4)=t_h3b;
    a3(7,5)=t_h1b;

    a3(8,0)=t_p6b;
    a3(8,1)=t_p4b;
    a3(8,2)=t_p5b;
    a3(8,3)=t_h1b;
    a3(8,4)=t_h3b;
    a3(8,5)=t_h2b;


    for (auto ia6=0; ia6<8; ia6++){
      if(a3(ia6,0) != 0) {
      for (auto ja6=ia6+1;ja6<9;ja6++) { //TODO: ja6 start ?
      if((a3(ia6,0) == a3(ja6,0)) && (a3(ia6,1) == a3(ja6,1))
       && (a3(ia6,2) == a3(ja6,2)) && (a3(ia6,3) == a3(ja6,3))
       && (a3(ia6,4) == a3(ja6,4)) && (a3(ia6,5) == a3(ja6,5)))
       {
        a3(ja6,0)=0;
        a3(ja6,1)=0;
        a3(ja6,2)=0;
        a3(ja6,3)=0;
        a3(ja6,4)=0;
        a3(ja6,5)=0;
      }
      } 
      }
    }

    max_dima = abuf_size2 / 9;
    max_dimb = bbuf_size2 / 9;

    // Index p4b,p5b,p6b,h1b,h2b,h3b;

    for (Index p7b=noab;p7b<noab+nvab;p7b++){

      std::vector<bool> sd_t_d2_exec(9,false);
      std::vector<size_t> sd_t_d2_args(63);
      
    for (auto ia6=0; ia6<9; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    
      if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0){ 
      if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         +k_spin[h1b]+k_spin[h2b]+k_spin[h3b]!=12){
         if(k_spin[p4b]+k_spin[p5b]+k_spin[p6b]
         == k_spin[h1b]+k_spin[h2b]+k_spin[h3b]) {

          //  size_t dimc=k_range[p4b]*k_range[p5b]*k_range[p6b]*
          //            k_range[h1b]*k_range[h2b]*k_range[h3b];

          // for (Index p7b=noab;p7b<noab+nvab;p7b++){

            if(k_spin[p4b]+k_spin[p7b]
              == k_spin[h1b]+k_spin[h2b]) {

                size_t dim_common = k_range[p7b];
                size_t dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
                size_t dima = dim_common*dima_sort;
                size_t dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
                size_t dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);

                  if(p7b<p4b) {

                    d_t2.get({p7b-noab,p4b-noab,h1b,h2b},k_a); //h2b,h1b,p4b-noab,p7b-noab
                    // for (auto x=0;x<dima;x++) k_a_sort[x] = -1 * k_a[x];
                    int perm[4]={3,2,1,0};
                    int size[4]={k_range[p7b],k_range[p4b],k_range[h1b],k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, -1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }
                  if(p4b<=p7b) {

                    d_t2.get({p4b-noab,p7b-noab,h1b,h2b},k_a); //h2b,h1b,p7b-noab,p4b-noab
                    int perm[4]={3,2,0,1}; //0,1,3,2
                    int size[4]={k_range[p4b],k_range[p7b],k_range[h1b],k_range[h2b]};
                    
                    auto plan = hptt::create_plan
                    (perm, 4, 1.0, &k_a[0], size, NULL, 0, &k_a_sort[0],
                        NULL, hptt::ESTIMATE, 1, NULL, true);
                    plan->execute();
                  }

    std::vector<T> k_b_sort(dimb);
    if(h3b <= p7b){
      d_v2.get({p5b,p6b,h3b,p7b},k_b_sort); //p7b,h3b,p6b,p5b


    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {

       sd_t_d2_args[0] = k_range[h1b];
       sd_t_d2_args[1] = k_range[h2b];
       sd_t_d2_args[2] = k_range[h3b];
       sd_t_d2_args[3] = k_range[p4b];
       sd_t_d2_args[4] = k_range[p5b];
       sd_t_d2_args[5] = k_range[p6b];
       sd_t_d2_args[6] = k_range[p7b];
       
       sd_t_d2_exec[0] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin());
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin());
        
     }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_args[7]  = k_range[h1b];
       sd_t_d2_args[8]  = k_range[h2b];
       sd_t_d2_args[9]  = k_range[h3b];
       sd_t_d2_args[10] = k_range[p4b];
       sd_t_d2_args[11] = k_range[p5b];
       sd_t_d2_args[12] = k_range[p6b];
       sd_t_d2_args[13] = k_range[p7b];
       
       sd_t_d2_exec[1] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 1*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 1*max_dimb);
      }

    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b)) 
     {
       sd_t_d2_args[14] = k_range[h1b];
       sd_t_d2_args[15] = k_range[h2b];
       sd_t_d2_args[16] = k_range[h3b];
       sd_t_d2_args[17] = k_range[p4b];
       sd_t_d2_args[18] = k_range[p5b];
       sd_t_d2_args[19] = k_range[p6b];
       sd_t_d2_args[20] = k_range[p7b];
       
       sd_t_d2_exec[2] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 2*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 2*max_dimb);
     }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
       sd_t_d2_args[21] = k_range[h1b];
       sd_t_d2_args[22] = k_range[h2b];
       sd_t_d2_args[23] = k_range[h3b];
       sd_t_d2_args[24] = k_range[p4b];
       sd_t_d2_args[25] = k_range[p5b];
       sd_t_d2_args[26] = k_range[p6b];
       sd_t_d2_args[27] = k_range[p7b];
       
       sd_t_d2_exec[3] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 3*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 3*max_dimb);
      }

    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_args[28] = k_range[h1b];
       sd_t_d2_args[29] = k_range[h2b];
       sd_t_d2_args[30] = k_range[h3b];
       sd_t_d2_args[31] = k_range[p4b];
       sd_t_d2_args[32] = k_range[p5b];
       sd_t_d2_args[33] = k_range[p6b];
       sd_t_d2_args[34] = k_range[p7b];
       
       sd_t_d2_exec[4] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 4*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 4*max_dimb);
     }
  
    if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
     {
       sd_t_d2_args[35] = k_range[h1b];
       sd_t_d2_args[36] = k_range[h2b];
       sd_t_d2_args[37] = k_range[h3b];
       sd_t_d2_args[38] = k_range[p4b];
       sd_t_d2_args[39] = k_range[p5b];
       sd_t_d2_args[40] = k_range[p6b];
       sd_t_d2_args[41] = k_range[p7b];
       
       sd_t_d2_exec[5] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 5*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 5*max_dimb);
     }

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
       sd_t_d2_args[42] = k_range[h1b];
       sd_t_d2_args[43] = k_range[h2b];
       sd_t_d2_args[44] = k_range[h3b];
       sd_t_d2_args[45] = k_range[p4b];
       sd_t_d2_args[46] = k_range[p5b];
       sd_t_d2_args[47] = k_range[p6b];
       sd_t_d2_args[48] = k_range[p7b];
       
       sd_t_d2_exec[6] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 6*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 6*max_dimb);
     }

     if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_args[49] = k_range[h1b];
       sd_t_d2_args[50] = k_range[h2b];
       sd_t_d2_args[51] = k_range[h3b];
       sd_t_d2_args[52] = k_range[p4b];
       sd_t_d2_args[53] = k_range[p5b];
       sd_t_d2_args[54] = k_range[p6b];
       sd_t_d2_args[55] = k_range[p7b];
       
       sd_t_d2_exec[7] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 7*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 7*max_dimb);
     }

    if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
      {
       sd_t_d2_args[56] = k_range[h1b];
       sd_t_d2_args[57] = k_range[h2b];
       sd_t_d2_args[58] = k_range[h3b];
       sd_t_d2_args[59] = k_range[p4b];
       sd_t_d2_args[60] = k_range[p5b];
       sd_t_d2_args[61] = k_range[p6b];
       sd_t_d2_args[62] = k_range[p7b];
       
       sd_t_d2_exec[8] = true;

       std::copy(k_a_sort.begin(),k_a_sort.end(),k_abuf2.begin() + 8*max_dima);
       std::copy(k_b_sort.begin(),k_b_sort.end(),k_bbuf2.begin() + 8*max_dimb);
     }

    }
    }}}
    }}
    } //end ia6

  if(std::all_of(sd_t_d2_exec.begin(),sd_t_d2_exec.end(), [](bool x){return x;}))  kcalls_fused++;
  int npfused = std::count_if(sd_t_d2_exec.begin(), sd_t_d2_exec.end(), [](bool x){ return x; });
  if(npfused > 1 && npfused < 9) kcalls_pfused++;

    if(std::any_of(sd_t_d2_exec.begin(),sd_t_d2_exec.end(), [](bool x){return x;})) {
        kcalls++;
        sd_t_d2_all_cuda_tgen(&sd_t_d2_args[0], &a_c[0],
                      &k_abuf2[0], abuf_size2,
                      &k_bbuf2[0], bbuf_size2,
                      sd_t_d2_exec, 1);
    }

    } //end p7b
    #endif


} //end double_gpu_driver

#endif //CCSD_T_DOUBLES_GPU_TGEN_HPP_