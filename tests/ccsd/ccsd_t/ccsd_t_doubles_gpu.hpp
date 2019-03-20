
#ifndef CCSD_T_DOUBLES_GPU_HPP_
#define CCSD_T_DOUBLES_GPU_HPP_

using namespace tamm;


void sd_t_d1_1_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_2_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_3_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_4_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_5_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_6_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_7_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_8_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d1_9_cuda(int,int,int,int,int,int,int,double*,double*,double*);

void sd_t_d2_1_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_2_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_3_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_4_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_5_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_6_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_7_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_8_cuda(int,int,int,int,int,int,int,double*,double*,double*);
void sd_t_d2_9_cuda(int,int,int,int,int,int,int,double*,double*,double*);


template<typename T>
void ccsd_t_doubles_gpu(ExecutionContext& ec,
                   const TiledIndexSpace& MO,
                   const size_t noab, const size_t nvab,
                   Matrix& k_spin,
                   std::vector<T>& a_c,
                   Tensor<T>& d_t2, //d_a
                   Tensor<T>& d_v2, //d_b
                   std::vector<T>& p_evl_sorted,
                   std::vector<int>& k_range,int t_h1b, int t_h2b, int t_h3b,
                   int t_p4b, int t_p5b, int t_p6b,
                   int usedevice=1) {

    // initmemmodule();

    Eigen::Matrix<int, 9,6, Eigen::RowMajor> a3;
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

    auto notset=1;

    for (auto ia6=0; ia6<8; ia6++){
      if(a3(ia6,1) != 0) {
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

    int p4b,p5b,p6b,h1b,h2b,h3b;
    for (auto ia6=0; ia6<8; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    

      if ((usedevice==1)&&(notset==1)) {
       dev_mem_d(k_range[t_h1b],k_range[t_h2b],
                    k_range[t_h3b],k_range[t_p4b],
                    k_range[t_p5b],k_range[t_p6b]);
       notset=0;
      }


    if( (p4b<=p5b) && (h2b<=h3b) && p4b!=0){ 
      if(k_spin(p4b)+k_spin(p5b)+k_spin(p6b)
         +k_spin(h1b)+k_spin(h2b)+k_spin(h3b)!=12){
         if(k_spin(p4b)+k_spin(p5b)+k_spin(p6b)
         == k_spin(h1b)+k_spin(h2b)+k_spin(h3b)) {

           auto dimc=k_range[p4b]*k_range[p5b]*k_range[p6b]*
                     k_range[h1b]*k_range[h2b]*k_range[h3b];

          for (size_t h7b=0;h7b<noab;h7b++){

            if(k_spin(p4b)+k_spin(p5b)
              == k_spin(h1b)+k_spin(h7b)) {

                auto dim_common = k_range[h7b];
                auto dima_sort = k_range[p4b]*k_range[p5b]*k_range[h1b];
                auto dima = dim_common*dima_sort;
                auto dimb_sort = k_range[p6b]*k_range[h2b]*k_range[h3b];
                auto dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);


                  //TODO
    //   IF ((h7b .lt. h1b)) THEN
    //   CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,int_mb(k_a_offset),(h1b_1
    //  & - 1 + noab * (h7b_1 - 1 + noab * (p5b_1 - noab - 1 + nvab * (p4b_
    //  &1 - noab - 1)))))
    //   CALL TCE_SORT_4(dbl_mb(k_a),dbl_mb(k_a_sort),int_mb(k_range+p4b-1)
    //  &,int_mb(k_range+p5b-1),int_mb(k_range+h7b-1),int_mb(k_range+h1b-1)
    //  &,4,2,1,3,-1.0d0)
    //   END IF
    //       IF ((h1b .le. h7b)) THEN
    //   CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,int_mb(k_a_offset),(h7b_1
    //  & - 1 + noab * (h1b_1 - 1 + noab * (p5b_1 - noab - 1 + nvab * (p4b_
    //  &1 - noab - 1)))))
    //   CALL TCE_SORT_4(dbl_mb(k_a),dbl_mb(k_a_sort),int_mb(k_range+p4b-1)
    //  &,int_mb(k_range+p5b-1),int_mb(k_range+h1b-1),int_mb(k_range+h7b-1)
    //  &,3,2,1,4,1.0d0)
    //   END IF

    std::vector<T> k_b_sort(dimb);
    //       IF ((h7b .le. p6b)) THEN
    //   if(.not.intorb) then
    //   CALL GET_HASH_BLOCK(d_b,dbl_mb(k_b_sort),
    //  &dimb,int_mb(k_b_offset),(h3b_2
    //  & - 1 + (noab+nvab) * (h2b_2 - 1 + (noab+nvab) * (p6b_2 - 1 + (noab
    //  &+nvab) * (h7b_2 - 1)))))

 if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
     {
        sd_t_d1_1_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

 if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
           sd_t_d1_2_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

  if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
     {
       sd_t_d1_3_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

  if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        sd_t_d1_4_cuda(k_range[h1b],k_range[h2b],
                k_range[h3b],k_range[h7b],k_range[p4b],
                k_range[p5b],k_range[p6b],
                &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

 if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
       sd_t_d1_5_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }
  
 if ((t_p4b == p6b) && (t_p5b == p4b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b))
     {
          sd_t_d1_6_cuda(k_range[h1b],k_range[h2b],
                         k_range[h3b],k_range[h7b],k_range[p4b],
                         k_range[p5b],k_range[p6b],
                       &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

  if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b)) 
      {
        sd_t_d1_7_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

 if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h1b) && (t_h3b == h3b))
      {
        sd_t_d1_8_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

if ((t_p4b == p4b) && (t_p5b == p6b) && (t_p6b == p5b)
     && (t_h1b == h2b) && (t_h2b == h3b) && (t_h3b == h1b)) 
      {
         sd_t_d1_9_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[h7b],k_range[p4b],
                    k_range[p5b],k_range[p6b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

     }}}
    }}}

    } //end ia6

    //--------------------------------BEGIN DOUBLES-2----------------------------//


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
      if(a3(ia6,1) != 0) {
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

    // int p4b,p5b,p6b,h1b,h2b,h3b;
    for (auto ia6=0; ia6<8; ia6++){ 
      p4b=a3(ia6,0);
      p5b=a3(ia6,1);
      p6b=a3(ia6,2);
      h1b=a3(ia6,3);
      h2b=a3(ia6,4);
      h3b=a3(ia6,5);
    
      if( (p5b<=p6b) && (h1b<=h2b) && p4b!=0){ 
      if(k_spin(p4b)+k_spin(p5b)+k_spin(p6b)
         +k_spin(h1b)+k_spin(h2b)+k_spin(h3b)!=12){
         if(k_spin(p4b)+k_spin(p5b)+k_spin(p6b)
         == k_spin(h1b)+k_spin(h2b)+k_spin(h3b)) {

           auto dimc=k_range[p4b]*k_range[p5b]*k_range[p6b]*
                     k_range[h1b]*k_range[h2b]*k_range[h3b];

          for (size_t p7b=noab+1;p7b<noab+nvab;p7b++){

            if(k_spin(p4b)+k_spin(p7b)
              == k_spin(h1b)+k_spin(h2b)) {

                auto dim_common = k_range[p7b];
                auto dima_sort = k_range[p4b]*k_range[h1b]*k_range[h2b];
                auto dima = dim_common*dima_sort;
                auto dimb_sort = k_range[p5b]*k_range[p6b]*k_range[h3b];
                auto dimb = dim_common*dimb_sort;
                
                if(dima > 0 && dimb > 0) {
                  std::vector<T> k_a(dima);
                  std::vector<T> k_a_sort(dima);

                  //TODO
    //                     IF ((p7b .lt. p4b)) THEN
    //   CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,int_mb(k_a_offset),(h2b_1
    //  & - 1 + noab * (h1b_1 - 1 + noab * (p4b_1 - noab - 1 + nvab * (p7b_
    //  &1 - noab - 1)))))
    //   CALL TCE_SORT_4(dbl_mb(k_a),dbl_mb(k_a_sort),int_mb(k_range+p7b-1)
    //  &,int_mb(k_range+p4b-1),int_mb(k_range+h1b-1),int_mb(k_range+h2b-1)
    //  &,4,3,2,1,-1.0d0)
    //   END IF
    //   IF ((p4b .le. p7b)) THEN
    //   CALL GET_HASH_BLOCK(d_a,dbl_mb(k_a),dima,int_mb(k_a_offset),(h2b_1
    //  & - 1 + noab * (h1b_1 - 1 + noab * (p7b_1 - noab - 1 + nvab * (p4b_
    //  &1 - noab - 1)))))
    //   CALL TCE_SORT_4(dbl_mb(k_a),dbl_mb(k_a_sort),int_mb(k_range+p4b-1)
    //  &,int_mb(k_range+p7b-1),int_mb(k_range+h1b-1),int_mb(k_range+h2b-1)
    //  &,4,3,1,2,1.0d0)
    //   END IF

    std::vector<T> k_b_sort(dimb);
    //   IF ((h3b .le. p7b)) THEN
    //   if(.not.intorb) then
    //   CALL GET_HASH_BLOCK(d_b,dbl_mb(k_b_sort),dimb,
    //  &int_mb(k_b_offset),(p7b_2
    //  & - 1 + (noab+nvab) * (h3b_2 - 1 + (noab+nvab) * (p6b_2 - 1 + (noab
    //  &+nvab) * (p5b_2 - 1)))))


    if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
     {
        sd_t_d2_1_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
        
     }

if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
           sd_t_d2_2_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

if ((t_p4b == p4b) && (t_p5b == p5b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b)) 
     {
       sd_t_d2_3_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

   if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        sd_t_d2_4_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                &a_c[0],&k_a_sort[0],&k_b_sort[0]);
      }

 if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
       sd_t_d2_5_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }
  
 if ((t_p4b == p5b) && (t_p5b == p4b) && (t_p6b == p6b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
     {
          sd_t_d2_6_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                       &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

 if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h2b) && (t_h3b == h3b))
      {
        sd_t_d2_7_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

 if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h3b) && (t_h2b == h1b) && (t_h3b == h2b))
      {
        sd_t_d2_8_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

if ((t_p4b == p5b) && (t_p5b == p6b) && (t_p6b == p4b)
     && (t_h1b == h1b) && (t_h2b == h3b) && (t_h3b == h2b))
      {
         sd_t_d2_9_cuda(k_range[h1b],k_range[h2b],
                    k_range[h3b],k_range[p4b],
                    k_range[p5b],k_range[p6b],k_range[p7b],
                    &a_c[0],&k_a_sort[0],&k_b_sort[0]);
     }

        }}}
    }}}

    } //end ia6


} //end double_gpu_driver

#endif //CCSD_T_DOUBLES_GPU_HPP_