

{
 
 index h1, h2, h3,h4,h5,h6,h7,h8 = O;
 index p1, p2, p3,p4,p5,p6,p7,p8 = V;

 array i0[V][O] : irrep_t;
 array i1[O][O] : irrep_t;
 array i2[O][V] : irrep_t;
 array f[V][V] : irrep_f;
 array v[V,V][V,V] : irrep_v; 
 array t1[V][V]: irrep_t;
 array t2[V,V][O,O] : irrep_t;

 array t[V][V]: irrep_t;

 array _i1_2[O][O], _i1_2_2[O][V], _i1_3[V][V], _i1_5[O][V], _i1_6[O,O][O,V];

t1_1:     i0[p2,h1] += f[p2,h1];
t1_2_1:   i1 [ h7, h1 ] += 1 * f [ h7, h1 ];
t1_2_2_1: i2 [ h7, p3 ] += 1 * f [ h7, p3 ] ;
t1_2_2_2: i2 [ h7, p3 ] += -1 * t [ p5, h6 ] * v [ h6, h7, p3, p5 ] ;
t1_2_2:   i1 [ h7, h1 ] += 1 * t [ p3, h1 ] * i2 [ h7, p3 ] ;
t1_2_3:   i1 [ h7, h1 ] += -1 * t [ p4, h5 ] * v [ h5, h7, h1, p4 ] ;
t1_2_4:   i1 [ h7, h1 ] += -1/2  * t [ p3, p4, h1, h5 ] * v [ h5, h7, p3, p4 ] ;
t1_2:     i0 [ p2, h1 ] += -1 * t [ p2, h7 ] * i1 [ h7, h1 ] ;
t1_3_1:   i1 [ p2, p3 ] += 1 * f [ p2, p3 ] ;
t1_3_2:   i1 [ p2, p3 ] += -1 * t [ p4, h5 ] * v [ h5, p2, p3, p4 ] ;
t1_3:     i0 [ p2, h1 ] += 1 * t [ p3, h1 ] * i1 [ p2, p3 ] ;
t1_4:     i0 [ p2, h1 ] += -1 * t [ p3, h4 ] * v [ h4, p2, h1, p3 ] ;
t1_5_1:   i1 [ h8, p7 ] += 1 * f [ h8, p7 ] ;
t1_5_2:   i1 [ h8, p7 ] += 1 * t [ p5, h6 ] * v [ h6, h8, p5, p7 ] ;
t1_5:     i0 [ p2, h1 ] += 1 * t [ p2, p7, h1, h8 ] * i1 [ h8, p7 ] ;
t1_6_1:   i1 [ h4, h5, h1, p3 ] += 1 * v [ h4, h5, h1, p3 ] ;
t1_6_2:   i1 [ h4, h5, h1, p3 ] += -1 * t [ p6, h1 ] * v [ h4, h5, p3, p6 ] ;
t1_6:     i0 [ p2, h1 ] += -1/2 * t [ p2, p3, h4, h5 ] * i1 [ h4, h5, h1, p3 ] ;
t1_7:     i0 [ p2, h1 ] += -1/2 * t [ p3, p4, h1, h5 ] * v [ h5, p2, p3, p4 ] ;
}
