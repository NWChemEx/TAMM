noga {

//    i, j, m, n
index h1,h2,h3,h4,h5,h6 = O;
//    a, b, c, d, e, f
index p1,p2,p3,p4,p5,p6 = V;

//array i0[V][O];
//array f[N][N]: irrep_f;
//array v[N,N][N,N]: irrep_v;
//array t_vo[V][O]: irrep_t;

array R[O][V];

array D[O][O];
array Z[O][O];
array T[O][V];

array F_OV[O][V];
array F_VO[V][O];
array F_OO[O][O];
array F_VV[V][V];

array X_VV[V][V];
array X_OO[O][O];
array X_OV[O][V];


hf_1: R[h1,p1] = 1.0 * F_OV[h1,p1] 
				- F_OV[h1,p2] * X_VV[p2,p1] 
			    - F_OO[h1,h3] * X_OV[h3,p1]
				+ X_OO[h1,h2] * F_OV[h2,p1]
				- X_OO[h1,h2] * F_OV[h2,p2] * X_VV[p2,p1] 
				- X_OO[h1,h2] * F_OO[h2,h3] * X_OV[h3,p1]
				+ X_OV[h1,p2] * F_VV[p2,p1] 
				- X_OV[h1,p2] * F_VV[p2,p3] * X_VV[p3,p1] 
				- X_OV[h1,p2] * F_VO[p2,h3] * X_OV[h3,p1] ;
	
hf_2: Z[h1,h2] = -1.0 * T[h1,p5] * T[h2,p5];

hf_3: D[h1,h2] += delta[h1,h2] + D[h1,h3] * Z[h3,h2];

hf_4: X_OV[h1,p1] += D[h1,h3] * Z[h3,p1];

hf_5: X_OO[h1,h2] += -1.0 * T[h1,p5] * X_OV[h2,p5];

hf_6: X_VV[p1,p2] += X_OV[h3,p1] * T[h3,p2];

}

// -------------------------------------------

R_ia = F_ia - F_ib * X_ba - F_im * X_ma
       + X_ij*F_ja - X_ij * F_jb * X_ba - X_ij * F_jm * X_ma
       + X_ib * F_ba - X_ib * F_bc * X_ca - X_ib * F_bm * X_ma

Z_ij = -1.0 * T_ie * T_je

D_ij = delta_ij + D_im * Z_mj  (Solve iteratively for D_ij)

X_ia = D_im * T_ma

X_ij = -1.0 * T_ie * T_ij
X_ab = X_ma * T_mb

//---------------------------------------------------------------------------

while(not converged) #|D(i+1) - D(i)| < thresh

	for i in 1..M
		t1(n+1) = t1(n)  + R_ia / (f_aa(n) - f_ii(n))  #preconditioner

	[t_ia] -> known after the M iterations

	Z_ij = -1.0 * T_ie * T_je;

	DO N
		D(n+1) = D(n) + fD (D(n)) / delta
	CONTINUE #until Dij tensor converges

	X_ia = D_im * T_ma
	
	X_ab = X_ma * T_mb 
     && X_ij = -1.0 * T_ie * X_je

	#Construct new fock matrix F(D(i+1)) from Density matrix D
	#X_ia,X_ab,X_ij (i+1)


