#include "getshifts.hpp"

VectorXd getshifts(MPI_Comm comm, MatrixXd &H, MatrixXd &S, int nev, int nshifts, int *maxcnt) 
{
   int rank;
   MPI_Comm_rank(comm, &rank);

   // set nlan
   int n = H.rows();

   // run nlan steps of Lanczos 
   int nlan = 100; // the most we take
   if (n < 5000) nlan = 50;
   if (n < 4000) nlan = 40;
   if (n < 3000) nlan = 30;
   if (n < 2000) nlan = 20;
   if (n < 1000) nlan = 10;

   VectorXd f(n), v0(n);
   MatrixXd V(n,nlan), T(nlan,nlan);
   T.setZero();
   V.setZero();
   GaussianRandom(0.0, 1.0, v0);
   f = HSLanczos(H, S, v0, T, V);

   // diagonalize the tridiagonal T returned from Lanczos
   VectorXd d(nlan);
   char vec = 'V', lower = 'L';
   int  lwork = nlan*nlan, ierr = 0;
   double *work;
   //work = new double [lwork];
   //dsyev_(&vec,&lower,&nlan,T.data(),&nlan,d.data(),work,&lwork,&ierr);
   ierr = LAPACKE_dsyev(LAPACK_COL_MAJOR, vec, lower, nlan, T.data(), nlan,
                 d.data());
   if (ierr) {
      std::cout << "dsyev failed: ierr = " << ierr << std::endl;
      exit(1);
   }
   //delete[] work;

   VectorXd evbnds(2);
   double beta = f.norm();
   evbnds = getevbnd(d, T, beta);
   if (rank == 0) {
      cout << " lower ev bound = " << evbnds(0) << endl;
      cout << " upper ev bound = " << evbnds(1) << endl;
   }

   // compute gaps and use the median gap to define sigma
   // which defines width of the of the Gaussian broadening
   // used to construct the DOS and cumulative DOS
   VectorXd sigma(nlan);

   std::vector< double > gaps( nlan );
   std::copy_n( d.data(), nlan, gaps.begin() );

   // Get adjacent differences in place
   std::adjacent_difference( gaps.begin(),
                             gaps.end(),
                             gaps.begin() );
 
   // take the gap at d(i) to be the larger of the distances 
   // between d(i) and d(i-1) and between d(i) and d(i+1)
   // use lower and upper bounds of the spectrum for d(0) and d(nlan)
   double gapmax = 0.0, stol = 1.0e-4, logstol = std::log(stol);
   for (int iLanz = 0; iLanz < nlan; iLanz++) {
     if (iLanz==0) {
        //gapmax = ( d(0) - evbnds(0) +  gaps[0] ) / 2.0;
        gapmax = std::max( d(0) - evbnds(0), gaps[0] );
     }
     else if (iLanz==nlan-1) {
        // gapmax = ( evbnds(1) - d(nlan-1) + gaps[nlan-1] ) / 2.0;
        gapmax = std::max( evbnds(1) - d(nlan-1), gaps[nlan-1] );
     }
     else {
        //gapmax = (gaps[iLanz-1]+gaps[iLanz])/2.0;
        gapmax = std::max( gaps[iLanz-1], gaps[iLanz] );
     }
     // make exp(-gapmax^2/sigma^2) smaller than tol
     sigma[iLanz] = gapmax/std::sqrt(-logstol);
   }

   // define DOS parameter 
   DensityOfStates DOS;
   int nSamp = 1; //for now
   for( int64_t iLanz = 0; iLanz < nlan; ++iLanz ) {
      double exp_fact  = 2. * sigma(iLanz) * sigma(iLanz);
      double gamma2 = T(0,iLanz)*T(0,iLanz);
      double prefactor = (n*gamma2 / nSamp) /
                          std::sqrt( M_PI * exp_fact );
      DOS.prefactor.emplace_back( prefactor );
      DOS.center.emplace_back( d(iLanz) );
      DOS.exp.emplace_back( 1. / exp_fact );
   }

   // The DOS profile defined on a spectrum mesh 
   // is NOT used in the current version
   // generate a spectrum mesh between evbnds(0) and evbnds(1)
   int nPts = 5000;
   DOS.domain = linspace( evbnds(0), evbnds(1), nPts+2 );
   // Remove end points
   for( int64_t j = 1; j < DOS.domain.size(); ++j )
      DOS.domain[j-1] = DOS.domain[j];
   DOS.domain.pop_back();
   DOS.domain.pop_back();

   std::tuple< std::vector< double >, std::vector< double > >
      shift_data;

   auto& shifts = std::get<0>( shift_data );
   auto& evcnts = std::get<1>( shift_data );

   auto& domain = DOS.domain;

   std::vector <double> yy(nPts+2);
   yy=DeferredDOSIntegralEval( DOS, domain);

/*
   cout << "xx = " << endl;
   for (int i = 0; i < domain.size(); i++)
      cout << domain[i] << endl;

   cout << "yy = " << endl;
   for (int i = 0; i < yy.size(); i++)
   cout << yy[i] << endl;
*/

   // set the target average number of eigenvalue per interval
   int nevloc = std::ceil(nev*1.1/nshifts);
   logOFS << " input nshifts = " << nshifts << endl;
   logOFS << " define nevloc = " << nevloc << endl;
/*
   int maxref = 1, numref = 0, refine=0;

   while (numref < maxref) {
      int i0 = 0;
      for (auto k = 0; k < nshifts; k++) {
         for (auto ix = i0; ix < domain.size(); ix++) {
            if (yy[ix] > (k+1)*nevloc) {
               i0 = ix;
               cout << "k = " << k << " i0 = " << i0 << endl;
               cout << " xx = " << domain[i0] << " yy = " << yy[ix] << endl;
               shifts.emplace_back(domain[i0]);
               break;
            }
         }
         if (k > 0 && shifts[k]==shifts[k-1]) {
            refine = 1;
            cout << " refine! numref = " << numref << endl;
            nPts = nPts*10;
            DOS.domain = linspace( evbnds(0), evbnds(1), nPts+2 );
            // Remove end points
            for( int64_t j = 1; j < DOS.domain.size(); ++j )
               DOS.domain[j-1] = DOS.domain[j];
            DOS.domain.pop_back();
            DOS.domain.pop_back();

            break;
         };
      }
      if (refine == 0) break;
      numref++;
   }
*/
   for( int64_t k = (nshifts); k >= 1; k-- ) {

      auto obj_f = [&]( double x ) {

        auto z_eval = DeferredDOSIntegralEval( DOS, { x } );
        return z_eval[0] - (k*nevloc);

      };

      double upper_bnd =
        shifts.size() ?  shifts.back() : evbnds(1);

      shifts.emplace_back(
        Bisection( obj_f, evbnds(0), upper_bnd )
      );

   }

   // shifts.emplace_back( evbnds(0) ); // Begining is a good bet

   std::sort( shifts.begin(), shifts.end() );

   // go through the shifts and eleminate duplicates
   int iover = 0;
   double mingap = 1.0e-3;
   for (int i = 1; i < nshifts; i++)
      if (fabs(shifts[i] - shifts[i-1]) < mingap ) {
         iover++;
         logOFS << "iover = " << iover << endl;
         auto obj_f = [&]( double x ) {

            auto z_eval = DeferredDOSIntegralEval( DOS, { x } );
            return z_eval[0] - ((nshifts+iover)*nevloc);

         };

         shifts.emplace_back(
            Bisection( obj_f, shifts.back(), evbnds(1) )
         );
      }

   int nshifts0 = shifts.size();
   if (rank == 0) cout << "getshifts: nshifts = " << nshifts0 << endl;

   // cleanup and copy to export
   VectorXd Eigen_shifts(nshifts); 
   Eigen_shifts(0) = shifts[0]; 
   int k = 1;
   for (int i = 1; i < nshifts0; i++) {
      if (fabs(shifts[i] - shifts[i-1]) >= mingap ) {
         Eigen_shifts(k) = shifts[i];
         k++;
      }
   }

   // estimate the number of eigenvalue between shifts
   for (int i = 0; i < nshifts; i++) {
      auto fshift = DeferredDOSIntegralEval( DOS, {Eigen_shifts(i)} );
      evcnts.emplace_back(fshift.back());
      if (rank == 0) {
         cout << " shift = " << Eigen_shifts(i) << " cdos = " << evcnts[i] << endl;
      }
   }
   for (int i = nshifts-1; i > 0; i--) {
      evcnts[i] = evcnts[i] - evcnts[i-1];
   }

   *maxcnt = 0;
   for (int i = 0; i < nshifts; i++)
      if (evcnts[i] > *maxcnt) *maxcnt = std::ceil(evcnts[i]);

   return Eigen_shifts;
}

