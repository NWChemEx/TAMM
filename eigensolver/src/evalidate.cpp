#include "evalidate.hpp"

void evalidate(MPI_Comm comm, SpectralProbe *SPs, int nslices, int n, int nev) {
   int nwant, spnev, nvalid, accleft, tvalid, iprev;
   double midpt;

   VectorXd sliceevals, sliceresnrms, tempresnrms;
   VectorXi lengths = VectorXi::Zero(nslices);
   VectorXi originds, tempvec;
/*
   for (int ip = 0; ip < nslices; ip++) {
      inds[ip].resize(nev);
   }
*/

   // initialize each probe's valid index array
   for (int ip = 0; ip < nslices; ip++)  {
      spnev = SPs[ip].evals.size();
      (SPs[ip].valind).resize(spnev);
      (SPs[ip].valind).setZero();
      SPs[ip].nvalid = 0;
   }

   for (int ip = 0; ip < nslices; ip++) {
      if (SPs[ip].prev == -1) {
         // the leftmost probe
         logOFS << "validating the leftmost slice" << endl;
         nwant = SPs[ip].nev_below_shift; // inertia count for the interval (-inf,shifts(0))
         spnev = SPs[ip].evals.size(); // number of eigenvalues within the leftmost probe 
         nvalid = 0; tvalid = 0;

         sliceevals.resize(spnev);
         sliceevals.setZero();
         sliceresnrms.resize(spnev);
         sliceresnrms.setZero();

         originds.resize(spnev);

         // pick out the approximate eigenvalues to the left of the shift
         // from the leftmost probe; save the corresponding residuals
         logOFS << "  validating eigenvalues" << endl;
         for (int j = 0; j < SPs[ip].evals.size(); j++) {
            if (SPs[ip].evals(j) < SPs[ip].shift) {
               sliceevals(nvalid) = SPs[ip].evals(j);
               sliceresnrms(nvalid) = SPs[ip].resnrms(j);
               originds(nvalid) = j;
               nvalid++;
            }
         }

         logOFS << "### slice: " << 0 << ", right shift: " << SPs[ip].shift << ", num in slice: " << nvalid << ", inertial count: " << nwant << std::endl;

         // sort the residuals associated with the selected eigenvalues
         VectorXi resinds = sortinds(sliceresnrms.head(nvalid));

         logOFS << scientific;

         // copy the original indices of the selected eigenvalues in 
         // the leftmost probe 
         for (int j = 0; j < min(nwant,nvalid); j++) {
//chao            inds[ip](lengths(ip)) = originds(resinds[j]);
//
            SPs[ip].valind(SPs[ip].nvalid) = originds(resinds[j]); 
            logOFS << "n: " << tvalid << ", slice: " << 0 << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
            tvalid++;
            lengths(ip)++;
            SPs[ip].nvalid++;
         }
      }
      else {
         // all other probes
         // expected number of eigenvalues between shifts(iprev) and shifts(ip) (inertial count
         iprev = SPs[ip].prev;
         nwant = SPs[ip].nev_below_shift - SPs[iprev].nev_below_shift;
         // the total number of eigenvalues from targest probe i-1 and i
         spnev = SPs[iprev].evals.size() + SPs[ip].evals.size();
         nvalid = 0; accleft = 0;
         midpt = (SPs[ip].shift + SPs[iprev].shift)/2.0;

         sliceevals.resize(spnev);
         sliceresnrms.resize(spnev);

         originds.resize(spnev);

         // select eigenvalues within the interval [shift(i-1),shifts(i)]
         // from probe SPs[ip].prev
         for (int j = 0; j < SPs[iprev].evals.size(); j++) {
            if (SPs[iprev].evals(j) >= SPs[iprev].shift && SPs[iprev].evals(j) < midpt) {
               sliceevals(nvalid) = SPs[iprev].evals(j);
               sliceresnrms(nvalid) = SPs[iprev].resnrms(j);
               originds(nvalid) = j;
               nvalid++;
               accleft++;
            }
         }
         // select eigenvalues within the interval [shift(i-1),shifts(i)]
         // from the target slice i
         for (int j = 0; j < SPs[ip].evals.size(); j++) {
            if (SPs[ip].evals(j) >= midpt && SPs[ip].evals(j) < SPs[ip].shift) {
               sliceevals(nvalid) = SPs[ip].evals(j);
               sliceresnrms(nvalid) = SPs[ip].resnrms(j);
               originds(nvalid) = j;
               nvalid++;
            }
         }

         logOFS << "### slice: " << ip << ", right shift: " << SPs[ip].shift << ", num in slice: " << nvalid << ", inertial count: " << nwant << std::endl;

         // sort the residual norms associated with the selected 
         // eigenvalues and return the sorted indices
         VectorXi resinds = sortinds(sliceresnrms.head(nvalid));

         // copy the eigenvectors associated with the converged eigenvalues 
         // back to target SPs to prepare the starting guess for 
         // the next SCF iteration
         if (tvalid + min(nvalid,nwant) <= nev) { 
            logOFS << scientific; 
            for (int j = 0; j < min(nvalid,nwant); j++) {
               if (resinds[j] < accleft) {
//chao                  inds[iprev](lengths(iprev)) = originds(resinds[j]);
                  (SPs[iprev]).valind(SPs[iprev].nvalid) = originds(resinds[j]);
                  SPs[iprev].nvalid++;
                  lengths(iprev)++;
                  logOFS << "n: " << tvalid << ", slice: " << iprev << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals(resinds[j]) << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               else {
//chao                  inds[ip](lengths(ip)) = originds(resinds[j]);  
                  SPs[ip].valind(SPs[ip].nvalid) = originds(resinds[j]);  
                  SPs[ip].nvalid++;
                  lengths(ip)++;
                  logOFS << "n: " << tvalid << ", slice: " << ip << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               tvalid++;
            }
         }
         else {
            /*for (int j = 0; j < nvalid; j++) {
            cout << "--- eval: " << sliceevals[j] << ", resnrm: " << sliceresnrms[j] << endl;
            }*/
            // look at the remaining eigenvalues and pick ones with
            // small residual norms?
            VectorXi evalinds = sortinds(sliceevals.head(nvalid));
            /*cout << "resinds: " << endl;
            cout << resinds << endl;
            cout << "evalinds: " << endl;
            cout << evalinds << endl;*/
            int j = 0; 
            logOFS << scientific; 
            bool accept;
            while (tvalid < nev && j < nvalid) {
               accept = false;
               for (int l = 0; l < min(nvalid,nwant); l++){
                  if (evalinds[j] == resinds[l]) {
                     accept = true;
                  }
               }
               if (accept) {
                  if (evalinds[j] < accleft) {
//Chao                     inds[iprev](lengths(iprev)) = originds(evalinds[j]);
                     SPs[iprev].valind(SPs[iprev].nvalid) = originds(evalinds[j]);
                     SPs[iprev].nvalid++;
                     lengths(iprev)++;
                     logOFS << "n: " << tvalid << ", slice: " << iprev << ", ind: " << originds(evalinds[j]) << ", eval: " << sliceevals[evalinds[j]] << ", resnrm: " << sliceresnrms[evalinds[j]] << endl;
                  }
                  else {
//                     inds[ip](lengths(ip)) = originds(evalinds[j]);  
                     SPs[ip].valind(SPs[ip].nvalid) = originds(evalinds[j]);  
                     SPs[ip].nvalid++;
                     lengths(ip)++;
                     logOFS << "n: " << tvalid << ", slice: " << ip << ", ind: " << originds(evalinds[j]) << ", eval: " << sliceevals[evalinds[j]] << ", resnrm: " << sliceresnrms[evalinds[j]] << endl;
                  }
                  tvalid++;
               }
               j++;
            }
            break;
         }
      } // end if (ip == 0)
   }  // end for ip
/*
   for (int i = 0; i < nslices; i++) {
      tempvec.resize(lengths(i));
      tempvec = inds[i].head(lengths(i));
      inds[i] = tempvec;
   }
*/
}

