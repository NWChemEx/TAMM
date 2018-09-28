#include "evalidate.hpp"

void evalidate(MPI_Comm comm, SpectralProbe *SPs, int nslices, int n, int nev) {
   int nwant, spnev, nvalid, accleft, tselect, iprev;
   double midpt;

   VectorXd sliceevals, sliceresnrms, tempresnrms;
   VectorXi originds;

   // initialize each probe's valid index array
   for (int ip = 0; ip < nslices; ip++)  {
      spnev = SPs[ip].evals.size();
      (SPs[ip].selind).resize(spnev);
      (SPs[ip].selind).setZero();
      SPs[ip].nselect = 0;
   }

   for (int ip = 0; ip < nslices; ip++) {
      if (SPs[ip].prev == -1) {
         // the leftmost probe
         logOFS << "validating the leftmost slice" << endl;
         nwant = SPs[ip].nev_below_shift; // inertia count for the interval (-inf,shifts(0))
         spnev = SPs[ip].evals.size(); // number of eigenvalues within the leftmost probe 
         nvalid = 0; tselect = 0;

         sliceevals.resize(spnev);
         sliceevals.setZero();
         sliceresnrms.resize(spnev);
         sliceresnrms.setZero();

         originds.resize(spnev);

         // pick out the approximate eigenvalues to the left of the shift
         // from the leftmost probe; save the corresponding residuals
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
            SPs[ip].selind(SPs[ip].nselect) = originds(resinds[j]); 
            logOFS << "n: " << tselect << ", probe: " << ip << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
            tselect++;
            SPs[ip].nselect++;
         }
      }
      else {
         // all other probes
         // expected number of eigenvalues between shifts(iprev) and shifts(ip) (inertial count
         logOFS << "validating other slices" << endl;
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
         // from probe ip
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
         // eigenvalues and return the indices in the order of sorted
         // residuals
         VectorXi resinds = sortinds(sliceresnrms.head(nvalid));

         // copy the eigenvectors associated with the converged eigenvalues 
         // back to target SPs to prepare the starting guess for 
         // the next SCF iteration
         if (tselect + min(nvalid,nwant) <= nev) { 
            logOFS << scientific; 
            for (int j = 0; j < min(nvalid,nwant); j++) {
               if (resinds[j] < accleft) {
                  (SPs[iprev]).selind(SPs[iprev].nselect) = originds(resinds[j]);
                  SPs[iprev].nselect++;
                  logOFS << "n: " << tselect << ", probe: " << iprev << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals(resinds[j]) << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               else {
                  SPs[ip].selind(SPs[ip].nselect) = originds(resinds[j]);  
                  SPs[ip].nselect++;
                  logOFS << "n: " << tselect << ", probe: " << ip << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               tselect++;
            }
         }
         else {
            // the last slice?
            /*for (int j = 0; j < nvalid; j++) {
            cout << "--- eval: " << sliceevals[j] << ", resnrm: " << sliceresnrms[j] << endl;
            }*/
            // look at the remaining eigenvalues and pick ones with
            // small residual norms?
            VectorXi eselinds = sortinds(sliceevals.head(nvalid));
            /*cout << "resinds: " << endl;
            cout << resinds << endl;
            cout << "eselinds: " << endl;
            cout << eselinds << endl;*/
            int j = 0; 
            logOFS << scientific; 
            bool accept;
            while (tselect < nev && j < nvalid) {
               accept = false;
               for (int l = 0; l < min(nvalid,nwant); l++){
                  if (eselinds[j] == resinds[l]) {
                     accept = true;
                  }
               }
               if (accept) {
                  if (eselinds[j] < accleft) {
                     SPs[iprev].selind(SPs[iprev].nselect) = originds(eselinds[j]);
                     SPs[iprev].nselect++;
                     logOFS << "n: " << tselect << ", probe: " << iprev << ", ind: " << originds(eselinds[j]) << ", eval: " << sliceevals[eselinds[j]] << ", resnrm: " << sliceresnrms[eselinds[j]] << endl;
                  }
                  else {
                     SPs[ip].selind(SPs[ip].nselect) = originds(eselinds[j]);  
                     SPs[ip].nselect++;
                     logOFS << "n: " << tselect << ", probe: " << ip << ", ind: " << originds(eselinds[j]) << ", eval: " << sliceevals[eselinds[j]] << ", resnrm: " << sliceresnrms[eselinds[j]] << endl;
                  }
                  tselect++;
               }
               j++;
            }
            break;
         }
      } // end if (ip == 0)
   }  // end for ip
}

