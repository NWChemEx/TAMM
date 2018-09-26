#include "enforcecounts.hpp"
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using std::cout;
using std::endl;
using std::min;
using std::scientific;
using std::showpos;
using std::noshowpos;

void enforcecounts_mpi(MPI_Comm comm, slice *slices, int nslices, int n, int nev, VectorXi *inds) {
   int inrtcount, accboth, accslice, accleft, acctotal;
   double midpt;

   VectorXd sliceevals, sliceresnrms, tempresnrms;
   VectorXi lengths = VectorXi::Zero(nslices);
   VectorXi originds, tempvec;

   int rank;
   MPI_Comm_rank(comm, &rank);

   for (int i = 0; i < nslices; i++) {
      inds[i].resize(nev);
   }

   // first slice
   inrtcount = slices[0].nleft;
   accboth = slices[0].evals.size();
   accslice = 0; acctotal = 0;

   sliceevals.resize(accboth);
   sliceresnrms.resize(accboth);

   originds.resize(accboth);

   for (int j = 0; j < slices[0].evals.size(); j++) {
      if (slices[0].evals(j) < slices[0].shift) {
         sliceevals(accslice) = slices[0].evals(j);
         sliceresnrms(accslice) = slices[0].resnrms(j);
         originds(accslice) = j;
         accslice++;
      }
   }

   std::cout << "### slice: " << 0 << ", right shift: " << slices[0].shift << ", num in slice: " << accslice << ", inertial count: " << inrtcount << std::endl;

   VectorXi resinds = sortinds(sliceresnrms.head(accslice));

   cout << scientific;

   for (int j = 0; j < min(inrtcount,accslice); j++) {
      inds[0](lengths(0)) = originds(resinds[j]);
      cout << "n: " << acctotal << ", slice: " << 0 << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
      acctotal++;
      lengths(0)++;
   }

   // all other slices
   for (int i = 1; i < nslices; i++) {
      inrtcount = slices[i].nleft - slices[i-1].nleft;
      accboth = slices[i-1].evals.size() + slices[i].evals.size();
      accslice = 0; accleft = 0;
      midpt = (slices[i].shift + slices[i-1].shift)/2.0;

      sliceevals.resize(accboth);
      sliceresnrms.resize(accboth);

      originds.resize(accboth);

      for (int j = 0; j < slices[i-1].evals.size(); j++) {
         if (slices[i-1].evals(j) >= slices[i-1].shift && slices[i-1].evals(j) < midpt) {
            sliceevals(accslice) = slices[i-1].evals(j);
            sliceresnrms(accslice) = slices[i-1].resnrms(j);
            originds(accslice) = j;
            accslice++;
            accleft++;
         }
      }
      for (int j = 0; j < slices[i].evals.size(); j++) {
         if (slices[i].evals(j) >= midpt && slices[i].evals(j) < slices[i].shift) {
            sliceevals(accslice) = slices[i].evals(j);
            sliceresnrms(accslice) = slices[i].resnrms(j);
            originds(accslice) = j;
            accslice++;
         }
      }

      std::cout << "### slice: " << i << ", right shift: " << slices[i].shift << ", num in slice: " << accslice << ", inertial count: " << inrtcount << std::endl;

      VectorXi resinds = sortinds(sliceresnrms.head(accslice));

      if (acctotal + min(accslice,inrtcount) <= nev) { 
         cout << scientific; 
         for (int j = 0; j < min(accslice,inrtcount); j++) {
            if (resinds[j] < accleft) {
               inds[i-1](lengths(i-1)) = originds(resinds[j]);
               lengths(i-1)++;
               cout << "n: " << acctotal << ", slice: " << i-1 << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals(resinds[j]) << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
            }
            else {
               inds[i](lengths(i)) = originds(resinds[j]);  
               lengths(i)++;
               cout << "n: " << acctotal << ", slice: " << i << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
            }
            acctotal++;
         }
      }
      else {
         VectorXi evalinds = sortinds(sliceevals.head(accslice));
         int j = 0; 
         cout << scientific; 
         bool accept;
         while (acctotal < nev && j < accslice) {
            accept = false;
            for (int l = 0; l < min(accslice,inrtcount); l++){
               if (evalinds[j] == resinds[l]) {
                  accept = true;
               }
            }
            if (accept) {
               if (resinds[j] < slices[i-1].evals.size()) {
                  inds[i-1](lengths(i-1)) = originds(resinds[j]);
                  lengths(i-1)++;
                  cout << "n: " << acctotal << ", slice: " << i-1 << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               else {
                  inds[i](lengths(i)) = originds(resinds[j]);  
                  lengths(i)++;
                  cout << "n: " << acctotal << ", slice: " << i << ", ind: " << originds(resinds[j]) << ", eval: " << sliceevals[resinds[j]] << ", resnrm: " << sliceresnrms[resinds[j]] << endl;
               }
               acctotal++;
            }
            j++;
         }
         break;
      }
   }

   int x = 0;
   for (int i = 0; i < nslices; i++) {
      x += lengths(i);
      tempvec.resize(lengths(i));
      tempvec = inds[i].head(lengths(i));
      inds[i] = tempvec;
   }
}

VectorXi sortinds(VectorXd xs) {
   VectorXi inds(xs.size());

   int numless, equal = 0;

   for (int i = 0; i < m; i++) {
      numless = 0;
      for (int j = 0; j < m; j++) {
         if (resnrms[j] < resnrms[i]) {
            numless++;
         }
         else if (resnrms[j] == resnrms[i] && i != j) {
            if (i < j) {
               numless++; 
            }
         }
      }
      inds[numless] = i;
   }
   return inds;
}
