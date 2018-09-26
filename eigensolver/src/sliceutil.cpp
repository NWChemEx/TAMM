#include "sliceutil.hpp"

void print_results(ofstream &resultsfile, int iter, VectorXd &evals, 
                   VectorXd & resnrms)
{
   resultsfile << "SCF iteration: " << iter << std::endl;
   resultsfile << scientific;
         
   resultsfile << std::setw(7) << "n" << std::setw(14) << "eval" << std::setw(14) << "resnrm" << std::endl;

   VectorXi evalinds = sortinds(evals);
   for (int j = 0; j < evals.size(); j++) {
      resultsfile << std::setw(7) << j << std::setw(14) << evals(evalinds[j]) << std::setw(14) << resnrms(evalinds[j]) << std::endl;
   }
   resultsfile << std::endl;
}


void collectevs(int nshifts, SpectralProbe *SPs, VectorXi *inds, 
                VectorXd &evals, VectorXd &resnrms)
{
   int l = 0;
   for (int k = 0; k < nshifts; k++) {
      for (int j = 0; j < inds[k].size(); j++) {
         // cout << "n: " << l << ", slice: " << k << ", ind: " << inds[k](j) << " of " << SPs[k].evals.size() << endl;
         evals(l) = SPs[k].evals(inds[k](j));
         // evecs.col(l) = SPs[k].evecs.col(inds[k](j));
         resnrms(l) = SPs[k].resnrms(inds[k](j));
         l++;
      }
   }
}

VectorXi sortinds(VectorXd xs) {
   int m = xs.size();

   VectorXi inds(m);

   int numless, equal = 0;

   for (int i = 0; i < m; i++) {
      numless = 0;
      for (int j = 0; j < m; j++) {
         if (xs[j] < xs[i]) {
            numless++;
         }
         else if (xs[j] == xs[i] && i != j) {
            if (i < j) {
               numless++; 
            }
         }
      }
      inds[numless] = i;
   }
   return inds;
}
