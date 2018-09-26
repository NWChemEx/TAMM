#include "collectevs.hpp"

void collectevs(int nshifts, SpectralProbe *SPs, 
                VectorXd &evals, VectorXd &resnrms)
{
   int l = 0;
   for (int k = 0; k < nshifts; k++) {
      for (int j = 0; j < SPs[k].nvalid; j++) {
         // cout << "n: " << l << ", slice: " << k << ", ind: " << inds[k](j) << " of " << SPs[k].evals.size() << endl;
         evals(l) = SPs[k].evals(SPs[k].valind(j));
         // evecs.col(l) = SPs[k].evecs.col(inds[k](j));
         resnrms(l) = SPs[k].resnrms(SPs[k].valind(j));
         l++;
      }
   }
}
