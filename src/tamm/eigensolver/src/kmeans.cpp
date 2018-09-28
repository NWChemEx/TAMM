#include "utilities.hpp"

void kmeans(VectorXd &shifts, VectorXd &eigvals)
{
    int maxiter = 5, kcluster;
    double tol = 1e-6;
    double dist, mindist, csum;
   
    int nshifts = shifts.size(); 
    int nclusters = nshifts;
    int nev = eigvals.size();

    MatrixXd cluster_sets(nshifts,nev);
    VectorXi cluster_size(nshifts);
    VectorXd cluster_diameter(nshifts);

    std::vector<double> centroids(nshifts);

    // initialize centroids to the input shifts
    for (int i = 0; i < nshifts; i++) centroids[i] = shifts(i);

    for (int iter = 0; iter < maxiter; iter++) {
       // clear previous clusters
       cluster_size.setZero();   
       cluster_sets.setZero();
       cluster_diameter.setZero();   

       // go through all eigenvalues and place them in some cluster
       for (int i = 0; i<nev; i++) {
          // logOFS << " " << i << "th eigenvalue: " << eigvals(i) << endl;
          mindist = std::abs(eigvals(i)-centroids[0]);
          kcluster = 0;
          for (int j = 1; j < nclusters; j++) {
             dist = fabs(eigvals(i)-centroids[j]);
             if (dist < mindist) {
                kcluster = j;
                mindist = dist;
             }
          } 
          // logOFS << "kcluster = " << kcluster << " old centroid = " << centroids[kcluster] << " mindist = " << mindist << endl;
          cluster_size(kcluster)++;
          cluster_sets(kcluster,cluster_size(kcluster)-1) = eigvals(i);
       }
       // logOFS << "cluster_sets = " << endl;
       // for (int i = 0; i < nclusters; i++) {
       //   logOFS << i << ":" << cluster_sets.block(i,0,1,cluster_size(i)) << endl;
       // }

       // compute the new centroid of each non-empty cluster
       int icluster = 0;
       int jmax = 0;
       double maxdiameter = 0.0;
       for (int j = 0; j < nclusters; j++) {
          if (cluster_size(j) <= 0) {
             logOFS << "cluster " << j << " empty " << " old shift = " << centroids[j] << " to be eliminated" << endl;
             // should eliminate the shift and add it to 
             // the cluster with the largest diameter or the largest
             // number of eigenvalues
          }
          else {
             csum = 0.0;
             for (int i = 0; i < cluster_size(j); i++) {
                csum += cluster_sets(j,i); // may want to weight by residual norm
             }
             centroids[icluster] = csum/((double) cluster_size(j));

             cluster_diameter(icluster) = cluster_sets.block(j,0,1,cluster_size(j)).maxCoeff()
                                        - cluster_sets.block(j,0,1,cluster_size(j)).minCoeff();
             if (cluster_diameter(icluster) > maxdiameter) {
                maxdiameter = cluster_diameter(icluster);
                jmax = icluster;       
             }

             logOFS << "new centroid " << icluster << " = " << centroids[icluster] << " cluster ev count = " << cluster_size(j) << " cluster diameter = " << cluster_diameter(icluster) << endl;
             icluster++;
          } /* end if */
       } /*end j */
       logOFS << " jmax = " << jmax << " max diameter = " << maxdiameter << endl;

       // eliminate an empty cluster if there is one
       // and create a new centroid in the cluster with the largest diameter
       nclusters = icluster;
       logOFS << " the number of new clusters = " << nclusters << endl;

       if (nclusters < nshifts) {
          // move the centroid in the cluster (jmax) with the largest diameter
          double lb = cluster_sets.block(jmax,0,1,cluster_size(jmax)).minCoeff();
          double ub = cluster_sets.block(jmax,0,1,cluster_size(jmax)).maxCoeff();
          centroids[jmax] = lb + (ub-lb)/3.0;
          nclusters++;
          centroids[nclusters-1] = ub - (ub-lb)/3.0;
       }
    } /* end main iter loop */

    // sort the centroids and copy back to shifts 
    std::sort(centroids.begin(),centroids.begin()+nclusters);

    for (int i = 0; i < nshifts; i++) shifts(i) = centroids[i];

}

