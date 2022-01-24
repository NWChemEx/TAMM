#pragma once

#include <iostream>

namespace tamm {

namespace internal {

/***********************************************************************
 * The code in this file is a part of Global Arrays.
 * https://github.com/GlobalArrays/ga/blob/master/global/src/decomp.c
 * *********************************************************************/

/****************************************************************************
 *--
 *--  double dd_ev evaluates the load balancing ratio as follows:
 *--
 *--  Let n1, n2 ... nd be the extents of the array dimensions
 *--  Let p1, p2 ... pd be the numbers of processes across
 *--      the corresponding dimensions of the process grid
 *--  Let there be npes processes available
 *--
 *--  Load balancing measure = (n1/p1)*...*(nd/pd)*npes
 *--                           ------------------------
 *--                                 n1*n2*...*nd
 *--  The communication volume measure is the sum of
 *--  all monomials (ni/pi) of degree d-1.
 *--
 ****************************************************************************/
static double dd_ev(const int64_t ndims, const std::vector<int64_t> ardims,
             const std::vector<int64_t> pedims) {
  double q, t;
  long   k;
  q = 1.0;
  t = 1.0;
  for(k = 0; k < ndims; k++) {
    q = (ardims[k] / pedims[k]) * pedims[k];
    t = t * (q / (double) ardims[k]);
  }
  return t;
}

/****************************************************************************
 *--
 *--  void dd_su computes the extents of the local block corresponding
 *--  to the element with least global indices, e.g. A[0][0][0].
 *--
 ****************************************************************************/
static void dd_su(const int64_t ndims, const std::vector<int64_t> ardims,
           const std::vector<int64_t> pedims, std::vector<int64_t>& blk) {
  long i;

  for(i = 0; i < ndims; i++) {
    blk[i] = ardims[i] / pedims[i];
    if(blk[i] < 1) blk[i] = 1;
  }
}

/************************************************************************
 *--
 *--  void ddb_ex implements a naive data mapping of a multi-dimensional
 *--  array across a process Cartesian topology. ndims is the number of
 *--  array dimensions. The resulting process grid also has ndims
 *--  dimensions but some of these can be degenerate.
 *--
 *--  Heuristic:   Let d be the number of dimensions of the data array.
 *--  Return that assignment p1, ..., pd that minimizes the communication
 *--  volume measure among those that maximizes the load balancing ratio
 *--  computed by dd_ev.
 *--  The communication volume measure is the sum of
 *--  all monomials (ni/pi) of degree d-1.
 *--
 *--  ddb_ex returns as soon as it has found a process distribution whose
 *--  load balance ratio is at least as large as the value of threshold.
 *--
 *--  This procedure allocates storage for 3*ndims+npes integers.
 *--
 ************************************************************************/
static void ddb_ex(const int64_t ndims, const std::vector<int64_t> ardims, const long npes,
            const double threshold, std::vector<int64_t>& blk, std::vector<int64_t>& pedims) {
  std::vector<int64_t> tdims(ndims);
  std::vector<long>    pdivs;
  long                 npdivs;
  long                 i, j, k;
  long                 bev;
  long                 pc, done;
  std::vector<long>    stack(ndims);
  std::vector<int64_t> tard(ndims);
  long                 r, cev;
  double               clb, blb;

  /*- Quick exit -*/
  if(ndims == 1) {
    pedims[0] = npes;
    blb       = dd_ev(ndims, ardims, pedims);
    dd_su(1, ardims, pedims, blk);
    return;
  }

  /*- Reset array dimensions to reflect granularity -*/
  for(i = 0; i < ndims; i++)
    if(blk[i] < 1) blk[i] = 1;
  for(i = 0; i < ndims; i++) tard[i] = ardims[i] / blk[i];
  for(i = 0; i < ndims; i++)
    if(tard[i] < 1) {
      tard[i] = 1;
      blk[i]  = ardims[i];
    }

  /*- Allocate memory to hold divisors of npes -*/
  npdivs = 1;
  for(i = 2; i <= npes; i++)
    if(npes % i == 0) npdivs += 1;
  pdivs.resize(npdivs);

  /*- Find all divisors of npes -*/
  for(j = 0, i = 1; i <= npes; i++)
    if(npes % i == 0) pdivs[j++] = i;

  /*- Pump priming the exhaustive search -*/
  blb = -1.0;
  bev = 1.0;
  for(i = 0; i < ndims; i++) bev *= tard[i];
  pedims[0] = npes;
  for(i = 1; i < ndims; i++) pedims[i] = 1;
  tdims[0] = 0;
  stack[0] = npes;
  pc       = 0;
  done     = 0;

  /*-  Recursion loop -*/
  do {
    if(pc == ndims - 1) {
      /*- Set the number of processes for the last dimension -*/
      tdims[pc] = stack[pc];

      /*- Evaluate current solution  -*/
      clb = dd_ev(ndims, tard, tdims);
      cev = 0;
      for(k = 0; k < ndims; k++) {
        r = 1;
        for(j = 0; j < ndims; j++) {
          if(j != k) r = r * (tard[j] / tdims[j]);
        }
        cev = cev + r;
      }
      if(clb > blb || (clb == blb && cev < bev)) {
        for(j = 0; j < ndims; j++) pedims[j] = tdims[j];
        blb = clb;
        bev = cev;
      }
      if(blb > threshold) break;
      tdims[pc] = 0;
      pc -= 1;
    }
    else {
      if(tdims[pc] == stack[pc]) {
        /*- Backtrack when current array dimension has exhausted
         *- all remaining processes
         */
        done      = (pc == 0);
        tdims[pc] = 0;
        pc -= 1;
      }
      else {
        /*- Increment the number of processes assigned to the current
         *- array axis.
         */
        for(tdims[pc] += 1; stack[pc] % tdims[pc] != 0; tdims[pc] += 1)
          ;
        pc += 1;
        stack[pc] = npes;
        for(i = 0; i < pc; i++) stack[pc] /= tdims[i];
        tdims[pc] = 0;
      }
    }
  } while(!done);

  dd_su(ndims, ardims, pedims, blk);
}

/**
 * @brief Compute an effective processor grid for the given number of ranks.
 * Note that not all ranks provided might be used.
 *
 * @param tensor_structure Tensor for which the grid is to be computed
 * @param nproc Number of ranks available
 * @return std::vector<Proc> The processor grid
 *
 * @post Product of the grid size along all dimensions is less than @param
 * nproc
 */

static std::vector<int64_t> compute_proc_grid(const int64_t              ndims,
                                              const std::vector<int64_t> ardims, const int64_t npes,
                                              double threshold, const int64_t bias) {
  std::vector<int64_t> pedims(ndims, 1);
  std::vector<int64_t> blk(ndims, -1);

  // if (ndim > 0) {
  //   proc_grid[0] = nproc;
  //   if(nproc > dim1_size) proc_grid[0] = dim1_size;
  // }

  /************************************************************************
   *--
   *-- void ddb_h2 lists all prime divisors of the number of
   *-- processes and distribute these among the array dimensions.
   *-- If the value of objective function attained with this heuristic
   *-- is less than threshold then an exhaustive search is performed.
   *-- The argument bias directs the search of ddb_h2. When bias is
   *-- positive the rightmost axes of the data array are preferentially
   *-- distributed, similarly when bias is negative. When bias is zero
   *-- the heuristic attempts to deal processes equally among the axes
   *-- of the data array.
   *--
   *-- ddb_h2 allocates storage for ndims+npes integers and may call ddb_ex.
   *--
   ************************************************************************/

  {
    long                 h, i, j, k;
    std::vector<long>    pdivs;
    std::vector<int64_t> tard(ndims);
    long                 npdivs;
    long                 p0;
    double               q, w;
    double               ub;
    long                 istart, istep, ilook;

    /*- Reset array dimensions to reflect granularity -*/
    for(i = 0; i < ndims; i++)
      if(blk[i] < 1) blk[i] = 1;

    for(i = 0; i < ndims; i++) tard[i] = (ardims[i] + blk[i] - 1) / blk[i]; /* JM */
    for(i = 0; i < ndims; i++)
      if(tard[i] < 1) {
        tard[i] = 1;
        blk[i]  = ardims[i];
      }

    /*- Allocate storage to old all divisors of npes -*/
    npdivs = 1;
    for(i = 2; i <= npes; i++)
      if(npes % i == 0) npdivs += 1;
    pdivs.resize(npdivs);

    /*- Find all divisors of npes -*/
    for(j = 0, i = 1; i <= npes; i++)
      if(npes % i == 0) pdivs[j++] = i;

    /*- Find all prime divisors of npes (with repetitions) -*/
    if(npdivs > 1) {
      k = 1;
      do {
        h = k + 1;
        for(j = h; j < npdivs; j++)
          if(pdivs[j] % pdivs[k] == 0) pdivs[h++] = pdivs[j] / pdivs[k];
        npdivs = h;
        k      = k + 1;
      } while(k < npdivs);
    }

    /*- Set istart and istep -*/
    istep  = 1;
    istart = 0;
    if(bias > 0) {
      istep  = -1;
      istart = ndims - 1;
    }
    /*- Set pedims -*/
    for(j = 0; j < ndims; j++) pedims[j] = 1.0;
    for(k = npdivs - 1; k >= 1; k--) {
      p0 = pdivs[k];
      h  = istart;
      q  = (tard[istart] < p0 * pedims[istart])
             ? 1.1
             : (tard[istart] % (p0 * pedims[istart])) / (double) tard[istart];
      for(j = 1; j < ndims; j++) {
        ilook = (istart + istep * j) % ndims;
        w     = (tard[ilook] < p0 * pedims[ilook])
                  ? 1.1
                  : (tard[ilook] % (p0 * pedims[ilook])) / (double) tard[ilook];
        if(w < q) {
          q = w;
          h = ilook;
        }
      }
      pedims[h] *= p0;
      if(bias == 0) istart = (istart + 1) % ndims;
    }

    ub = dd_ev(ndims, tard, pedims);

    /*- Do an exhaustive search is the heuristic returns a solution
     *- whose load balance ratio is less than the given threshold. -*/
    if(ub < threshold) { ddb_ex(ndims, tard, npes, threshold, blk, pedims); }

    dd_su(ndims, ardims, pedims, blk);

    for(i = 0; i < ndims; i++)
      if(pedims[i] > 0) { blk[i] = (tard[i] + pedims[i] - 1) / pedims[i]; }
      else {
        // tamm_terminate("process dimension is zero: ddb_h2");
      }
  }

  return pedims;
}

} // namespace internal
} // namespace tamm
