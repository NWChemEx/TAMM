#include "input.h"

namespace ctce {

static Assignment consAddOp(AddOp* add);

static Multiplication consMultOp(MultOp *mult);

static Range2Index range2indices[] = {
  {12, {H1B, H2B, H3B, H4B, H5B, H6B, H7B, H8B, H9B, H10B, H11B, H12B}}, //TO
  {12, {P1B, P2B, P3B, P4B, P5B, P6B, P7B, P8B, P9B, P10B, P11B, P12B}}, //TV
  {0, {}} //TN
};


void input_initialize(int num_ranges, RangeEntry *ranges,
		      int num_indices, IndexEntry *indices,
		      int num_tensors, TensorEntry *tensors,
		      int num_operations, Operation *ops) {
  int inames[RANGE_UB] = {0};

  for(int i=0; i<num_ranges; i++) {
    if(!strcmp(ranges[i].name, OSTR)) {
      ranges[i].rt = TO;
      continue;
    }
    else if(!strcmp(ranges[i].name, VSTR)) {
      ranges[i].rt = TV;
      continue;
    }
    else if(!strcmp(ranges[i].name, NSTR)) {
      ranges[i].rt = TN;
      continue;
    }
    else {
      printf("Unsupported range type %s\n", ranges[i].name);
      exit(1);
    }
  }

  for(int i=0; i<num_indices; i++) {
    RangeType rt = indices[i].range->rt;
    assert(inames[rt]<range2indices[rt].nindices);
    indices[i].index = range2indices[rt].names[inames[rt]++];
  }
	
  for(int i=0; i<num_tensors; i++) {
    RangeType rts[MAX_TENSOR_DIMS];
    for(int j=0; j<tensors[i].ndim; j++) {
      rts[j] = tensors[i].dims[j]->rt;
    }
    /*@BUG: @FIXME: dist_nw is a placeholder. Should be correct before this object is used*/
    /*@BUG: @FIXME: irrep is not set.. Should be correctly set before this object is used*/
    DistType bug_dist = dist_nw;
    int bug_irrep = 0;
    tensors[i].tensor = new Tensor(tensors[i].ndim, tensors[i].nupper, bug_irrep, rts, bug_dist);
  }
}


void input_ops_initialize(int num_ranges, RangeEntry *ranges,
		      int num_indices, IndexEntry *indices,
		      int num_tensors, TensorEntry *tensors,
		      int num_operations, Operation *ops) {
  //distributon, irrep, allocate/attach
  for(int i=0; i<num_operations; i++) {
    switch(ops[i].optype) {
    case OpTypeAdd:
      ops[i].add = consAddOp((AddOp*)ops[i].op_entry);
      break;
    case OpTypeMult:
      ops[i].mult = consMultOp((MultOp*)ops[i].op_entry);
      break;
    default:
      assert(0);
    }
  }
}

  static Assignment consAddOp(AddOp* add) {
    vector<IndexName> aids, cids;
    assert(add);
    assert(add->ta && add->tc);
    assert(add->tc->ndim == add->ta->ndim);
    int ndim = add->tc->ndim;

    assert(ndim > 0);
    aids.resize(ndim);
    cids.resize(ndim);
    for(int i=0; i<ndim; i++) {
      aids[i] = add->ta_ids[i]->index;
      cids[i] = add->tc_ids[i]->index;
    }
    return Assignment(*add->tc->tensor, *add->ta->tensor, add->alpha, cids, aids);
  }


  static Multiplication consMultOp(MultOp *mult) {
    vector<IndexName> aids, bids, cids;
    assert(mult);
    assert(mult->ta && mult->tb && mult->tc);
    
    int cndim = mult->tc->ndim;
    int andim = mult->ta->ndim;
    int bndim = mult->tb->ndim;
    assert(andim+bndim >= cndim);

    aids.resize(andim);
    bids.resize(bndim);
    cids.resize(cndim);

    for(int i=0; i<andim; i++) {
      aids[i] = mult->ta_ids[i]->index;
    }
    for(int i=0; i<bndim; i++) {
      bids[i] = mult->tb_ids[i]->index;
    }
    for(int i=0; i<cndim; i++) {
      cids[i] = mult->tc_ids[i]->index;
    }
    return Multiplication(*mult->tc->tensor, cids, *mult->ta->tensor, aids, *mult->tb->tensor, bids, mult->alpha);
  }
}
