#include <vector>
#include <map>
#include <set>
#include "tensor.h"
#include "expression.h"
#include "input.h"
#include "equations.h"

namespace ctce {

  void lazy_tensor_alloc(std::vector<Tensor> &tensors,
                         std::vector<Operation> &ops,
                         std::vector<vector<Operation *> > op_levels,
                         std::vector<vector<Tensor*> > &tensor_create_levels,
                         std::vector<vector<Tensor*> > &tensor_destroy_levels);
  void eager_tensor_alloc(std::vector<Tensor> &tensors,
                         std::vector<Operation> &ops,
                         std::vector<vector<Operation *> > op_levels,
                         std::vector<vector<Tensor*> > &tensor_create_levels,
                         std::vector<vector<Tensor*> > &tensor_destroy_levels);

  static void execute(Operation *op);
  static int writes(Operation *op, std::vector<Tensor> &tensors);
  static vector<int> reads(Operation *op, std::vector<Tensor> &tensors);
  static void schedule(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops,
                       std::vector<vector<Tensor*> > &tensor_create_levels,
                       std::vector<vector<Tensor*> > &tensor_destroy_levels,
                       std::vector<vector<Operation *> > &op_levels);
  /**
   * Execution operations in input order. Allocate just before
   * definition. Deallocate right after last use.
   */
  void schedule_linear_lazy(std::vector<Tensor> &tensors,
                            std::vector<Operation> &ops) {
    std::vector<vector<Tensor*> > tensor_create_levels;
    std::vector<vector<Tensor*> > tensor_destroy_levels;
    std::vector<vector<Operation *> > op_levels;
    
    op_levels.resize(ops.size());
    for(int i=0; i<ops.size(); i++) {
      op_levels[i].push_back(&ops[i]);
    }
    lazy_tensor_alloc(tensors, ops, op_levels, tensor_create_levels, tensor_destroy_levels);

    schedule(tensors, ops, tensor_create_levels, tensor_destroy_levels, op_levels);
  }


  void eager_tensor_alloc(std::vector<Tensor> &tensors,
                          std::vector<Operation> &ops,
                          std::vector<vector<Operation *> > op_levels,
                          std::vector<vector<Tensor*> > &tensor_create_levels,
                          std::vector<vector<Tensor*> > &tensor_destroy_levels) {
    tensor_create_levels.clear();
    tensor_destroy_levels.clear();
    tensor_create_levels.resize(op_levels.size());
    tensor_destroy_levels.resize(op_levels.size());

    for(int i=0; i<tensors.size(); i++) {
      Tensor *t = &tensors[i];
      if(!t->attached() && !t->allocated()) {
        tensor_create_levels[0].push_back(t);
        tensor_destroy_levels[op_levels.size()-1].push_back(t);
      }
    }
  }

  void lazy_tensor_alloc(std::vector<Tensor> &tensors,
                         std::vector<Operation> &ops,
                         std::vector<vector<Operation *> > op_levels,
                         std::vector<vector<Tensor*> > &tensor_create_levels,
                         std::vector<vector<Tensor*> > &tensor_destroy_levels) {
    std::vector<int> first_def(tensors.size(), op_levels.size()), last_use(tensors.size(), -1);

    int ta, tb, tc;
    for(int i=0; i<op_levels.size(); i++) {
      for(int j=0; j<op_levels[i].size(); j++) {
        Operation *op = op_levels[i][j];
        int wa = writes(op, tensors);
        vector<int> rds = reads(op, tensors);
        first_def[wa] = std::min(i,first_def[wa]);
        for(int r=0; r<rds.size(); r++) {
          last_use[rds[r]] = i;
        }
      }
    }

    tensor_create_levels.clear(); 
    tensor_create_levels.resize(op_levels.size());
    tensor_destroy_levels.clear(); 
    tensor_destroy_levels.resize(op_levels.size());

    for(int i=0; i<tensors.size(); i++) {
      Tensor &t = tensors[i];
      if(t.attached() || t.allocated()) {
        continue;
      }
      if(first_def[i] == ops.size()) {
        assert(last_use[i] == -1);
        continue;
      }
      int fd = first_def[i];
      assert(fd>=0 && fd<op_levels.size());
      tensor_create_levels[fd].push_back(&t);
      int lu = last_use[i];
      assert(lu>=0 && lu<op_levels.size());
      tensor_destroy_levels[lu].push_back(&t);
    }
  }


  bool has_dependence(std::vector<Operation> &ops,
                      std::vector<Tensor> &tensors,
                      int i, int j) {
    int iw = writes(&ops[i], tensors);
    vector<int> irds = reads(&ops[i], tensors);
    int jw = writes(&ops[j], tensors);
    vector<int> jrds = reads(&ops[j], tensors);
    if(std::find(irds.begin(), irds.end(), jw) != irds.end() ||
       std::find(jrds.begin(), jrds.end(), iw) != jrds.end()) {
      return true;
    }
    return false;
  }

  void levelize(std::vector<Tensor> &tensors,
                std::vector<Operation> &ops,
                vector<vector<Operation *> > &levels) {
    int n = ops.size();
    int level_id[n], max_level=0;
    for(int i=0; i<n; i++) {
      int l = 0;
      for(int j=0; j<i; j++) {
        if(has_dependence(ops,tensors,i,j)) {
          l = max(l, level_id[j]+1);
        }
      }
      level_id[i] = l;
      max_level = max(max_level, l);
    }
    levels.clear();
    levels.resize(max_level+1);
    for(int i=0; i<ops.size(); i++) {
      levels[level_id[i]].push_back(&ops[i]);
    }
  }

  void schedule_levels(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops) {
    using std::vector;
    vector<vector<Operation *> > op_levels;
    std::vector<Tensor*> created_tensors;
    std::vector<vector<Tensor*> > tensor_create_levels;
    std::vector<vector<Tensor*> > tensor_destroy_levels;

    levelize(tensors, ops, op_levels);
    lazy_tensor_alloc(tensors, ops, op_levels, tensor_create_levels, tensor_destroy_levels);
    schedule(tensors, ops, tensor_create_levels, tensor_destroy_levels, op_levels);
  }

  static void execute(Operation *op) {
    switch(op->optype) {
    case OpTypeAdd:
      op->add.execute();
      break;
    case OpTypeMult:
      op->mult.execute();
      break;
    default:
      printf("Unsupported operation type\n");
      assert(0);
    }
  }

  static int writes(Operation *op, std::vector<Tensor> &tensors) {
    assert(op);
    int wa=-1;
    switch(op->optype) {
    case OpTypeAdd:
      wa = &op->add.tC() - &tensors[0];
      break;
    case OpTypeMult:
      wa = &op->mult.tC()  - &tensors[0];
      break;
    default:
      assert(0);
    }
    return wa;
  }

  static vector<int> reads(Operation *op, std::vector<Tensor> &tensors) {
    assert(op);
    vector<int> rds;
    int ra1, ra2;
    switch(op->optype) {
    case OpTypeAdd:
      ra1 = &op->add.tA() - &tensors[0];
      rds.push_back(ra1);
      break;
    case OpTypeMult:
      ra1 = &op->mult.tA()  - &tensors[0];
      ra2 = &op->mult.tB()  - &tensors[0];
      rds.push_back(ra1);
      rds.push_back(ra2);
      break;
    default:
      assert(0);
    }
    return rds;
  }

  static void schedule(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops,
                       std::vector<vector<Tensor*> > &tensor_create_levels,
                       std::vector<vector<Tensor*> > &tensor_destroy_levels,
                       std::vector<vector<Operation *> > &op_levels) {
    int nlevels = op_levels.size();
    assert(tensor_create_levels.size() == tensor_destroy_levels.size());
    assert(op_levels.size() == tensor_create_levels.size());

    for(int i=0; i<nlevels; i++) {
      for(int j=0; j<tensor_create_levels[i].size(); j++) {
        tensor_create_levels[i][j]->create();
      }
      for(int j=0; j<op_levels[i].size(); j++) {
        execute(op_levels[i][j]);
      }
      for(int j=0; j<tensor_destroy_levels[i].size(); j++) {
        tensor_destroy_levels[i][j]->destroy();
      }
    }
  }

  /**
   * Allocate all intermediate arrays upfront. Execute operations in
   * input order. Deallocate all allocated arrays at the end.
   */
  void schedule_linear(std::vector<Tensor> &tensors,
                       std::vector<Operation> &ops) {
    std::vector<vector<Tensor*> > tensor_create_levels;
    std::vector<vector<Tensor*> > tensor_destroy_levels;
    std::vector<vector<Operation *> > op_levels;

    op_levels.resize(ops.size());
    for(int i=0; i<ops.size(); i++) {
      op_levels[i].push_back(&ops[i]);
    }
    eager_tensor_alloc(tensors, ops, op_levels, tensor_create_levels, tensor_destroy_levels);
    schedule(tensors, ops, tensor_create_levels, tensor_destroy_levels, op_levels);
  }
}

