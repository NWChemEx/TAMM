//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include "tensor/schedulers.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "tensor/equations.h"
#include "tensor/expression.h"
#include "tensor/input.h"
#include "tensor/tensor.h"

using std::vector;
using std::cout;
using std::endl;

namespace tamm {

void lazy_tensor_alloc(std::map<std::string, tamm::Tensor> *tensors,
                       std::vector<Operation> *ops,
                       std::vector<vector<Operation *> > *op_levels,
                       std::vector<vector<Tensor *> > *tensor_create_levels,
                       std::vector<vector<Tensor *> > *tensor_destroy_levels);
void eager_tensor_alloc(std::map<std::string, tamm::Tensor> *tensors,
                        std::vector<Operation> *ops,
                        const std::vector<vector<Operation *> > &op_levels,
                        std::vector<vector<Tensor *> > *tensor_create_levels,
                        std::vector<vector<Tensor *> > *tensor_destroy_levels);
static void schedule(vector<std::map<std::string, tamm::Tensor> *> *tensors,
                     vector<vector<Operation> *> *ops,
                     vector<vector<vector<Tensor *> > > *tensor_create_levels,
                     vector<vector<vector<Tensor *> > > *tensor_destroy_levels,
                     vector<vector<vector<Operation *> > > *op_levels);

static void execute(Operation *op, gmem::Handle sync_ga, int spos);
static int writes(const Operation &op,
                  const std::map<std::string, tamm::Tensor> &tensors);
static vector<int> reads(const Operation &op,
                         const std::map<std::string, tamm::Tensor> &tensors);
static void schedule(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops,
                     std::vector<vector<Tensor *> > *tensor_create_levels,
                     std::vector<vector<Tensor *> > *tensor_destroy_levels,
                     std::vector<vector<Operation *> > *op_levels);

static int find_tensor(const std::map<std::string, tamm::Tensor> &tensors,
                       const Tensor &t) {
  int pos = 0;
  for (std::map<std::string, tamm::Tensor>::const_iterator i = tensors.begin();
       i != tensors.end(); i++) {
    const Tensor *entry = &i->second;
    if (&t == entry) {
      return pos;
    }
    pos++;
  }
  return -1;
}

/**
 * Execution operations in input order. Allocate just before
 * definition. Deallocate right after last use.
 */
void schedule_linear_lazy(std::map<std::string, tamm::Tensor> *tensors,
                          std::vector<Operation> *ops) {
  std::vector<vector<Tensor *> > tensor_create_levels;
  std::vector<vector<Tensor *> > tensor_destroy_levels;
  std::vector<vector<Operation *> > op_levels;

  op_levels.resize(ops->size());
  for (int i = 0; i < ops->size(); i++) {
    op_levels[i].push_back(&(*ops)[i]);
  }
  lazy_tensor_alloc(tensors, ops, &op_levels, &tensor_create_levels,
                    &tensor_destroy_levels);

  schedule(tensors, ops, &tensor_create_levels, &tensor_destroy_levels,
           &op_levels);
}

void eager_tensor_alloc(std::map<std::string, tamm::Tensor> *tensors,
                        std::vector<Operation> *ops,  // Not used
                        const std::vector<vector<Operation *> > &op_levels,
                        std::vector<vector<Tensor *> > *tensor_create_levels,
                        std::vector<vector<Tensor *> > *tensor_destroy_levels) {
  tensor_create_levels->clear();
  tensor_destroy_levels->clear();
  tensor_create_levels->resize(op_levels.size());
  tensor_destroy_levels->resize(op_levels.size());

  // for(int i=0; i<tensors.size(); i++) {
  for (std::map<std::string, tamm::Tensor>::iterator i = tensors->begin();
       i != tensors->end(); i++) {
    Tensor *t = &(*tensors)[i->first];
    if (!t->attached() && !t->allocated()) {
      (*tensor_create_levels)[0].push_back(t);
      (*tensor_destroy_levels)[op_levels.size() - 1].push_back(t);
    }
  }
}

void lazy_tensor_alloc(std::map<std::string, tamm::Tensor> *tensors,
                       std::vector<Operation> *ops,  // Not used
                       std::vector<vector<Operation *> > *op_levels,
                       std::vector<vector<Tensor *> > *tensor_create_levels,
                       std::vector<vector<Tensor *> > *tensor_destroy_levels) {
  std::vector<int> first_def(tensors->size(), op_levels->size()),
      last_use(tensors->size(), -1);

  int ta, tb, tc;
  for (int i = 0; i < op_levels->size(); i++) {
    for (int j = 0; j < (*op_levels)[i].size(); j++) {
      Operation *op = (*op_levels)[i][j];
      int wa = writes(*op, *tensors);
      vector<int> rds = reads(*op, *tensors);
      first_def[wa] = std::min(i, first_def[wa]);
      for (int r = 0; r < rds.size(); r++) {
        last_use[rds[r]] = i;
      }
    }
  }

  tensor_create_levels->clear();
  tensor_create_levels->resize(op_levels->size());
  tensor_destroy_levels->clear();
  tensor_destroy_levels->resize(op_levels->size());

  // for(int i=0; i<tensors.size(); i++) {

  int id = 0;
  for (std::map<std::string, tamm::Tensor>::iterator i = tensors->begin();
       i != tensors->end(); i++) {
    Tensor *t = &i->second;
    int pos = id;
    id++;

    if (t->attached() || t->allocated()) {
      continue;
    }
    if (first_def[pos] == op_levels->size()) {
      assert(last_use[pos] == -1);
      continue;
    }
    int fd = first_def[pos];
    assert(fd >= 0 && fd < op_levels->size());
    (*tensor_create_levels)[fd].push_back(t);
    int lu = last_use[pos];
    if (!(lu >= 0 && lu < op_levels->size())) {
      cout << "ABOUT TO THROW FOR tensor " << i->first << endl;
      cout << "Last use=" << lu << endl;
    }
    assert(lu >= 0 && lu < op_levels->size());
    (*tensor_destroy_levels)[lu].push_back(t);
  }
}

bool has_dependence(const std::vector<Operation> &ops,
                    const std::map<std::string, tamm::Tensor> &tensors, int i,
                    int j) {
  int iw = writes(ops[i], tensors);
  vector<int> irds = reads(ops[i], tensors);
  int jw = writes(ops[j], tensors);
  vector<int> jrds = reads(ops[j], tensors);
  if (std::find(irds.begin(), irds.end(), jw) != irds.end() ||
      std::find(jrds.begin(), jrds.end(), iw) != jrds.end()) {
    return true;
  }
  return false;
}

void levelize(const std::map<std::string, tamm::Tensor> &tensors,
              std::vector<Operation> *ops,
              vector<vector<Operation *> > *levels) {
  int n = ops->size();
  int level_id[n], max_level = 0;
  for (int i = 0; i < n; i++) {
    int l = 0;
    for (int j = 0; j < i; j++) {
      if (has_dependence(*ops, tensors, i, j)) {
        l = std::max(l, level_id[j] + 1);
      }
    }
    level_id[i] = l;
    max_level = std::max(max_level, l);
  }
  levels->clear();
  levels->resize(max_level + 1);
  for (int i = 0; i < ops->size(); i++) {
    (*levels)[level_id[i]].push_back(&(*ops)[i]);
  }
}

void schedule_levels(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops) {
  vector<vector<Operation *> > op_levels;
  std::vector<Tensor *> created_tensors;
  std::vector<vector<Tensor *> > tensor_create_levels;
  std::vector<vector<Tensor *> > tensor_destroy_levels;

  levelize(*tensors, ops, &op_levels);
  lazy_tensor_alloc(tensors, ops, &op_levels, &tensor_create_levels,
                    &tensor_destroy_levels);
  schedule(tensors, ops, &tensor_create_levels, &tensor_destroy_levels,
           &op_levels);
}

void schedule_levels(
    std::vector<std::map<std::string, tamm::Tensor> *> *tensors_lst,
    std::vector<std::vector<Operation> *> *ops_lst) {
  using std::vector;
  vector<vector<vector<Operation *> > > op_levels;
  vector<vector<vector<Tensor *> > > tensor_create_levels;
  vector<vector<vector<Tensor *> > > tensor_destroy_levels;

  assert(ops_lst->size() == tensors_lst->size());

  op_levels.resize(ops_lst->size());
  tensor_create_levels.resize(ops_lst->size());
  tensor_destroy_levels.resize(ops_lst->size());
  for (int e = 0; e < ops_lst->size(); e++) {
    levelize(*(*tensors_lst)[e], (*ops_lst)[e], &op_levels[e]);
    lazy_tensor_alloc((*tensors_lst)[e], (*ops_lst)[e], &op_levels[e],
                      &tensor_create_levels[e], &tensor_destroy_levels[e]);
  }
  schedule(tensors_lst, ops_lst, &tensor_create_levels, &tensor_destroy_levels,
           &op_levels);
}

static void execute(Operation *op, gmem::Handle sync_ga, int spos) {
  switch (op->optype) {
    case OpTypeAdd:
      op->add.execute(sync_ga, spos);
      break;
    case OpTypeMult:
      op->mult.execute(sync_ga, spos);
      break;
    default:
      printf("Unsupported operation type\n");
      assert(0);
  }
}

static int writes(const Operation &op,
                  const std::map<std::string, tamm::Tensor> &tensors) {
  // assert(op);
  int wa;
  switch (op.optype) {
    case OpTypeAdd:
      // wa = &op->add.tC() - &tensors[0];
      wa = find_tensor(tensors, op.add.tC());
      break;
    case OpTypeMult:
      // wa = &op->mult.tC()  - &tensors[0];
      wa = find_tensor(tensors, op.mult.tC());
      break;
    default:
      assert(0);
  }
  return wa;
}

static vector<int> reads(const Operation &op,
                         const std::map<std::string, tamm::Tensor> &tensors) {
  // assert(op);
  vector<int> rds;
  int ra1, ra2;
  switch (op.optype) {
    case OpTypeAdd:
      // ra1 = &op->add.tA() - &tensors[0];
      ra1 = find_tensor(tensors, op.add.tA());
      rds.push_back(ra1);
      break;
    case OpTypeMult:
      // ra1 = &op->mult.tA()  - &tensors[0];
      // ra2 = &op->mult.tB()  - &tensors[0];
      ra1 = find_tensor(tensors, op.mult.tA());
      ra2 = find_tensor(tensors, op.mult.tB());
      rds.push_back(ra1);
      rds.push_back(ra2);
      break;
    default:
      assert(0);
  }
  return rds;
}

static void schedule(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops,
                     vector<vector<Tensor *> > *tensor_create_levels,
                     std::vector<vector<Tensor *> > *tensor_destroy_levels,
                     std::vector<vector<Operation *> > *op_levels) {
#if 1
  vector<std::map<std::string, tamm::Tensor> *> tensors_lst(1, tensors);
  vector<vector<Operation> *> ops_lst(1, ops);
  vector<vector<vector<Tensor *> > > tcl(1, *tensor_create_levels);
  vector<vector<vector<Tensor *> > > tdl(1, *tensor_destroy_levels);
  vector<vector<vector<Operation *> > > ol(1, *op_levels);

  schedule(&tensors_lst, &ops_lst, &tcl, &tdl, &ol);
#else
  int nlevels = op_levels.size();
  assert(tensor_create_levels.size() == nlevels);
  assert(tensor_destroy_levels.size() == nlevels);

  vector<gmem::handle> sync_gas;
  for (int i = 0; i < op_levels.size(); i++) {
    assert(op_levels[i].size() > 0);
    int taskDim = op_levels[i].size();
    char taskStr[10] = "NXTASK";
    gmem::Handle taskHandle =
        gmem::create(Int, taskDim, taskStr);  // global array for next task
    gmem::zero(taskHandle);                   // initialize to zero
    sync_gas.push_back(taskHandle);
  }
  gmem::sync();

  for (int i = 0; i < nlevels; i++) {
    for (int j = 0; j < tensor_create_levels[i].size(); j++) {
      tensor_create_levels[i][j]->create();
    }
    for (int j = 0; j < op_levels[i].size(); j++) {
      execute(op_levels[i][j], sync_gas[i], j);
    }
    for (int j = 0; j < tensor_destroy_levels[i].size(); j++) {
      tensor_destroy_levels[i][j]->destroy();
    }
  }
  gmem::sync();
  for (int i = 0; i < sync_gas.size(); i++) {
    gmem::destroy(sync_gas[i]);
  }
#endif  // if 1
}

static void schedule(vector<std::map<std::string, tamm::Tensor> *> *tensors,
                     vector<vector<Operation> *> *ops,  // Not used
                     vector<vector<vector<Tensor *> > > *tensor_create_levels,
                     vector<vector<vector<Tensor *> > > *tensor_destroy_levels,
                     vector<vector<vector<Operation *> > > *op_levels) {
  int neqs = op_levels->size();
  assert(tensor_create_levels->size() == tensor_destroy_levels->size());
  assert(op_levels->size() == tensor_create_levels->size());

  size_t nlevels = 0;
  for (int e = 0; e < neqs; e++) {
    nlevels = std::max(nlevels, (*op_levels)[e].size());
    assert((*op_levels)[e].size() == (*tensor_create_levels)[e].size());
    assert((*op_levels)[e].size() == (*tensor_destroy_levels)[e].size());
  }
  for (int e = 0; e < neqs; e++) {
    (*op_levels)[e].resize(nlevels);
    (*tensor_create_levels)[e].resize(nlevels);
    (*tensor_destroy_levels)[e].resize(nlevels);
  }

  vector<gmem::Handle> sync_gas;
  for (int l = 0; l < nlevels; l++) {
    int taskDim = 0;
    for (int e = 0; e < neqs; e++) {
      taskDim += (*op_levels)[e][l].size();
      assert((*op_levels)[e][l].size() >= 0);
    }
    char taskStr[10] = "NXTASK";
    gmem::Handle taskHandle = gmem::create(
        gmem::Int, taskDim, taskStr);  // global array for next task
    gmem::zero(taskHandle);            // initialize to zero
    sync_gas.push_back(taskHandle);
  }
  gmem::sync();

  for (int l = 0; l < nlevels; l++) {
    for (int e = 0; e < neqs; e++) {
      for (int t = 0; t < (*tensor_create_levels)[e][l].size(); t++) {
        (*tensor_create_levels)[e][l][t]->create();
      }
    }
    gmem::sync();
    for (int e = 0, c = 0; e < neqs; e++) {
      for (int o = 0; o < (*op_levels)[e][l].size(); o++, c++) {
        execute((*op_levels)[e][l][o], sync_gas[l], c);
      }
    }
    gmem::sync();
    for (int e = 0; e < neqs; e++) {
      for (int t = 0; t < (*tensor_destroy_levels)[e][l].size(); t++) {
        (*tensor_destroy_levels)[e][l][t]->destroy();
      }
    }
  }
  gmem::sync();
  for (int i = 0; i < sync_gas.size(); i++) {
    gmem::destroy(sync_gas[i]);
  }
}

/**
 * Allocate all intermediate arrays upfront. Execute operations in
 * input order. Deallocate all allocated arrays at the end.
 */
void schedule_linear(std::map<std::string, tamm::Tensor> *tensors,
                     std::vector<Operation> *ops) {
  std::vector<vector<Tensor *> > tensor_create_levels;
  std::vector<vector<Tensor *> > tensor_destroy_levels;
  std::vector<vector<Operation *> > op_levels;

  op_levels.resize(ops->size());
  for (int i = 0; i < ops->size(); i++) {
    op_levels[i].push_back(&(*ops)[i]);
  }
  eager_tensor_alloc(tensors, ops, op_levels, &tensor_create_levels,
                     &tensor_destroy_levels);
  schedule(tensors, ops, &tensor_create_levels, &tensor_destroy_levels,
           &op_levels);
}
}  // namespace tamm
