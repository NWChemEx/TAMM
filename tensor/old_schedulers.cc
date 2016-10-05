#include <map>
#include <set>
#include <vector>
#include "equations.h"
#include "expression.h"
#include "input.h"
#include "tensor.h"

namespace tamm {
/**
 * Allocate all intermediate arrays upfront. Execute operations in
 * input order. Deallocate all allocated arrays at the end.
 */
void schedule_linear(std::vector<Tensor> &tensors,
                     std::vector<Operation> &ops) {
  std::vector<Tensor *> created_tensors;

  for (int i = 0; i < tensors.size(); i++) {
    Tensor &t = tensors[i];
    if (!t.attached() && !t.allocated()) {
      t.create();
      created_tensors.push_back(&t);
    }
  }

  for (int i = 0; i < ops.size(); i++) {
    switch (ops[i].optype) {
      case OpTypeAdd:
        ops[i].add.execute();
        break;
      case OpTypeMult:
        ops[i].mult.execute();
        break;
      default:
        printf("Unsupported operation type\n");
        assert(0);
    }
  }

  for (size_t i = 0; i < created_tensors.size(); i++) {
    created_tensors[i]->destroy();
  }
}

/**
 * Execution operations in input order. Allocate just before
 * definition. Deallocate right after last use.
 */
void schedule_linear_lazy(std::vector<Tensor> &tensors,
                          std::vector<Operation> &ops) {
  std::vector<std::set<Tensor *> > create_tensor(ops.size()),
      destroy_tensor(ops.size());
  std::vector<int> first_def(tensors.size(), ops.size()),
      last_use(tensors.size(), -1);

  int ta, tb, tc;
  for (int i = 0; i < ops.size(); i++) {
    switch (ops[i].optype) {
      case OpTypeAdd:
        ta = &ops[i].add.tA() - &tensors[0];
        tc = &ops[i].add.tC() - &tensors[0];
        assert(ta >= 0 && ta < tensors.size());
        assert(tc >= 0 && tc < tensors.size());
        last_use[ta] = i;
        first_def[tc] = std::min(i, first_def[tc]);
        break;
      case OpTypeMult:
        ta = &ops[i].mult.tA() - &tensors[0];
        tb = &ops[i].mult.tB() - &tensors[0];
        tc = &ops[i].mult.tC() - &tensors[0];
        assert(ta >= 0 && ta < tensors.size());
        assert(tb >= 0 && tb < tensors.size());
        assert(tc >= 0 && tc < tensors.size());
        last_use[ta] = i;
        last_use[tb] = i;
        first_def[tc] = std::min(i, first_def[tc]);
        break;
      default:
        printf("Unsupported operation type\n");
        assert(0);
    }
  }

  for (int i = 0; i < tensors.size(); i++) {
    Tensor &t = tensors[i];
    if (t.attached() || t.allocated()) {
      // cout<<&t<<" already allocated or attached"<<endl;
      continue;
    }
    if (first_def[i] == ops.size()) {
      assert(last_use[i] == -1);
      continue;
    }
    int fd = first_def[i];
    assert(fd >= 0 && fd < ops.size());
    create_tensor[fd].insert(&t);
    if (last_use[i] == -1) {
      assert(0);
      // defined but never used
      destroy_tensor[fd].insert(&t);
    } else {
      int lu = last_use[i];
      assert(fd <= lu);
      assert(lu < ops.size());
      destroy_tensor[lu].insert(&t);
    }
  }

  // std::vector<Tensor*> created_tensors;
  // for(int i=0; i<tensors.size(); i++) {
  //   Tensor &t = tensors[i];
  //   if(!t.attached() && !t.allocated()) {
  //     t.create();
  //     created_tensors.push_back(&t);
  //   }
  // }

  for (int i = 0; i < ops.size(); i++) {
    for (std::set<Tensor *>::iterator itr = create_tensor[i].begin();
         itr != create_tensor[i].end(); itr++) {
      Tensor *t = *itr;
      assert(!t->attached() && !t->allocated());
      t->create();
      // created_tensors.push_back(t);
    }

    switch (ops[i].optype) {
      case OpTypeAdd:
        ops[i].add.execute();
        break;
      case OpTypeMult:
        ops[i].mult.execute();
        break;
      default:
        printf("Unsupported operation type\n");
        assert(0);
    }

    for (std::set<Tensor *>::iterator itr = destroy_tensor[i].begin();
         itr != destroy_tensor[i].end(); itr++) {
      Tensor *t = *itr;
      t->destroy();
    }
  }

  // for(size_t i=0; i<created_tensors.size(); i++) {
  //   created_tensors[i]->destroy();
  // }
}

void levelize(std::vector<Tensor> &tensors, std::vector<Operation> &ops,
              vector<vector<Operation *> > &levels) {
  {
    int n = ops.size();
    int deps[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        deps[i][j] = 0;
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        Tensor *ra1 = NULL, *ra2 = NULL, *wa = NULL, *rb1 = NULL, *rb2 = NULL,
               *wb = NULL;
        switch (ops[i].optype) {
          case OpTypeAdd:
            wa = &ops[i].add.tC();
            ra1 = &ops[i].add.tA();
            break;
          case OpTypeMult:
            wa = &ops[i].mult.tC();
            ra1 = &ops[i].mult.tA();
            ra2 = &ops[i].mult.tB();
            break;
          default:
            assert(0);
        }
        switch (ops[j].optype) {
          case OpTypeAdd:
            wb = &ops[j].add.tC();
            rb1 = &ops[j].add.tA();
            break;
          case OpTypeMult:
            wb = &ops[j].mult.tC();
            rb1 = &ops[j].mult.tA();
            rb2 = &ops[j].mult.tB();
            break;
          default:
            assert(0);
        }
        if (ra1 == wb || ra2 == wb || rb1 == wa || rb2 == wa) {
          deps[i][j] = 1;
        }
      }
    }
    int level_id[n], max_level = 0;
    for (int i = 0; i < n; i++) {
      int l = 0;
      for (int j = 0; j < i; j++) {
        if (deps[i][j]) {
          l = max(l, level_id[j] + 1);
        }
      }
      level_id[i] = l;
      max_level = max(max_level, l);
      // cout<<"level["<<i<<"]="<<l<<endl;
    }
    levels.clear();
    levels.resize(max_level + 1);
    for (int i = 0; i < ops.size(); i++) {
      levels[level_id[i]].push_back(&ops[i]);
    }
  }
}

void schedule_levels(std::vector<Tensor> &tensors,
                     std::vector<Operation> &ops) {
  using std::vector;
  vector<vector<Operation *> > levels;
  std::vector<Tensor *> created_tensors;

  levelize(tensors, ops, levels);

  // cout<<"nlevels="<<levels.size()<<endl;
  for (int i = 0; i < tensors.size(); i++) {
    Tensor &t = tensors[i];
    if (!t.attached() && !t.allocated()) {
      t.create();
      created_tensors.push_back(&t);
    }
  }

  for (int i = 0; i < levels.size(); i++) {
    for (int j = 0; j < levels[i].size(); j++) {
      Operation &op = *levels[i][j];
      switch (op.optype) {
        case OpTypeAdd:
          op.add.execute();
          break;
        case OpTypeMult:
          op.mult.execute();
          break;
        default:
          printf("Unsupported operation type\n");
          assert(0);
      }
    }
  }

  for (size_t i = 0; i < created_tensors.size(); i++) {
    created_tensors[i]->destroy();
  }
}
}
