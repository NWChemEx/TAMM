#pragma once

#include "tamm/op_visitors.hpp"
#include "tamm/opmin.hpp"
#include "tamm/scheduler.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace tamm {
class OpCostCalculator {
public:
  // Ctors
  OpCostCalculator() = default;

  OpCostCalculator(SymbolTable& symbol_table): symbol_table_{symbol_table} {}

  // Copy/Move Ctors and Assignment Operators
  OpCostCalculator(OpCostCalculator&&)                 = default;
  OpCostCalculator(const OpCostCalculator&)            = default;
  OpCostCalculator& operator=(OpCostCalculator&&)      = default;
  OpCostCalculator& operator=(const OpCostCalculator&) = default;

  // Dtor
  ~OpCostCalculator() = default;

  template<typename T>
  void report_to_json(const Tensor<T>& tensor, std::string eq_type, std::string eqs_name,
                      json& output, bool use_opmin = true) {
    auto   updates      = tensor.get_updates();
    size_t update_count = 0;
    for(size_t update_idx = 0; update_idx < updates.size(); update_idx++) {
      auto& update            = updates[update_idx];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);
      bool  use_old_lhs       = (canonicalized_ops.size() == 1);
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        double                               op_cost     = 0.0;
        double                               mem_cost    = 0.0;
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        op_cost  = get_op_cost(op->clone(), tensor(lhs_labels), update.is_update_, use_opmin);
        mem_cost = get_op_mem_cost(op->clone(), tensor(lhs_labels), use_opmin);

        output[eq_type][eqs_name][std::to_string(update_idx)]["op_cost"] = op_cost;
        output[eq_type][eqs_name][std::to_string(update_idx)]["mem_cost"] =
          std::to_string(mem_cost / std::pow(2, 30)) + " GBs";
      }
    }
  }

  template<typename T>
  void print_tensor_execution_report(const Tensor<T>& tensor, bool use_opmin = true) {
    auto   updates      = tensor.get_updates();
    size_t update_count = 0;
    for(size_t update_idx = 0; update_idx < updates.size(); update_idx++) {
      auto& update            = updates[update_idx];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);
      bool  use_old_lhs       = (canonicalized_ops.size() == 1);
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        double                               op_cost     = 0.0;
        double                               mem_cost    = 0.0;
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        std::cout << "Update " << update_idx << " to Tensor "
                  << symbol_table_[tensor.get_symbol_ptr()] << "\n";
        print_op_binarized(new_ltop, op->clone(), use_opmin);

        op_cost  = get_op_cost(op->clone(), tensor(lhs_labels), update.is_update_, use_opmin);
        mem_cost = get_op_mem_cost(op->clone(), tensor(lhs_labels), use_opmin);

        std::cout << "Op cost : " << op_cost << "\n";
        std::cout << "Intermediate memory: " << mem_cost / std::pow(2, 30) << " GBs\n";
      }
    }
  }

  template<typename T>
  auto get_total_op_cost(const Tensor<T>& tensor, bool use_opmin = false) {
    auto   updates       = tensor.get_updates();
    size_t update_count  = 0;
    double total_op_cost = 0.0;
    double max_mem_cost  = 0.0;
    for(size_t i = 0; i < updates.size(); i++) {
      auto& update            = updates[i];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);
      bool  use_old_lhs       = (canonicalized_ops.size() == 1);
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        total_op_cost += get_op_cost(op->clone(), tensor(lhs_labels), update.is_update_, use_opmin);
        max_mem_cost =
          std::max(max_mem_cost, get_op_mem_cost(op->clone(), tensor(lhs_labels), use_opmin));
      }
    }
    return std::make_pair(total_op_cost, max_mem_cost);
  }

  template<typename T>
  double get_op_cost(std::unique_ptr<new_ops::Op> op, const LabeledTensor<T>& labeled_tensor,
                     bool is_update, bool use_opmin = false) {
    std::unique_ptr<new_ops::Op> optree = op->clone();
    optree->set_attribute<new_ops::NeededLabelsAttribute>(labeled_tensor.labels());
    if(use_opmin) {
      new_ops::LTOp new_ltop{labeled_tensor.tensor(), labeled_tensor.labels()};
      OpMin         opmin{symbol_table_};

      optree = opmin.optimize_all(new_ltop, *optree, is_update);
    }

    new_ops::OpCostVisitor cost_visitor;

    optree->accept(cost_visitor);

    double op_cost = optree->get_attribute<new_ops::OpCostAttribute>().get();

    return op_cost;
  }

  template<typename T>
  double get_op_mem_cost(std::unique_ptr<new_ops::Op> op, const LabeledTensor<T>& labeled_tensor,
                         bool use_opmin = false) {
    std::unique_ptr<new_ops::Op> optree = op->clone();
    optree->set_attribute<new_ops::NeededLabelsAttribute>(labeled_tensor.labels());
    if(use_opmin) {
      new_ops::LTOp new_ltop{labeled_tensor.tensor(), labeled_tensor.labels()};
      OpMin         opmin{symbol_table_};

      optree = opmin.optimize_all(new_ltop, *optree, true);
    }

    new_ops::OpMemCostVisitor cost_visitor;

    optree->accept(cost_visitor);

    double op_cost = optree->get_attribute<new_ops::OpMemCostAttribute>().get();

    return op_cost;
  }

  void print_op_binarized(const new_ops::LTOp& lhs_ltop, const std::unique_ptr<new_ops::Op>& in_op,
                          bool use_opmin = true) {
    auto canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*in_op);

    bool use_old_lhs = (canonicalized_ops.size() == 1);
    for(size_t i = 0; i < canonicalized_ops.size(); i++) {
      auto&                        op_lbl_pair = canonicalized_ops[i];
      std::unique_ptr<new_ops::Op> op          = std::move(op_lbl_pair.first);
      std::unique_ptr<new_ops::Op> optree      = op->clone();
      optree->set_attribute<new_ops::NeededLabelsAttribute>(lhs_ltop.labels());
      if(use_opmin) {
        new_ops::LTOp new_ltop{lhs_ltop.tensor(), lhs_ltop.labels()};
        OpMin         opmin{symbol_table_};

        optree = opmin.optimize_all(new_ltop, *optree, true);
      }
      auto                                 label_pair = op_lbl_pair.second;
      std::map<std::string, TensorVariant> inter_tensors;
      std::vector<new_ops::BinarizedOp>    binops;
      auto                                 lhs_labels = lhs_ltop.labels();

      if(!use_old_lhs) {
        auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
        EXPECTS(it != lhs_labels.end());
        auto index        = std::distance(lhs_labels.begin(), it);
        lhs_labels[index] = label_pair.second;
      }

      op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);

      binops = new_ops::BinarizeOpsVisitor::binarize_op(*optree, symbol_table_);

      // last op will be updated with updates LHS LT
      new_ops::BinarizedOp last_op = binops.back();
      binops.pop_back();
      new_ops::TensorInfo new_lhs{symbol_table_[lhs_ltop.tensor().get_symbol_ptr()],
                                  lhs_ltop.tensor(),
                                  lhs_ltop.labels(),
                                  lhs_ltop.tensor_type(),
                                  lhs_ltop.coeff(),
                                  false};
      last_op.lhs_       = new_lhs;
      last_op.is_assign_ = true;

      binops.push_back(last_op);
      for(const auto& binop: binops) { std::cout << binop.op_string(symbol_table_) << "\n"; }
    }
  }

protected:
  SymbolTable& symbol_table_;
}; // class OpCostCalculator

class OpExecutor {
public:
  using TensorStrMap = std::map<TensorBase*, std::string>;
  using LabelStrMap  = std::map<TileLabelElement, std::string>;
  // Ctors
  OpExecutor() = default;

  OpExecutor(Scheduler& sch, SymbolTable& symbol_table): sch_{sch}, symbol_table_{symbol_table} {}

  // Copy/Move Ctors and Assignment Operators
  OpExecutor(OpExecutor&&)                 = default;
  OpExecutor(const OpExecutor&)            = default;
  OpExecutor& operator=(OpExecutor&&)      = default;
  OpExecutor& operator=(const OpExecutor&) = default;

  // Dtor
  ~OpExecutor() = default;

  template<typename T>
  void execute(Tensor<T> tensor, bool use_opmin = false, ExecutionHW execute_on = ExecutionHW::CPU,
               bool profile = false) {
    auto   updates      = tensor.get_updates();
    size_t update_count = 0;
    for(size_t i = tensor.version(); i < updates.size(); i++) {
      auto& update            = updates[i];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);

      bool use_old_lhs = (canonicalized_ops.size() == 1);
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        if(use_opmin) {
          OpMin opmin{symbol_table_};
          auto  optimized_op = opmin.optimize_all(new_ltop, *op, !update.is_update_);
          binops = new_ops::BinarizeOpsVisitor::binarize_op(*optimized_op, symbol_table_);
        }
        else { binops = new_ops::BinarizeOpsVisitor::binarize_op(*op, symbol_table_); }

        // last op will be updated with updates LHS LT
        new_ops::BinarizedOp last_op = binops.back();
        binops.pop_back();
        new_ops::TensorInfo new_lhs{symbol_table_[tensor.get_symbol_ptr()],
                                    new_ltop.tensor(),
                                    new_ltop.labels(),
                                    new_ltop.tensor_type(),
                                    new_ltop.coeff(),
                                    false};
        last_op.lhs_       = new_lhs;
        last_op.is_assign_ = !update.is_update_;

        binops.push_back(last_op);

        // populate intermeadiate tensors for allocation
        construct_inter(binops, inter_tensors);

        for(auto& [key, tens]: inter_tensors) { tens.allocate(sch_); }

        for(auto& op: binops) { construct_op(op); }

        for(auto& [key, tens]: inter_tensors) { tens.deallocate(sch_); }
      }
      update_count++;
    }
    sch_.execute(execute_on, profile);
    tensor.update_version(update_count);
  }

  template<typename T>
  void opmin_execute(Tensor<T>& tensor) {
    execute(tensor, true);
  }

  void construct_op(const new_ops::BinarizedOp& binop) {
    new_ops::TensorInfo lhs  = binop.lhs_;
    new_ops::TensorInfo rhs1 = binop.rhs1_;
    new_ops::TensorInfo rhs2 = binop.rhs2_;

    // std::cout << "constructing binop: \n" << binop.op_string(symbol_table_) << "\n";

    std::visit(overloaded{[&](auto lhs_t, auto rhs1_t, auto rhs2_t, auto alpha, auto rhs1_alpha,
                              auto rhs2_alpha) {
                 using a_type  = decltype(alpha);
                 using a1_type = decltype(rhs1_alpha);
                 using a2_type = decltype(rhs2_alpha);

                 using lhs_type  = decltype(lhs_t);
                 using rhs1_type = decltype(rhs1_t);
                 using rhs2_type = decltype(rhs2_t);

                 if(binop.optype_ == new_ops::BinOpType::multop) {
                   if constexpr(std::is_same_v<lhs_type, Tensor<a_type>> &&
                                std::is_same_v<lhs_type, rhs1_type> &&
                                std::is_same_v<rhs1_type, rhs2_type>) {
                     auto lhs_lt  = lhs_t(lhs.ilv_);
                     auto rhs1_lt = rhs1_t(rhs1.ilv_);
                     auto rhs2_lt = rhs2_t(rhs2.ilv_);

                     auto multop = MultOp(lhs_lt, alpha, rhs1_lt, rhs2_lt, binop.is_assign_);
                     sch_(multop);
                   }
                   else { UNREACHABLE(); }
                 }
                 else if(binop.optype_ == new_ops::BinOpType::addop) {
                   if constexpr(std::is_same_v<a_type, a1_type> &&
                                std::is_same_v<a1_type, a2_type> &&
                                std::is_same_v<lhs_type, Tensor<a_type>> &&
                                std::is_same_v<rhs1_type, Tensor<a_type>> &&
                                std::is_same_v<rhs2_type, Tensor<a_type>>) {
                     auto lhs_lt  = lhs_t(lhs.ilv_);
                     auto rhs1_lt = rhs1_t(rhs1.ilv_);
                     auto rhs2_lt = rhs2_t(rhs2.ilv_);

                     // first op is same as binary op
                     auto addop1 = AddOp(lhs_lt, alpha, rhs1_lt, binop.is_assign_);
                     // second op is always accumulate
                     auto addop2 = AddOp(lhs_lt, alpha, rhs2_lt, false);
                     sch_(addop1);
                     sch_(addop2);
                   }
                   else { UNREACHABLE(); }
                 }
                 else if(binop.optype_ == new_ops::BinOpType::setop) {
                   if constexpr(std::is_same_v<a_type, a1_type> &&
                                std::is_same_v<lhs_type, Tensor<a_type>> &&
                                std::is_same_v<rhs1_type, Tensor<a_type>>) {
                     auto lhs_lt  = lhs_t(lhs.ilv_);
                     auto rhs1_lt = rhs1_t(rhs1.ilv_);

                     // first op is same as binary op
                     auto addop1 = AddOp(lhs_lt, rhs1_alpha, rhs1_lt, binop.is_assign_);
                     // second op is always accumulate
                     sch_(addop1);
                   }
                   else { UNREACHABLE(); }
                 }
                 else { UNREACHABLE(); }
               }},
               lhs.tensor_.value(), rhs1.tensor_.value(), rhs2.tensor_.value(),
               binop.scale_.value(), rhs1.scale_.value(), rhs2.scale_.value());
  }

  void construct_inter(const std::vector<new_ops::BinarizedOp>& binops,
                       std::map<std::string, TensorVariant>&    inter_tensors) {
    for(const new_ops::BinarizedOp& op: binops) {
      if(op.lhs_.is_intermediate_ && inter_tensors.find(op.lhs_.name_) == inter_tensors.end()) {
        inter_tensors[op.lhs_.name_] = op.lhs_.tensor_;
      }

      if(op.rhs1_.is_intermediate_ && inter_tensors.find(op.rhs1_.name_) == inter_tensors.end()) {
        inter_tensors[op.rhs1_.name_] = op.rhs1_.tensor_;
      }

      if(op.rhs2_.is_intermediate_ && inter_tensors.find(op.rhs2_.name_) == inter_tensors.end()) {
        inter_tensors[op.rhs2_.name_] = op.rhs2_.tensor_;
      }
    }
  }

  void print_op_binarized(const new_ops::LTOp& lhs_ltop, const std::unique_ptr<new_ops::Op>& in_op,
                          bool is_update = true, std::ofstream& os = std::cout) {
    auto canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*in_op);

    bool use_old_lhs = (canonicalized_ops.size() == 1);
    for(size_t i = 0; i < canonicalized_ops.size(); i++) {
      auto&                                op_lbl_pair = canonicalized_ops[i];
      std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
      auto                                 label_pair  = op_lbl_pair.second;
      std::map<std::string, TensorVariant> inter_tensors;
      std::vector<new_ops::BinarizedOp>    binops;
      auto                                 lhs_labels = lhs_ltop.labels();

      if(!use_old_lhs) {
        auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
        EXPECTS(it != lhs_labels.end());
        auto index        = std::distance(lhs_labels.begin(), it);
        lhs_labels[index] = label_pair.second;
      }

      op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);

      binops = new_ops::BinarizeOpsVisitor::binarize_op(*op, symbol_table_);

      // last op will be updated with updates LHS LT
      new_ops::BinarizedOp last_op = binops.back();
      binops.pop_back();
      new_ops::TensorInfo new_lhs{symbol_table_[lhs_ltop.tensor().get_symbol_ptr()],
                                  lhs_ltop.tensor(),
                                  lhs_ltop.labels(),
                                  lhs_ltop.tensor_type(),
                                  lhs_ltop.coeff(),
                                  false};
      last_op.lhs_       = new_lhs;
      last_op.is_assign_ = !is_update;

      binops.push_back(last_op);
      for(const auto& binop: binops) { std::cout << binop.op_string(symbol_table_) << "\n"; }
    }
  }

  template<typename T>
  void pretty_print_binarized(const Tensor<T>& tensor, bool use_opmin = false,
                              size_t start_update = 0) {
    auto updates = tensor.get_updates();
    for(size_t i = start_update; i < updates.size(); i++) {
      auto& update            = updates[i];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);

      bool use_old_lhs = (canonicalized_ops.size() == 1);
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        if(use_opmin) {
          OpMin opmin{symbol_table_};
          auto  optimized_op = opmin.optimize_all(new_ltop, *update.op_, !update.is_update_);
          binops = new_ops::BinarizeOpsVisitor::binarize_op(*optimized_op, symbol_table_);
        }
        else { binops = new_ops::BinarizeOpsVisitor::binarize_op(*op, symbol_table_); }

        // last op will be updated with updates LHS LT
        new_ops::BinarizedOp last_op = binops.back();
        binops.pop_back();
        new_ops::TensorInfo new_lhs{symbol_table_[tensor.get_symbol_ptr()],
                                    new_ltop.tensor(),
                                    new_ltop.labels(),
                                    new_ltop.tensor_type(),
                                    new_ltop.coeff(),
                                    false};
        last_op.lhs_       = new_lhs;
        last_op.is_assign_ = !update.is_update_;

        binops.push_back(last_op);
        for(const auto& binop: binops) { std::cout << binop.op_string(symbol_table_) << "\n"; }
      }
    }
  }

  template<typename T>
  void print_op_cost(const Tensor<T>& tensor, bool use_opmin = false) {
    auto   updates      = tensor.get_updates();
    size_t update_count = 0;
    for(size_t i = 0; i < updates.size(); i++) {
      auto& update            = updates[i];
      auto  canonicalized_ops = new_ops::CanonicalizeVisitor::canonicalize_ops(*update.op_);

      std::cout << "update " << i << " for tensor " << symbol_table_[tensor.get_symbol_ptr()]
                << "\n";
      bool   use_old_lhs   = (canonicalized_ops.size() == 1);
      double total_op_cost = 0.0;
      for(size_t i = 0; i < canonicalized_ops.size(); i++) {
        auto&                                op_lbl_pair = canonicalized_ops[i];
        std::unique_ptr<new_ops::Op>         op          = std::move(op_lbl_pair.first);
        auto                                 label_pair  = op_lbl_pair.second;
        std::map<std::string, TensorVariant> inter_tensors;
        std::vector<new_ops::BinarizedOp>    binops;
        auto                                 lhs_labels = update.ilv_;

        if(!use_old_lhs) {
          auto it = std::find(lhs_labels.begin(), lhs_labels.end(), label_pair.first);
          EXPECTS(it != lhs_labels.end());
          auto index        = std::distance(lhs_labels.begin(), it);
          lhs_labels[index] = label_pair.second;
        }

        op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);
        new_ops::LTOp new_ltop{tensor, lhs_labels};

        total_op_cost +=
          get_op_cost(std::move(op), tensor(lhs_labels), update.is_update_, use_opmin);
      }
      std::cout << "Total cost " << total_op_cost << "\n";
    }
  }

  template<typename T>
  double get_op_cost(std::unique_ptr<new_ops::Op> op, const LabeledTensor<T>& labeled_tensor,
                     bool is_update, bool use_opmin = false) {
    std::unique_ptr<new_ops::Op> optree = op->clone();
    optree->set_attribute<new_ops::NeededLabelsAttribute>(labeled_tensor.labels());
    if(use_opmin) {
      new_ops::LTOp new_ltop{labeled_tensor.tensor(), labeled_tensor.labels()};
      OpMin         opmin{symbol_table_};

      optree = opmin.optimize_all(new_ltop, *optree, is_update);
    }

    new_ops::OpCostVisitor cost_visitor;

    optree->accept(cost_visitor);

    double op_cost = optree->get_attribute<new_ops::OpCostAttribute>().get();

    return op_cost;
  }

  template<typename T>
  size_t get_op_mem_cost(std::unique_ptr<new_ops::Op> op, const LabeledTensor<T>& labeled_tensor,
                         bool use_opmin = false) {
    std::unique_ptr<new_ops::Op> optree = op->clone();
    optree->set_attribute<new_ops::NeededLabelsAttribute>(labeled_tensor.labels());
    if(use_opmin) {
      new_ops::LTOp new_ltop{labeled_tensor.tensor(), labeled_tensor.labels()};
      OpMin         opmin{symbol_table_};

      optree = opmin.optimize_all(new_ltop, *optree, true);
    }

    new_ops::OpMemCostVisitor cost_visitor;

    optree->accept(cost_visitor);

    size_t op_cost = optree->get_attribute<new_ops::OpMemCostAttribute>().get();

    return op_cost;
  }

  Scheduler& scheduler() { return sch_; }

  SymbolTable& symbol_table() { return symbol_table_; }

  void set_symbol_table(const SymbolTable& symbol_table) { symbol_table_ = symbol_table; }

protected:
  Scheduler&   sch_;
  SymbolTable& symbol_table_;
}; // class OpExecutor
} // namespace tamm
