#pragma once

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "tamm/op_dag.hpp"

namespace tamm {

#if 0
  namespace opmin::util {
  template <typename Itr>
  std::ostream& join_dump(std::ostream& os, Itr begin, Itr end,
                          const std::string& wo, const std::string& wc,
                          const std::string& sep) {
    auto dist = std::distance(begin, end);
    if (dist == 0) return os;
    if (dist == 1) return os << wo << *begin << wc;

    os << wo << *begin;
    for (auto itr = ++begin; itr != end; ++itr) {
      os << sep << *itr;
    }
    return os << wc;
  }

  template <typename T>
  std::ostream& vector_dump(std::ostream& os, const std::vector<T>& vec,
                            const std::string& wo = "(",
                            const std::string& wc = ")",
                            const std::string& sep = ", ") {
    return join_dump(os, vec.begin(), vec.end(), wo, wc, sep);
  }
  } //namespace util
#endif
///////////////////////////////////////////////////////////////////

namespace opmin::internal {

using U64 = uint64_t;

// // print bit indices of x
// void setPrint(U64 x) {
//   std::cout << "{ ";
//   for (auto i = 1; x; x >>= 1, ++i)
//     if (x & 1) std::cout << i << ", ";
//   std::cout << "}\n";
// }

// get next greater subset of set with
// Same Number Of One Bits
inline U64 snoob(U64 sub, U64 set) {
  U64 tmp = sub - 1;
  U64 rip = set & (tmp + (sub & (0 - sub)) - set);
  for(sub = (tmp & sub) ^ rip; sub &= sub - 1; rip ^= tmp, set ^= tmp) tmp = set & (0 - set);
  return rip;
}

template<typename Fn, typename... Args>
void enumerate_power_set_minus_empty_set(unsigned n, Fn&& fn, Args&&... args) {
  auto set = (1ULL << n) - 1; // n bits
  for(unsigned i = 1; i < n + 1; ++i) {
    auto sub = (1ULL << i) - 1; // i bits
    U64  x   = sub;
    U64  y;
    do {
      fn(x, std::forward<Args>(args)...);
      y = snoob(x, set); // get next subset
      x = y;
    } while((y > sub));
  }
}

constexpr std::uint32_t intlog2(std::uint32_t n) { return (n > 1) ? 1 + intlog2(n >> 1) : 0; }

inline unsigned int get_first_bit_pos(uint64_t n) { return intlog2(n & -n) + 1; }
} // namespace opmin::internal

#if 0
  namespace opmin {

  class children_t;

  ////////////////////////////////////////////////////////////////////
  //
  //  Placeholder OpExpr (redo using tamm::OpTree)
  //
  ///////////////////////////////////////////////////////////////////

  using Range = double;
  class OpStmt;

  class Index {
  public:
    Index(const std::string& name = "", const std::vector<Index>& indices = {})
        : name_{name}, indices_{indices} {}

    const std::vector<Index>& dep_indices() const { return indices_; }

    const std::string& name() const { return name_; }

  private:
    std::string name_;
    std::vector<Index> indices_;
    friend std::ostream& operator<<(std::ostream& os, const Index& idx) {
      os << idx.name_;
      return util::vector_dump(os, idx.indices_);
    }

    friend bool operator<(const Index& lhs, const Index& rhs) {
      std::stringstream ss1, ss2;
      ss1 << lhs;
      ss2 << rhs;
      return ss1.str() < ss2.str();
    }
  };

  class Optimizer;

  class OpExpr {
  public:
    virtual ~OpExpr() {}
    virtual std::ostream& print(std::ostream&) const = 0;

    virtual bool has_intermediate_name() const = 0;

    /**
    * @brief Set the intermediate name object, if it does not have one
    * 
    * @param name name to be set to
    * @return true if the name was set
    * @return false if the expression already has an unchangeable name
    */
    virtual bool set_intermediate_name(const std::string& name) = 0;

    virtual const std::vector<Index>& indices() const = 0;

    virtual bool set_indices(const std::vector<Index>& indices) = 0; 

    /**
    * @brief Checks if this is a tensor or not
    * 
    * @return true If this is a tensor
    * @return false otherwise
    */
    virtual bool is_tensor() const = 0;

    virtual void set_intermediate_name_visitor(const std::string& base_name,
                                              int& counter) = 0;

    virtual void set_indices_visitor(
        const Optimizer& opt, const std::vector<uint64_t>& required_index_ids,
        const std::vector<children_t>& children, int pos) = 0;

    virtual void binarize_visitor(const OpStmt& stmt,
                                  std::vector<OpStmt>& out_stmts,
                                  bool is_root = true) = 0;

      virtual const std::string& name() const = 0;

  private:
    friend std::ostream& operator<<(std::ostream& os, const OpExpr& oe) {
      return oe.print(os);
    }
  };

  class Tensor : public OpExpr {
  public:
    Tensor(const std::string& name, const std::vector<Index>& indices = {})
        : name_{name}, indices_{indices} {}

    const std::string& name() const override { return name_; }

    bool has_intermediate_name() const override { return true; }

    bool set_intermediate_name(const std::string& name) override { return false; }


    const std::vector<Index>& indices() const override { return indices_; }

    bool set_indices(const std::vector<Index>& indices) override {
        return false;
    }

    void set_indices_visitor(const Optimizer& opt,
                            const std::vector<uint64_t>& required_index_ids,
                            const std::vector<children_t>& children,
                            int pos) override {}

    bool is_tensor() const { return true; }

    void set_intermediate_name_visitor(const std::string& base_name,
                                              int& counter) override { }

    void binarize_visitor(const OpStmt& stmt, std::vector<OpStmt>& out_stmts,
                          bool is_root) override {
      if (is_root) {
        out_stmts.push_back(stmt);  // just identity
      }
    }

  private:
    std::ostream& print(std::ostream& os) const override { return os << *this; }

    std::string name_;
    std::vector<Index> indices_;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
      os << t.name_;
      return util::vector_dump(os, t.indices_);
    }
  };

  class BinOpExpr : public OpExpr {
  public:
    BinOpExpr(const std::shared_ptr<OpExpr>& child1,
              const std::shared_ptr<OpExpr>& child2,
              const std::string name = "")
        : child1_{child1}, child2_{child2}, name_{name} {}

  bool has_intermediate_name() const override {
        return name_.size() != 0;
    }

  const std::string& name() const override {
      return name_;
  }

  bool set_intermediate_name(const std::string& name) override {
        name_ = name;
        return true;
    }

    void set_intermediate_name_visitor(const std::string& base_name,
                                      int& counter) override {
      child1_->set_intermediate_name_visitor(base_name, counter);
      child2_->set_intermediate_name_visitor(base_name, counter);
      set_intermediate_name(base_name + "_" + std::to_string(counter++));
    }

    void set_indices_visitor(const Optimizer& opt,
                            const std::vector<uint64_t>& required_index_ids,
                            const std::vector<children_t>& children,
                            int pos) override;

    const std::vector<Index>& indices() const override {
        return indices_;
    }

    bool set_indices(const std::vector<Index>& indices) {
        index_is_set_ = true;
        indices_ = indices;
        return true;
    }

    /**
    * @brief Checks if this is a tensor or not
    *
    * @return true If this is a tensor
    * @return false otherwise
    */
    bool is_tensor() const override { return false; }

    void binarize_visitor(const OpStmt& stmt, std::vector<OpStmt>& out_stmts,
                          bool is_root) override;

  private:
    std::ostream& print(std::ostream& os) const override { return os << *this; }
    std::shared_ptr<OpExpr> child1_;
    std::shared_ptr<OpExpr> child2_;
    std::string name_;
    bool index_is_set_ = false;
    std::vector<Index> indices_;

    friend std::ostream& operator<<(std::ostream& os, const BinOpExpr& boe) {
      return os << "(" << *boe.child1_ << " * " << *boe.child2_ << ")";
    }
  };

  enum class OpType { equal, plus_equal, minus_equal };

  std::string opstring(OpType opt) {
    switch (opt) {
      case OpType::equal:
        return "=";
        break;
      case OpType::plus_equal:
        return "+=";
        break;
      case OpType::minus_equal:
        return "-=";
        break;
    }
    return "";
  }

  class OpStmt {
  public:
    OpStmt(const Tensor& lhs, OpType opt, const std::string& scalar,
          const std::vector<Tensor>& rhs, const std::string& name = "")
        : lhs_{lhs}, opt_{opt}, scalar_{scalar}, rhs_{rhs}, name_{name} {
      std::set<Index> index_set{lhs.indices().begin(), lhs.indices().end()};
      for (const auto& ten : rhs_) {
        index_set.insert(ten.indices().begin(), ten.indices().end());
      }
      uint64_t id = 1;
      // for (const auto& idx : index_set) {
      //   id_to_index_[id] = idx;
      //   index_to_id_[idx] = id;
      //   id <<= 1;
      // }
    }

    const Tensor& lhs() const {
        return lhs_;
    }

    const std::vector<Tensor>& rhs() const {
        return rhs_;
    }

    const std::string& scalar() const {
        return scalar_;
    }

    OpType optype() const {
        return opt_;
    }

    const std::string& name() const {
        return name_;
    }

    // const std::map<uint64_t, Index>& id_to_index() const { return id_to_index_; }

    // const std::map<Index, uint64_t>& index_to_id() const { return index_to_id_; }

    // std::vector<Index> translate_index_ids(uint64_t ids) const {
    //   std::vector<Index> indices;
    //   while (ids != 0) {
    //   //   std::cerr << "iterating through bits ids=" << ids << "\n";
    //     int pos = internal::get_first_bit_pos(ids);
    //   //   std::cerr<<"pos="<<pos<<"\n";
    //     indices.push_back(id_to_index_.find(1<<(pos-1))->second);
    //     ids ^= (1<<(pos-1));
    //   }
    //   // std::cerr<<"exiting\n";
    //   return indices;
    // }

  private:
    Tensor lhs_;
    OpType opt_;
    std::string scalar_;
    std::vector<Tensor> rhs_;
    std::string name_;
    // std::map<uint64_t, Index> id_to_index_;
    // std::map<Index, uint64_t> index_to_id_;

    friend std::ostream& operator << (std::ostream& os, const OpStmt& stmt) {
      os << stmt.name_ << ": " << stmt.lhs_ << " " << opstring(stmt.opt_) << " "
        << stmt.scalar_  << (stmt.scalar_ == "" ? "" : " * ");
      util::join_dump(os, stmt.rhs_.begin(), stmt.rhs_.end(), "", "", " * ");
      os << ";";
      return os;
    }
  };

  struct Input {
    using RangeName = std::string;
    using IndexName = std::string;
    // std::map<RangeName, int64_t> range_values_;
    // std::map<IndexName, RangeName> index_table_;
    std::map<IndexName, int64_t> index_values_;
    std::vector<OpStmt> stmts_;
    std::vector<std::shared_ptr<OpExpr>> optmin_opexpr_;
  };

  struct children_t {
    int64_t child1;  //<0 indicates leaf
    int64_t child2;  //<0 indicates leaf
  };

  }  // namespace opmin
#endif

////////////////////////////////////////////////////////////////////
//
//                   opmin optimizer
//
///////////////////////////////////////////////////////////////////

namespace opmin {

struct children_t {
  int64_t child1; //<0 indicates leaf
  int64_t child2; //<0 indicates leaf
};
enum class OpType { equal, plus_equal, minus_equal };

class OpStmt {
public:
  OpStmt(const new_ops::TensorInfo& lhs, OpType opt, const Scalar& scalar,
         const std::vector<new_ops::TensorInfo>& rhs):
    lhs_{lhs}, opt_{opt}, scalar_{scalar}, rhs_{rhs} {}

  const new_ops::TensorInfo& lhs() const { return lhs_; }

  const std::vector<new_ops::TensorInfo>& rhs() const { return rhs_; }

  const Scalar& scalar() const { return scalar_; }

  OpType optype() const { return opt_; }

private:
  new_ops::TensorInfo              lhs_;
  OpType                           opt_;
  Scalar                           scalar_;
  std::vector<new_ops::TensorInfo> rhs_;
};

class Optimizer {
public:
  Optimizer(OpStmt& stmt, const std::map<TiledIndexLabel, std::string>& index_names):
    stmt_{stmt}, nterms_{stmt.rhs().size()}, index_names_{index_names} {
    computed_list_.reserve(1l << nterms_);
    children_.resize(1l << nterms_);
    // compute_cost_.resize(1l << nterms_, INT64_MAX);
    compute_cost_.resize(1l << nterms_, std::numeric_limits<double>::max());
    available_index_ids_.resize(1l << nterms_);
    required_index_ids_.resize(1l << nterms_);

    order_indices();
    lhs_index_ids_ = 0;
    for(const auto& idx: stmt.lhs().ilv_) {
      // lhs_index_ids_ |= stmt.index_to_id().find(idx)->second;
      lhs_index_ids_ |= index_to_id_.find(index_names.at(idx))->second;
    }
  }

  std::unique_ptr<new_ops::Op> optimize() {
    const unsigned n = nterms_;
    for(uint64_t i = 0; i < n; ++i) {
      // std::cerr << "Updating pos=" << (1 << i) << "\n";
      children_[1 << i]            = {-1, -1};
      compute_cost_[1 << i]        = 0;
      available_index_ids_[1 << i] = 0;
      // std::cerr << "num indices in rhs[" << i
      //           << "]=" << stmt_.rhs()[i].indices().size() << "\n";
      for(const auto& idx: stmt_.rhs()[i].ilv_) {
        // std::cerr << "looking up index=" << idx << "\n";
        // std::cerr << index_to_id_.size() << "\n";
        assert(index_to_id_.find(index_names_.at(idx)) != index_to_id_.end());
        available_index_ids_[1 << i] |= index_to_id_.find(index_names_.at(idx))->second;
      }
    }
    computed_list_.clear();
    internal::enumerate_power_set_minus_empty_set(n, compute_available_index_ids, *this);
    internal::enumerate_power_set_minus_empty_set(n, compute_required_ids, *this);
    computed_list_.clear();
    internal::enumerate_power_set_minus_empty_set(n, optimize_optree, *this);

    auto ope = construct_op_expr(stmt_.rhs(), children_);
    ope->set_coeff(stmt_.scalar());
    // std::cerr << opes.back() << "\n";
    // ope->set_indices_visitor(*this, required_index_ids_, children_,
    //                          (1l << nterms_) - 1);
    return ope;
  }

  double compute_id_product_size(unsigned int ids) const {
    double product_size = 1;
    while(ids != 0) {
      int pos = internal::get_first_bit_pos(ids);
      assert(pos - 1 < ordered_index_sizes_.size());
      product_size *= ordered_index_sizes_[pos - 1];
      ids ^= (1 << (pos - 1));
    }
    return product_size;
  }

  double id_size(unsigned int id) const {
    unsigned pos = internal::intlog2(id);
    assert(pos < ordered_index_sizes_.size());
    return ordered_index_sizes_[pos];
  }

  std::vector<TiledIndexLabel> translate_index_ids(uint64_t ids) const {
    std::vector<TiledIndexLabel> indices;
    while(ids != 0) {
      int pos = internal::get_first_bit_pos(ids);
      indices.push_back(id_to_index_obj_.find(1 << (pos - 1))->second);
      ids ^= (1 << (pos - 1));
    }
    return indices;
  }

private:
  static void compute_available_index_ids(unsigned x, Optimizer& opt) {
    for(auto y: opt.computed_list_) {
      opt.available_index_ids_[x | y] = opt.available_index_ids_[x] | opt.available_index_ids_[y];
    }
    opt.computed_list_.push_back(x);
  }

  static void compute_required_ids(unsigned x, Optimizer& opt) {
    opt.required_index_ids_[x] =
      opt.available_index_ids_[x] &
      (opt.available_index_ids_[((1l << opt.nterms_) - 1) & (~x)] | opt.lhs_index_ids_);
  }

  static void optimize_optree(unsigned x, Optimizer& opt) {
    for(auto y: opt.computed_list_) {
      if((x & y) == 0) {
        auto   product_index_ids = opt.required_index_ids_[x] | opt.required_index_ids_[y];
        double this_compute_cost = opt.compute_id_product_size(product_index_ids);
        if(this_compute_cost + opt.compute_cost_[x] + opt.compute_cost_[y] <
           opt.compute_cost_[x | y]) {
          // std::cout << "Op cost for " << std::bitset<16>{x | y} << " : "
          //           << (double)this_compute_cost << "\n";
          opt.compute_cost_[x | y] =
            this_compute_cost + opt.compute_cost_[x] + opt.compute_cost_[y];
          opt.children_[x | y].child1 = x;
          opt.children_[x | y].child2 = y;
        }
      }
    }
    opt.computed_list_.push_back(x);
  }

  std::unique_ptr<new_ops::Op> construct_op_expr(const std::vector<new_ops::TensorInfo>& tensors,
                                                 const std::vector<children_t>&          children,
                                                 int64_t                                 pos = -1) {
    if(pos == -1) {
      pos = children.size() - 1; // initial condition. last is the full tree
    }
    bool leaf   = (children[pos].child1 == -1 && children[pos].child2 == -1);
    bool binary = (children[pos].child1 != -1 && children[pos].child2 != -1);
    // check that this is full binary node
    assert(leaf || binary);

    if(leaf) {
      return new_ops::LTOp{tensors[internal::intlog2(pos)].tensor_,
                           tensors[internal::intlog2(pos)].ilv_}
        .clone();
    }
    else {
      auto c1 = construct_op_expr(tensors, children, children[pos].child1);
      auto c2 = construct_op_expr(tensors, children, children[pos].child2);
      return new_ops::MultOp{c1, c2}.clone();
    }
  }

  void order_indices() {
    std::set<TiledIndexLabel> index_set{stmt_.lhs().ilv_.begin(), stmt_.lhs().ilv_.end()};
    for(const auto& ten: stmt_.rhs()) { index_set.insert(ten.ilv_.begin(), ten.ilv_.end()); }

    uint64_t id = 1;
    for(const auto& idx: index_set) {
      assert(index_names_.find(idx) != index_names_.end());
      id_to_index_[id]                   = index_names_.at(idx);
      index_to_id_[index_names_.at(idx)] = id;
      id_to_index_obj_[id]               = idx;
      ordered_index_sizes_.push_back(idx.tiled_index_space().max_num_indices());
      id <<= 1;
    }
  }

  OpStmt                         stmt_;
  std::vector<double>            ordered_index_sizes_;
  std::map<std::string, int>     index_to_id_;
  std::map<int, TiledIndexLabel> id_to_index_obj_; // for translation
  std::map<int, std::string>     id_to_index_;
  size_t                         nterms_; // number of rhs labeled tensors
  uint64_t                       lhs_index_ids_;
  std::vector<children_t>        children_;
  std::vector<unsigned int>      computed_list_;
  std::vector<double>            compute_cost_;
  std::vector<uint64_t>          available_index_ids_;
  std::vector<uint64_t>          required_index_ids_;

  std::map<TiledIndexLabel, std::string> index_names_;

}; // class Optimizer
} // namespace opmin

class OpMin {
public:
  // Ctors
  OpMin() = default;
  OpMin(const SymbolTable& symbol_table): symbol_table_{symbol_table} {}

  // Copy/Move Ctors and Assignment Operators
  OpMin(OpMin&&)                 = default;
  OpMin(const OpMin&)            = default;
  OpMin& operator=(OpMin&&)      = default;
  OpMin& operator=(const OpMin&) = default;

  // Dtor
  ~OpMin() = default;

  std::unique_ptr<new_ops::Op> optimize_all(const new_ops::LTOp& lhs_op, new_ops::Op& rhs_op,
                                            bool is_assign = false) {
    new_ops::TensorInfo             lhs{symbol_table_[lhs_op.tensor().get_symbol_ptr()],
                            lhs_op.tensor(),
                            lhs_op.labels(),
                            lhs_op.tensor_type(),
                            lhs_op.coeff(),
                            false};
    new_ops::AvailableLabelsVisitor available_labels;
    new_ops::UsedTensorInfoVisitor  tensor_info{symbol_table_};
    new_ops::SeparateSumOpsVisitor  sum_visitor;

    rhs_op.accept(available_labels);
    rhs_op.accept(tensor_info);

    auto all_labels = rhs_op.get_attribute<new_ops::AvailableLabelsAttribute>().get();
    auto lhs_labels = lhs_op.labels();
    all_labels.insert(lhs_labels.begin(), lhs_labels.end());

    std::map<TiledIndexLabel, std::string> label_names;
    for(const auto& lbl: all_labels) {
      if(symbol_table_.find(lbl.get_symbol_ptr()) == symbol_table_.end()) {
        label_names[lbl] = lbl.label_str();
      }
      else { label_names[lbl] = symbol_table_[lbl.get_symbol_ptr()]; }
    }

    auto                       sum_ops = sum_visitor.sum_vectors(rhs_op);
    new_ops::OpStringGenerator str_generator{symbol_table_};

    opmin::OpType optype = opmin::OpType::plus_equal;
    if(is_assign == true) { optype = opmin::OpType::equal; }
    std::vector<std::unique_ptr<new_ops::Op>> optimized_ops;

    for(auto& op: sum_ops) {
      auto tensors = op->get_attribute<new_ops::UsedTensorInfoAttribute>().get();
      if(tensors.size() > 2) {
        opmin::OpStmt    stmt{lhs, optype, op->coeff(), tensors};
        opmin::Optimizer optimizer{stmt, label_names};
        auto             optimized_op = optimizer.optimize();
        optimized_ops.push_back(std::move(optimized_op));
      }
      else { optimized_ops.push_back(std::move(op->clone())); }
    }

    std::unique_ptr<new_ops::Op> result_op = (*optimized_ops.at(0)).clone();
    for(size_t i = 1; i < optimized_ops.size(); i++) {
      result_op = new_ops::AddOp{result_op, optimized_ops.at(i)}.clone();
    }

    new_ops::ClearAttributesVisitor clear_visitor;
    result_op->accept(clear_visitor);
    result_op->set_attribute<new_ops::NeededLabelsAttribute>(lhs_labels);

    return std::move(result_op);
  }

protected:
  SymbolTable symbol_table_;
}; // class OpMin

// ////////////////////////////////////////////////////////////////////
// //
// //  glue code between OpExpr and opmin optimizer
// //  (redo using tamm::OpTree)
// //
// ///////////////////////////////////////////////////////////////////

// namespace opmin {

// void BinOpExpr::set_indices_visitor(
//     const Optimizer& opt, const std::vector<uint64_t>& required_index_ids,
//     const std::vector<children_t>& children, int pos) {
//   assert(pos >= 0);
//   assert(children[pos].child1 >= 0);
//   assert(children[pos].child2 >= 0);
//   child1_->set_indices_visitor(opt, required_index_ids, children,
//                                children[pos].child1);
//   child2_->set_indices_visitor(opt, required_index_ids, children,
//                                children[pos].child2);
//   indices_ = opt.translate_index_ids(required_index_ids[pos]);
// }

// void BinOpExpr::binarize_visitor(const OpStmt& stmt,
//                                  std::vector<OpStmt>& out_stmts, bool is_root) {
//   child1_->binarize_visitor(stmt, out_stmts, false);
//   child2_->binarize_visitor(stmt, out_stmts, false);
//   if (is_root) {
//     OpStmt ostmt{stmt.lhs(),
//                  stmt.optype(),
//                  stmt.scalar(),
//                  {{Tensor{child1_->name(), child1_->indices()},
//                    Tensor{child2_->name(), child2_->indices()}}},
//                  name_};
//     out_stmts.push_back(ostmt);
//   } else {
//     OpStmt ostmt{Tensor{name_, indices_},
//                  OpType::equal,
//                  "",
//                  {Tensor{child1_->name(), child1_->indices()},
//                   Tensor{child2_->name(), child2_->indices()}},
//                  name_};
//     out_stmts.push_back(ostmt);
//   }
// }

// } //namespace opmin

// ////////////////////////////////////////////////////////////////////
// //
// //                   driver to call the optimizer
// //
// ///////////////////////////////////////////////////////////////////

// void optimize_expr(opmin::Input& input,
//                    const std::map<opmin::Index, double>& index_sizes,
//                    std::vector<std::shared_ptr<opmin::OpExpr>>& opes,
//                    std::vector<std::vector<opmin::OpStmt>>& out_stmts) {
//   // for each statement
//   opes.clear();
//   out_stmts.clear();
//   for (auto& stmt : input.stmts_) {
//     opes.push_back(opmin::Optimizer{stmt, index_sizes}.optimize());
//   }
//   int counter = 0;
//   for (unsigned si = 0; si < input.stmts_.size(); ++si) {
//     opes[si]->set_intermediate_name_visitor(input.stmts_[si].name(), counter);
//     std::vector<opmin::OpStmt> out_stmt;
//     opes[si]->binarize_visitor(input.stmts_[si], out_stmt);
//     out_stmts.push_back(out_stmt);
//   }
// }

} // namespace tamm

// //----------------test driver --------------------

// int main(int argc, char* argv[]) {
// //   assert(argc == 2);
// //   int n = atoi(argv[1]);
// //   assert(n < 30);  // arbitrary bound to avoid massive runs
//   // enumerate(n);
//   // generatePowerSet(n);
//   using namespace opmin;
//   Input input;
//   Index i0{"i0"}, i1{"i1"}, i2{"i2"}, i3{"i3"}, i4{"i4"};
//   input.index_values_["i0"] = 10;
//   input.index_values_["i1"] = 10;
//   input.index_values_["i2"] = 10;
//   input.index_values_["i3"] = 10;
//   input.index_values_["i4"] = 10;
// #if 0
//   input.stmts_.push_back(
//       opmin::OpStmt{Tensor{"T0"},
//                     OpType::plus_equal,
//                     "-2",
//                     {Tensor{"T1", {i0, i1}}, Tensor{"T2", {i1, i2}},
//                      Tensor{"T4", {i2, i3}}, Tensor{"T8", {i3, i4}}}});
// #else
//   opmin::OpStmt opstmt{
//       Tensor{"T0", {i4}},
//       OpType::plus_equal,
//       "-2",
//       {Tensor{"T1", {i0, i1, i2, i3}}, Tensor{"T2", {i0, i1, i2, i3}}, Tensor{"T1", {i0, i1, i2,
//       i3, i4}},
//        Tensor{"T1", {i0, i1, i2, i3}}, Tensor{"T2", {i0, i1, i2, i3}}, Tensor{"T1", {i0, i1, i2,
//        i3, i4}}, Tensor{"T2", {i0, i1, i2, i3}}, Tensor{"T4"}, Tensor{"T8"}},
//       "S1"};
//   input.stmts_.push_back(opstmt);

// #endif
//   // optimize_expr(input);

//   std::map<Index, double> index_sizes;
//   index_sizes[i0] = 10;
//   index_sizes[i1] = 10;
//   index_sizes[i2] = 10;
//   index_sizes[i3] = 10;
//   index_sizes[i4] = 10;

//   std::vector<std::vector<opmin::OpStmt>> out_stmts;
//   std::vector<std::shared_ptr<opmin::OpExpr>> opes;
//   optimize_expr(input, index_sizes, opes, out_stmts);
//   for(auto& osv : out_stmts) {
//     for(auto& os: osv){
//       std::cout<<os<<"\n";
//     }
//   }
//   //   auto poe = std::make_shared<BinOpExpr>(
//   //       std::make_shared<BinOpExpr>(
//   //           std::make_shared<Tensor>("t1",
//   //                                    std::vector<Index>{Index{"i"},
//   //                                    Index{"j"}}),
//   //           std::make_shared<Tensor>("t2")),
//   //       std::make_shared<Tensor>("t3"));
//   //   std::cout << *poe << "\n";
//   //     optimize_expr({Tensor{"t1", std::vector<Index>{Index{"i"},
//   //     Index{"j"}}},
//   //             Tensor{"t2"}, Tensor{"t3"}});
//   //     std::cout << subtrees[subtrees.size() / 2] << "\n";
//   return 0;
// }
