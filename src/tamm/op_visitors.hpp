#pragma once

#include "tamm/op_dag.hpp"
#include "tamm/tiled_index_space.hpp"
#include "tamm/interfaces.hpp"
#include "tamm/op_cost.hpp"
#include "tamm/types.hpp"

namespace tamm {
namespace new_ops {
//////////////////////////////////////////////////////////////////////////
//
//                         visitors
//
//////////////////////////////////////////////////////////////////////////
class ToStringVisitor : public VisitorBase {
public:
    ToStringVisitor()                       = default;
    ToStringVisitor(const ToStringVisitor&) = default;
    ToStringVisitor(const SymbolTable& symbol_table, bool use_ltop = false) :
      symbol_table_{symbol_table}, use_ltop_{use_ltop} {}

    void visit(MultOp& mop) override {
        using namespace std::string_literals;
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        const std::string& lhs_str =
          mop.lhs().get_attribute<ToStringAttribute>().get();
        const std::string& rhs_str =
          mop.rhs().get_attribute<ToStringAttribute>().get();
        std::string coeff_str = mop.coeff().to_string();
        if (coeff_str == "1" || coeff_str == "1.0" || coeff_str == "1.00") {
          coeff_str = "";
        } else {
          coeff_str += " * ";
        }
        mop.set_attribute<ToStringAttribute>("("s + coeff_str + lhs_str +
                                             " * " + rhs_str + ")");
    }

    void visit(AddOp& aop) override {
        using namespace std::string_literals;
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        const std::string& lhs_str =
          aop.lhs().get_attribute<ToStringAttribute>().get();
        const std::string& rhs_str =
          aop.rhs().get_attribute<ToStringAttribute>().get();

        aop.set_attribute<ToStringAttribute>(lhs_str + " + \n" + rhs_str);
    }

    void visit(LTOp &ltop) override {
      if (!ltop.has_attribute<ToStringAttribute>()) {
        using namespace std::string_literals;
        std::string result;
        std::string coeff_str = ltop.coeff().to_string();
        if (coeff_str != "1" && coeff_str != "1.0" && coeff_str != "1.00" ) {
          result += coeff_str + " * ";
        }
        
        if (use_ltop_) {
          result += "(tamm::new_ops::LTOp)";
        }

        if (symbol_table_.find(ltop.tensor().get_symbol_ptr()) ==
            symbol_table_.end()) {
          result += "LT(";
          // result += "LT<"s + eltype_to_string(ltop.tensor_type()) + ">(";
        } else {
          result +=symbol_table_[ltop.tensor().get_symbol_ptr()] + "(";
          // result += symbol_table_[ltop.tensor().get_symbol_ptr()] + "<"s +
          //          eltype_to_string(ltop.tensor_type()) + ">(";
        }

        for (const auto &lbl : ltop.labels()) {
          if (symbol_table_.find(lbl.get_symbol_ptr()) == symbol_table_.end()) {
            if (lbl.label_str() == "") {
              result += std::to_string(lbl.label()) + ",";
            } else {
              result += "\"" + lbl.label_str() + "\",";
            }
          } else {
            result += symbol_table_[lbl.get_symbol_ptr()] + ",";
          }
        }
        result.pop_back();
        result += ")"s;
        // ltop.set_attribute<ToStringAttribute>("LT<"s+eltype_to_string(ltop.tensor_type())+">");
        ltop.set_attribute<ToStringAttribute>(result);
      }
    }

    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp& parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    ToStringVisitor(ToStringVisitor&& other) noexcept : ToStringVisitor{} {
        swap(*this, other);
    }

    ToStringVisitor& operator=(ToStringVisitor other) noexcept {
        swap(*this, other);
        return *this;
    }

    friend void swap(ToStringVisitor& first, ToStringVisitor& second) noexcept {
      using std::swap;
      swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
      swap(first.symbol_table_, second.symbol_table_);
      // no-op
    }

private:
    SymbolTable symbol_table_;
    bool use_ltop_;
};

class AvailableLabelsVisitor : public VisitorBase {
public:
    AvailableLabelsVisitor()                              = default;
    AvailableLabelsVisitor(const AvailableLabelsVisitor&) = default;

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        apply_binop(mop, mop.lhs(), mop.rhs());
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        apply_binop(aop, aop.lhs(), aop.rhs());
    }

    void visit(LTOp& ltop) override {
        ltop.set_attribute<AvailableLabelsAttribute>(ltop.labels());
    }

    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override {
        auto labels = reshapeop.labels();
        std::set<TiledIndexLabel> ils{labels.begin(), labels.end()};
        reshapeop.set_attribute<AvailableLabelsAttribute>(ils);
    }

    void visit(LambdaOp& lambdaop) override {}

    void visit(ParForOp& parforop) override {}

    AvailableLabelsVisitor(AvailableLabelsVisitor&& other) :
      AvailableLabelsVisitor{} {
        swap(*this, other);
    }

    AvailableLabelsVisitor& operator=(AvailableLabelsVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(AvailableLabelsVisitor& first,
                     AvailableLabelsVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
        // no-op
    }

private:
    void apply_binop(Op& binop, Op& left_child, Op& right_child) {
        const std::set<TiledIndexLabel>& llabels =
          left_child.get_attribute<AvailableLabelsAttribute>().get();
        const std::set<TiledIndexLabel>& rlabels =
          right_child.get_attribute<AvailableLabelsAttribute>().get();
        std::set<TiledIndexLabel> result{llabels};
        result.insert(rlabels.begin(), rlabels.end());
        binop.set_attribute<AvailableLabelsAttribute>(result);
    }
};

class NeededLabelsVisitor : public VisitorBase {
public:
    NeededLabelsVisitor()                           = default;
    NeededLabelsVisitor(const NeededLabelsVisitor&) = default;

    void visit(MultOp& mop) override {
        init(mop);
        apply_binop(mop, mop.lhs(), mop.rhs(), true);
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
    }

    void visit(AddOp& aop) override {
        init(aop);
        apply_binop(aop, aop.lhs(), aop.rhs(), false);
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
    }

    void visit(LTOp& ltop) override { init(ltop); }
    
    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override { init(reshapeop); }

    void visit(LambdaOp& lambdaop) override {}

    void visit(ParForOp& parforop) override {}

    NeededLabelsVisitor(NeededLabelsVisitor&& other) : NeededLabelsVisitor{} {
        swap(*this, other);
    }

    NeededLabelsVisitor& operator=(NeededLabelsVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(NeededLabelsVisitor& first,
                     NeededLabelsVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
        // no-op
    }

private:
    void init(Op& op) {
        if(!op.has_attribute<AvailableLabelsAttribute>()) {
          AvailableLabelsVisitor al_visitor;
          op.accept(al_visitor);
        }

        const auto& alabels =
          op.get_attribute<AvailableLabelsAttribute>().get();
        if(!op.has_attribute<NeededLabelsAttribute>()) {
            op.set_attribute<NeededLabelsAttribute>(alabels);
        }
    }

    void apply_binop(Op& binop, Op& left_child, Op& right_child, bool is_mult_op = true) {
        const std::set<TiledIndexLabel>& nlabels =
          binop.get_attribute<NeededLabelsAttribute>().get();
        const std::set<TiledIndexLabel>& llabels =
          left_child.get_attribute<AvailableLabelsAttribute>().get();
        const std::set<TiledIndexLabel>& rlabels =
          right_child.get_attribute<AvailableLabelsAttribute>().get();
        std::set<TiledIndexLabel> result_lhs{nlabels};
        std::set<TiledIndexLabel> result_rhs{nlabels};

        if(is_mult_op) {
            result_rhs.insert(llabels.begin(), llabels.end());
            result_lhs.insert(rlabels.begin(), rlabels.end());
        }

        right_child.set_attribute<NeededLabelsAttribute>(result_rhs);
        left_child.set_attribute<NeededLabelsAttribute>(result_lhs);

        // binop.set_attribute<NeededLabelsAttribute>(result);
    }
};

class AllocLabelsVisitor : public VisitorBase {
public:
    AllocLabelsVisitor()                          = default;
    AllocLabelsVisitor(const AllocLabelsVisitor&) = default;

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        init(mop);
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        init(aop);
    }
    void visit(LTOp& ltop) override { init(ltop); }

    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override { init(reshapeop); }

    void visit(LambdaOp& lambdaop) override {}

    void visit(ParForOp& parforop) override {}

    AllocLabelsVisitor(AllocLabelsVisitor&& other) : AllocLabelsVisitor{} {
        swap(*this, other);
    }

    AllocLabelsVisitor& operator=(AllocLabelsVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(AllocLabelsVisitor& first,
                     AllocLabelsVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
        // no-op
    }

private:
    void init(Op& op) {

      if (!op.has_attribute<AvailableLabelsAttribute>() ||
          !op.has_attribute<NeededLabelsAttribute>()) {
        NeededLabelsVisitor nl_visitor;
        op.accept(nl_visitor);
      }

        const auto& alabels =
          op.get_attribute<AvailableLabelsAttribute>().get();
        const auto& nlabels = op.get_attribute<NeededLabelsAttribute>().get();
        std::set<TiledIndexLabel> result;

        std::set_intersection(alabels.begin(), alabels.end(), nlabels.begin(),
                              nlabels.end(),
                              std::inserter(result, result.begin()));
        op.set_attribute<AllocLabelsAttribute>(result);
    }
};

class NameVisitor : public VisitorBase {
public:
    NameVisitor()                   = default;
    NameVisitor(const NameVisitor&) = default;

    NameVisitor(const SymbolTable& symbol_table) :
      symbol_table_{symbol_table}{}

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        init(mop);
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        init(aop);
    }

    void visit(LTOp& ltop) override {
        if(symbol_table_.find(ltop.tensor().get_symbol_ptr()) != symbol_table_.end()) {
            ltop.set_attribute<NameAttribute>(symbol_table_[ltop.tensor().get_symbol_ptr()]);
        } else {
            init(ltop);
        }
    }

    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp& parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    NameVisitor(NameVisitor&& other) : NameVisitor{} { swap(*this, other); }

    NameVisitor& operator=(NameVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(NameVisitor& first, NameVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
        swap(first.symbol_table_, second.symbol_table_);
        // no-op
    }

private:
    unsigned id() {
        static unsigned id_ = 0;
        return id_++;
    }

    void init(Op& op) {
        using namespace std::string_literals;
        op.set_attribute<NameAttribute>("_T"s + std::to_string(id()));
    }
    SymbolTable symbol_table_;
};

class ElTypeComputeVisitor : public VisitorBase {
public:
    ElTypeComputeVisitor()                            = default;
    ElTypeComputeVisitor(const ElTypeComputeVisitor&) = default;

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        ElType lt     = mop.lhs().get_attribute<ElTypeAttribute>().get();
        ElType rt     = mop.rhs().get_attribute<ElTypeAttribute>().get();
        ElType result = lub(lt, rt);
        mop.set_attribute<ElTypeAttribute>(result);
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        ElType lt     = aop.lhs().get_attribute<ElTypeAttribute>().get();
        ElType rt     = aop.rhs().get_attribute<ElTypeAttribute>().get();
        ElType result = lub(lt, rt);
        aop.set_attribute<ElTypeAttribute>(result);
    }

    void visit(LTOp& ltop) override {
        ltop.set_attribute<ElTypeAttribute>(ltop.tensor_type());
    }

    void visit(EinSumOp& einsumop) override {}
    
    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp& parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    ElTypeComputeVisitor(ElTypeComputeVisitor&& other) :
      ElTypeComputeVisitor{} {
        swap(*this, other);
    }

    ElTypeComputeVisitor& operator=(ElTypeComputeVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(ElTypeComputeVisitor& first,
                     ElTypeComputeVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
    }
};

class BinarizedPrintVisitor : public VisitorBase {
public:
    BinarizedPrintVisitor()                             = default;
    BinarizedPrintVisitor(const BinarizedPrintVisitor&) = default;
    BinarizedPrintVisitor(const SymbolTable& symbol_table) :
      symbol_table_{symbol_table} {}

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        apply_binop(mop, mop.lhs(), mop.rhs(), "*");
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        apply_binop(aop, aop.lhs(), aop.rhs(), "+");
    }

    void visit(LTOp& ltop) override {
        init(ltop);
    }

    void visit(EinSumOp& einsumop) override {}

    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp &parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    BinarizedPrintVisitor(BinarizedPrintVisitor&& other) :
      BinarizedPrintVisitor{} {
        swap(*this, other);
    }

    BinarizedPrintVisitor& operator=(BinarizedPrintVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(BinarizedPrintVisitor& first,
                     BinarizedPrintVisitor& second) noexcept {
      using std::swap;
      swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
      swap(first.symbol_table_, second.symbol_table_);
    }

    static std::vector<std::string> binarized_op_str(Op& op, const SymbolTable& symbol_table) {
        NameVisitor nv{symbol_table};
        AvailableLabelsVisitor alv;
        NeededLabelsVisitor nlv;
        AllocLabelsVisitor allv;
        BinarizedPrintVisitor bpv{symbol_table};

        op.accept(nv);
        op.accept(alv);
        op.accept(nlv);
        op.accept(allv);
        op.accept(bpv);

        return op.get_attribute<BinarizedStringAttribute>().get();
    }

private:
    SymbolTable symbol_table_;

    void init(Op& op) {
        std::string op_str;
        std::string coeff_str = op.coeff().to_string();
        if(coeff_str != "1" && coeff_str != "1.0")
            op_str += coeff_str + " * ";    
        op_str += constructString(op);

        op.set_attribute<BinarizedStringAttribute>(std::vector<std::string>{op_str});
    }

    void apply_binop(Op& binop, const Op& left_child, const Op& right_child,
                         std::string op_str) {
        EXPECTS(binop.has_attribute<AvailableLabelsAttribute>());
        EXPECTS(binop.has_attribute<NeededLabelsAttribute>());
        EXPECTS(binop.has_attribute<NameAttribute>());
        EXPECTS(binop.has_attribute<AllocLabelsAttribute>());

        EXPECTS(left_child.has_attribute<BinarizedStringAttribute>());
        EXPECTS(right_child.has_attribute<BinarizedStringAttribute>());

        const auto& lhs_str_vec = left_child.get_attribute<BinarizedStringAttribute>().get();
        const auto& rhs_str_vec = right_child.get_attribute<BinarizedStringAttribute>().get();
        std::vector<std::string> bin_str_vec;

        std::string bin_str = constructString(binop);

        std::string coeff_str = binop.coeff().to_string() == "1" ? "" : binop.coeff().to_string() + " * ";

        size_t pos = lhs_str_vec.back().find("=");
        if(pos != std::string::npos) {
            bin_str_vec.insert(bin_str_vec.end(), lhs_str_vec.begin(), lhs_str_vec.end());
            bin_str += " = " + coeff_str + lhs_str_vec.back().substr(0, pos-1);
        } else
            bin_str += " = " + coeff_str + lhs_str_vec.back();

        bin_str += " " + op_str + " ";
        pos = rhs_str_vec.back().find("=");
        if(pos != std::string::npos) {
            bin_str_vec.insert(bin_str_vec.end(), rhs_str_vec.begin(), rhs_str_vec.end());
            bin_str += rhs_str_vec.back().substr(0, pos-1);
        } else
            bin_str += rhs_str_vec.back();
        bin_str += ";";

        bin_str_vec.push_back(bin_str);

        binop.set_attribute<BinarizedStringAttribute>(bin_str_vec);
    }

    void apply_assign_op(Op& assignop, const Op& left_child, const Op& right_child, const std::string& op_str) {
        EXPECTS(assignop.has_attribute<AvailableLabelsAttribute>());
        EXPECTS(assignop.has_attribute<NeededLabelsAttribute>());
        EXPECTS(assignop.has_attribute<NameAttribute>());
        EXPECTS(assignop.has_attribute<AllocLabelsAttribute>());

        EXPECTS(left_child.has_attribute<BinarizedStringAttribute>());
        EXPECTS(right_child.has_attribute<BinarizedStringAttribute>());

        std::string assign_str = left_child.get_attribute<BinarizedStringAttribute>().get()[0];
        const auto& rhs_str_vec = right_child.get_attribute<BinarizedStringAttribute>().get();

        std::vector<std::string> assign_str_vec;

        assign_str += " " + op_str + " ";

        size_t pos = rhs_str_vec.back().find("=");
        if(pos != std::string::npos) {
            assign_str_vec.insert(assign_str_vec.end(), rhs_str_vec.begin(), rhs_str_vec.end()-1);
            assign_str += rhs_str_vec.back().substr(pos+2);
        } else
            assign_str += rhs_str_vec.back();

        assign_str_vec.push_back(assign_str);

        assignop.set_attribute<BinarizedStringAttribute>(assign_str_vec);
    }

    std::string constructString(const Op& op) {
        EXPECTS(op.has_attribute<AvailableLabelsAttribute>());
        EXPECTS(op.has_attribute<NeededLabelsAttribute>());
        EXPECTS(op.has_attribute<NameAttribute>());
        EXPECTS(op.has_attribute<AllocLabelsAttribute>());

        const std::string& name_str =
          op.get_attribute<NameAttribute>().get();
        const std::set<TiledIndexLabel>& alloc_lbls =
          op.get_attribute<AllocLabelsAttribute>().get();

        std::string result = "";

        result += name_str;
        if(!alloc_lbls.empty()) {
            result += "(";

            for(const auto& lbl : alloc_lbls) {
                EXPECTS(symbol_table_.find(lbl.get_symbol_ptr()) != symbol_table_.end());
                result += symbol_table_[lbl.get_symbol_ptr()];
                if(!lbl.secondary_labels().empty()){
                    result += "(";
                    for(const auto& dep_lbl : lbl.secondary_labels()) {
                        EXPECTS(symbol_table_.find(dep_lbl.get_symbol_ptr()) != symbol_table_.end());
                        result += symbol_table_[dep_lbl.get_symbol_ptr()] + ", ";
                    }
                    result.pop_back();
                    result.pop_back();
                    result += ")";
                }
                result += ", ";
            }
            result.pop_back();
            result.pop_back();
            result += ")";
        }

        return result;
    }
};


class TensorInfoVisitor : public VisitorBase {
public:
    TensorInfoVisitor()                         = default;
    TensorInfoVisitor(const TensorInfoVisitor&) = default;

    TensorInfoVisitor(const SymbolTable& symbol_table) :
      symbol_table_{symbol_table} {}

    void visit(MultOp& mop) override {
        init(mop);
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
    }

    void visit(AddOp& aop) override {
        init(aop);
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
    }

    void visit(LTOp& ltop) override {
        init_lt(ltop);
    }

    void visit(EinSumOp& einsumop) override {}
    
    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp &parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    TensorInfoVisitor(TensorInfoVisitor&& other) :
      TensorInfoVisitor{} {
        swap(*this, other);
    }

    TensorInfoVisitor& operator=(TensorInfoVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(TensorInfoVisitor& first,
                     TensorInfoVisitor& second) noexcept {
        using std::swap;
        swap(static_cast<VisitorBase&>(first),
             static_cast<VisitorBase&>(second));
        swap(first.symbol_table_, second.symbol_table_);
    }

private:
    SymbolTable symbol_table_;

    void init(Op& op) {
        ElTypeComputeVisitor elv;
        NameVisitor nv{symbol_table_};
        AvailableLabelsVisitor alv;
        NeededLabelsVisitor nlv;
        AllocLabelsVisitor allv;

        op.accept(elv);
        op.accept(nv);
        op.accept(alv);
        op.accept(nlv);
        op.accept(allv);

        auto t_name = op.get_attribute<NameAttribute>().get();
        auto t_ils = op.get_attribute<AllocLabelsAttribute>().get();
        IndexLabelVec t_ilv(t_ils.begin(), t_ils.end());

        auto t_eltype = op.get_attribute<ElTypeAttribute>().get();
        auto t_scale = Scalar{1.0};

        bool is_intermediate = t_name.substr(0, 2) == "_T";
        TensorVariant tensor;

        if(is_intermediate){
            tensor = TensorVariant{t_eltype, t_ilv};
        }

        TensorInfo t_info{t_name, tensor, t_ilv, t_eltype, t_scale, is_intermediate};

        op.set_attribute<TensorInfoAttribute>(t_info); 
    }

    void init_lt(LTOp& ltop) {
        auto t_name   = symbol_table_[ltop.tensor().get_symbol_ptr()];
        auto t_ilv    = ltop.labels();
        auto t_eltype = ltop.tensor().to_eltype();
        auto t_scale  = ltop.coeff();

        TensorInfo t_info{t_name, ltop.tensor(), t_ilv, t_eltype, t_scale, false};

        ltop.set_attribute<TensorInfoAttribute>(t_info);
    }

};

class BinarizeOpsVisitor : public VisitorBase {
public:
    BinarizeOpsVisitor()                         = default;
    BinarizeOpsVisitor(const BinarizeOpsVisitor&) = default;
    BinarizeOpsVisitor(const SymbolTable& symbol_table) :
      symbol_table_{symbol_table} {}

    void visit(MultOp& mop) override {
        init(mop);
        level_++;
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        apply_binop(mop, mop.lhs(), mop.rhs(), BinOpType::multop);
    }

    void visit(AddOp& aop) override {
        init(aop);
        level_++;
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        apply_binop(aop, aop.lhs(), aop.rhs(), BinOpType::addop);
    }

    void visit(LTOp& ltop) override {
        init(ltop);
        if(level_ == 0) 
          apply_setop(ltop);
    }

    void visit(EinSumOp& einsumop) override {}
    
    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp &parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    BinarizeOpsVisitor(BinarizeOpsVisitor&& other) :
      BinarizeOpsVisitor{} {
        swap(*this, other);
    }

    BinarizeOpsVisitor& operator=(BinarizeOpsVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(BinarizeOpsVisitor& first,
                     BinarizeOpsVisitor& second) noexcept {
      using std::swap;
      swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
      swap(first.symbol_table_, second.symbol_table_);
      swap(first.binarized_ops_, second.binarized_ops_);
    }

    std::vector<BinarizedOp> binary_ops() const { return binarized_ops_; }
    
    static std::vector<BinarizedOp> binarize_op(Op& op, const SymbolTable& symbol_table) {
        BinarizeOpsVisitor bov{symbol_table};
        op.accept(bov);

        return bov.binary_ops();
    }

private:
    int level_ = 0;
    SymbolTable symbol_table_;
    std::vector<BinarizedOp> binarized_ops_;

    void init(Op& op) {
        if(!op.has_attribute<TensorInfoAttribute>()) {
            TensorInfoVisitor tiv{symbol_table_};
            op.accept(tiv);
        }
    } 

    void apply_binop(const Op& op, const Op& left_child, const Op& right_child, BinOpType optype) {
        auto new_lhs   = op.get_attribute<TensorInfoAttribute>().get();
        auto rhs1_info = left_child.get_attribute<TensorInfoAttribute>().get();
        auto rhs2_info = right_child.get_attribute<TensorInfoAttribute>().get();
        auto scale     = op.coeff();

        binarized_ops_.emplace_back(new_lhs, rhs1_info, rhs2_info, scale, optype);
    }

    void apply_setop(const Op& op) {
        auto rhs = op.get_attribute<TensorInfoAttribute>().get();
        binarized_ops_.emplace_back(TensorInfo{}, rhs, TensorInfo{}, op.coeff(), BinOpType::setop);
    }
};

// std::ostream& operator<<(std::ostream& os, const Op& op) {
//     std::unique_ptr<Op> pop{op.clone()};
//     ToStringVisitor tsv;
//     pop->accept(tsv);
//     std::string str = pop->get_attribute<ToStringAttribute>().get();
//     return os << str;
// }

struct OpStringGenerator {
    OpStringGenerator() = default;
    OpStringGenerator(const OpStringGenerator& osg) 
    : symbol_table_{osg.symbol_table_} {}
    OpStringGenerator operator = (OpStringGenerator osg) noexcept {
        swap(*this, osg);
    }
    OpStringGenerator(const SymbolTable& symbol_table) :
        symbol_table_{symbol_table} {}
    OpStringGenerator(OpStringGenerator&& osg) noexcept : OpStringGenerator{} {
      swap(*this, osg);
    }

    std::string toString(const Op& op) {
        if(op.has_attribute<ToStringAttribute>())
          return op.get_attribute<ToStringAttribute>().get();

        std::unique_ptr<Op> pop{op.clone()};
        ToStringVisitor tsv{symbol_table_};
        pop->accept(tsv);
        return pop->get_attribute<ToStringAttribute>().get();
    }
    friend void swap(OpStringGenerator& first, OpStringGenerator& second) noexcept {
        using std::swap;
        swap(first.symbol_table_, second.symbol_table_);
    }
    SymbolTable symbol_table_;
};

class ClearAttributesVisitor : public VisitorBase {
 public:
  ClearAttributesVisitor() = default;
  ClearAttributesVisitor(const ClearAttributesVisitor&) = default;

  ClearAttributesVisitor(ClearAttributesVisitor&& other) noexcept
      : ClearAttributesVisitor{} {
    swap(*this, other);
  }

  ClearAttributesVisitor& operator=(ClearAttributesVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(ClearAttributesVisitor& first,
                   ClearAttributesVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
  }

  void visit(MultOp& mop) override {
    using namespace std::string_literals;
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
    mop.clear_attributes();
  }

  void visit(AddOp& aop) override {
    using namespace std::string_literals;
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    aop.clear_attributes();
  }

  void visit(LTOp& ltop) override {
    ltop.clear_attributes();
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

 private:
};

class SummationLabelsVisitor : public VisitorBase {
 public:
  SummationLabelsVisitor() = default;
  SummationLabelsVisitor(const SummationLabelsVisitor&) = default;

  SummationLabelsVisitor(SummationLabelsVisitor&& other) noexcept
      : SummationLabelsVisitor{} {
    swap(*this, other);
  }

  SummationLabelsVisitor& operator=(SummationLabelsVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(SummationLabelsVisitor& first,
                   SummationLabelsVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
  }

  void visit(MultOp& mop) override {
    init(mop);
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
    apply(mop);
  }

  void visit(AddOp& aop) override {
    init(aop);
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    apply(aop);
  }

  void visit(LTOp& ltop) override { /*no-op*/ }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

 private:
  void init(Op& op) {
    if (!op.has_attribute<AvailableLabelsAttribute>() ||
        !op.has_attribute<NeededLabelsAttribute>()) {
      NeededLabelsVisitor nl_visitor;
      op.accept(nl_visitor);
    }
  }

  void apply(Op &op) {
    // available labels - required labels
    const auto &available_labels =
        op.get_attribute<AvailableLabelsAttribute>().get();
    const auto &needed_labels = op.get_attribute<NeededLabelsAttribute>().get();
    std::set<TiledIndexLabel> summation_labels;
    std::set_difference(
        available_labels.begin(), available_labels.end(), needed_labels.begin(),
        needed_labels.end(),
        std::inserter(summation_labels, summation_labels.begin()));
    op.set_attribute<SummationLabelsAttribute>(summation_labels);
  }
};

class SeparateSumOpsVisitor : public VisitorBase {
 public:
  SeparateSumOpsVisitor() = default;
  SeparateSumOpsVisitor(const SeparateSumOpsVisitor&) = default;

  SeparateSumOpsVisitor(SeparateSumOpsVisitor&& other) noexcept
      : SeparateSumOpsVisitor{} {
    swap(*this, other);
  }

  SeparateSumOpsVisitor& operator=(SeparateSumOpsVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(SeparateSumOpsVisitor& first,
                   SeparateSumOpsVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
    swap(first.sum_vectors_, second.sum_vectors_);
  }

  void visit(MultOp& mop) override {
    sum_vectors_.push_back(mop.clone());
  }

  void visit(AddOp& aop) override {
    using namespace std::string_literals;
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
  }

  void visit(LTOp& ltop) override {
    sum_vectors_.push_back(ltop.clone());
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp& parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

  std::vector<std::unique_ptr<Op>> sum_vectors(Op& op) {
      sum_vectors_.clear();
      op.accept(*this);
      return std::move(sum_vectors_);
  }

 private:
 std::vector<std::unique_ptr<Op>> sum_vectors_;
};

class UsedTensorInfoVisitor : public VisitorBase {
public:
    UsedTensorInfoVisitor()                         = default;
    UsedTensorInfoVisitor(const UsedTensorInfoVisitor&) = default;

    UsedTensorInfoVisitor(const SymbolTable& symbol_table) :
      symbol_table_{symbol_table} {}

    void visit(MultOp& mop) override {
        mop.lhs().accept(*this);
        mop.rhs().accept(*this);
        apply_binop(mop, mop.lhs(), mop.rhs());
    }

    void visit(AddOp& aop) override {
        aop.lhs().accept(*this);
        aop.rhs().accept(*this);
        apply_binop(aop, aop.lhs(), aop.rhs());
    }

    void visit(LTOp& ltop) override {
        init_lt(ltop);
    }

    void visit(EinSumOp& einsumop) override {}
    
    void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp& parforop) override {}

    void visit(LambdaOp& lambdaop) override {}

    UsedTensorInfoVisitor(UsedTensorInfoVisitor&& other) :
      UsedTensorInfoVisitor{} {
        swap(*this, other);
    }

    UsedTensorInfoVisitor& operator=(UsedTensorInfoVisitor other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(UsedTensorInfoVisitor& first,
                     UsedTensorInfoVisitor& second) noexcept {
      using std::swap;
      swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
      swap(first.symbol_table_, second.symbol_table_);
    }
private:
    SymbolTable symbol_table_;

    void apply_binop(Op& op, Op& left_child, Op& right_child) {
        EXPECTS(left_child.has_attribute<UsedTensorInfoAttribute>());
        EXPECTS(right_child.has_attribute<UsedTensorInfoAttribute>());
        
        auto lhs_tensors = left_child.get_attribute<UsedTensorInfoAttribute>().get();
        auto rhs_tensors = right_child.get_attribute<UsedTensorInfoAttribute>().get();

        auto tensors = tamm::internal::merge_vector<std::vector<TensorInfo>>(lhs_tensors, rhs_tensors);
        op.set_attribute<UsedTensorInfoAttribute>(tensors);
    }

    void init_lt(LTOp& ltop) {
        auto t_name   = symbol_table_[ltop.tensor().get_symbol_ptr()];
        auto t_ilv    = ltop.labels();
        auto t_eltype = ltop.tensor().to_eltype();
        auto t_scale  = ltop.coeff();

        TensorInfo t_info{t_name, ltop.tensor(), t_ilv, t_eltype, t_scale, false};

        ltop.set_attribute<UsedTensorInfoAttribute>(std::vector<TensorInfo>{t_info});
    }

};

class OpCostVisitor : public VisitorBase {
 public:
  using size = OpCostAttribute::size;
  OpCostVisitor(const OpCostVisitor&) = default;
  OpCostVisitor(const std::map<TiledIndexLabel, size>& index_sizes={})
      : index_sizes_{index_sizes} {}

  void visit(MultOp& mop) override {
    init(mop);
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);

    std::set<TiledIndexLabel> alloc_labels = mop.get_attribute<AllocLabelsAttribute>().get();
    std::set<TiledIndexLabel> sum_labels = mop.get_attribute<SummationLabelsAttribute>().get();

    std::set<TiledIndexLabel> all_labels{alloc_labels};
    all_labels.insert(sum_labels.begin(), sum_labels.end());

    size this_cost = 1;
    for(const auto & il : all_labels) {
        if(index_sizes_.find(il) != index_sizes_.end()) {
            this_cost *= index_sizes_.find(il)->second;
        } else {
            this_cost *= il.tiled_index_space().index_space().max_num_indices();
        }
    }
    size lcost = mop.lhs().get_attribute<OpCostAttribute>().get();
    size rcost = mop.rhs().get_attribute<OpCostAttribute>().get();
    
    mop.set_attribute<OpCostAttribute>(lcost + rcost + this_cost);
  }

  void visit(AddOp& aop) override {
    init(aop);
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    size lcost = aop.lhs().get_attribute<OpCostAttribute>().get();
    size rcost = aop.rhs().get_attribute<OpCostAttribute>().get();
    aop.set_attribute<OpCostAttribute>(lcost + rcost);
  }

  void visit(LTOp& ltop) override { 
    ltop.set_attribute<OpCostAttribute>(0.0); 
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp& parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

  OpCostVisitor(OpCostVisitor&& other) : OpCostVisitor{} { swap(*this, other); }

  OpCostVisitor& operator=(OpCostVisitor other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(OpCostVisitor& first, OpCostVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
    swap(first.index_sizes_, second.index_sizes_);
  }

 private:
  std::map<TiledIndexLabel, size> index_sizes_;

  void init(Op& op) {
    if (!op.has_attribute<SummationLabelsAttribute>() ||
        !op.has_attribute<AllocLabelsAttribute>()) {
      AllocLabelsVisitor al_visitor;
      SummationLabelsVisitor sl_visitor;
      op.accept(al_visitor);
      op.accept(sl_visitor);
    }
  }

};

class ReplaceLTOpVisitor : public VisitorBase {
 public:
  ReplaceLTOpVisitor() = default;
  ReplaceLTOpVisitor(const ReplaceLTOpVisitor&) = default;

  ReplaceLTOpVisitor(const LTOp &target_op, const LTOp &replace_op)
      : target_op_{target_op}, replace_op_{std::move(replace_op)} {}

  ReplaceLTOpVisitor(ReplaceLTOpVisitor&& other) noexcept
      : ReplaceLTOpVisitor{} {
    swap(*this, other);
  }

  ReplaceLTOpVisitor& operator=(ReplaceLTOpVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(ReplaceLTOpVisitor& first,
                   ReplaceLTOpVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
  }

  void visit(MultOp& mop) override {
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
  }

  void visit(AddOp& aop) override {
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
  }

  void visit(LTOp& ltop) override {
    if(ltop.is_equal(target_op_)) {
      ltop = replace_op_;
    }
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

 private:
  LTOp target_op_;
  LTOp replace_op_;
};

class DLPNOLabelsTopDownVisitor : public VisitorBase {
 public:
  DLPNOLabelsTopDownVisitor() = default;
  DLPNOLabelsTopDownVisitor(const DLPNOLabelsTopDownVisitor&) = default;

  DLPNOLabelsTopDownVisitor(const SymbolTable &symbol_table)
      : symbol_table_{symbol_table} {}

  DLPNOLabelsTopDownVisitor(DLPNOLabelsTopDownVisitor&& other) noexcept
      : DLPNOLabelsTopDownVisitor{} {
    swap(*this, other);
  }

  DLPNOLabelsTopDownVisitor& operator=(DLPNOLabelsTopDownVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(DLPNOLabelsTopDownVisitor& first,
                   DLPNOLabelsTopDownVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
  }

  void visit(MultOp& mop) override {
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
    auto lhs_new_labels = mop.lhs().get_attribute<DLPNOLabelsStringAttribute>().get();
    auto rhs_new_labels = mop.rhs().get_attribute<DLPNOLabelsStringAttribute>().get();
    std::map<std::string, std::string> new_labels_map{lhs_new_labels};

    new_labels_map.insert(rhs_new_labels.begin(), rhs_new_labels.end());

    mop.set_attribute<DLPNOLabelsStringAttribute>(new_labels_map);
  }

  void visit(AddOp& aop) override {
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    std::map<std::string, std::string> new_labels_map;
    aop.set_attribute<DLPNOLabelsStringAttribute>(new_labels_map);
  }

  void visit(LTOp& ltop) override {
    auto it = symbol_table_.find(ltop.tensor().get_symbol_ptr());
    EXPECTS(it != symbol_table_.end());
    std::string name = it->second;

    if (name == "t1") {
      it = symbol_table_.find(ltop.labels()[0].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto a = it->second;
      it = symbol_table_.find(ltop.labels()[1].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto i = it->second;
      auto ii = i + i;
      auto a_ii = a + "_" + ii;
      std::map<std::string, std::string> new_labels_map{{a, a_ii}};

      ltop.set_attribute<DLPNOLabelsStringAttribute>(new_labels_map);
    } else if (name == "t2") {

      it = symbol_table_.find(ltop.labels()[0].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto a = it->second;
      it = symbol_table_.find(ltop.labels()[1].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto b = it->second;
      it = symbol_table_.find(ltop.labels()[2].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto i = it->second;
      it = symbol_table_.find(ltop.labels()[3].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto j = it->second;

      auto ij = i + j;
      auto a_ij = a + "_" + ij;
      auto b_ij = b + "_" + ij;

      std::map<std::string, std::string> new_labels_map{{a, a_ij}, {b, b_ij}};

      ltop.set_attribute<DLPNOLabelsStringAttribute>(new_labels_map);
    } else {
      std::map<std::string, std::string> new_labels_map;
      ltop.set_attribute<DLPNOLabelsStringAttribute>(new_labels_map);
    }
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

    void visit(ParForOp &parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

 private:
  SymbolTable symbol_table_;
};

class DLPNOLabelsVisitor : public VisitorBase {
 public:
  DLPNOLabelsVisitor() = default;
  DLPNOLabelsVisitor(const DLPNOLabelsVisitor&) = default;

  DLPNOLabelsVisitor(const SymbolTable &symbol_table)
      : symbol_table_{symbol_table} {}

  DLPNOLabelsVisitor(DLPNOLabelsVisitor&& other) noexcept
      : DLPNOLabelsVisitor{} {
    swap(*this, other);
  }

  DLPNOLabelsVisitor& operator=(DLPNOLabelsVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(DLPNOLabelsVisitor& first,
                   DLPNOLabelsVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
  }

  void visit(MultOp& mop) override {
    init(mop);
    auto new_labels = mop.get_attribute<DLPNOLabelsStringAttribute>().get();

    Op& lhs_op = mop.lhs();
    Op& rhs_op = mop.rhs();

    lhs_op.remove_attribute<DLPNOLabelsStringAttribute>();
    rhs_op.remove_attribute<DLPNOLabelsStringAttribute>();

    lhs_op.set_attribute<DLPNOLabelsStringAttribute>(new_labels);
    rhs_op.set_attribute<DLPNOLabelsStringAttribute>(new_labels);

    auto lhs_labels = lhs_op.get_attribute<DLPNOLabelsStringAttribute>().get();
    auto rhs_labels = rhs_op.get_attribute<DLPNOLabelsStringAttribute>().get();

    lhs_op.accept(*this);
    rhs_op.accept(*this);
  }

  void visit(AddOp& aop) override {
    init(aop);
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
  }

  void visit(LTOp& ltop) override {
    init(ltop);
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp& parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

 private:
  void init(Op& op) {
    if (!op.has_attribute<DLPNOLabelsStringAttribute>()) {
      DLPNOLabelsTopDownVisitor lbls_visitor{symbol_table_};
      op.accept(lbls_visitor);
    }
  }
  
  SymbolTable symbol_table_;
};

class DLPNORewriteVisitor : public VisitorBase {
public:
  DLPNORewriteVisitor() = default;
  DLPNORewriteVisitor(const DLPNORewriteVisitor &) = default;

  DLPNORewriteVisitor(const IndexLabelVec &lhs_pno_labels,
                      const IndexLabelVec &lhs_lmo_pair_labels,
                      const SymbolTable &symbol_table, const TiledIndexSpace& mo_v_space)
      : lhs_pno_labels_{lhs_pno_labels},
        lhs_lmo_pair_labels_{lhs_lmo_pair_labels}, 
        symbol_table_{symbol_table},
        mo_v_space_{mo_v_space} {
    if (!lhs_lmo_pair_labels_.empty()) {
      EXPECTS(lhs_lmo_pair_labels_.size() == 2);
      auto it = symbol_table_.find(lhs_lmo_pair_labels_[0].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      lhs_lmo_pair_str_ += it->second;
      it = symbol_table_.find(lhs_lmo_pair_labels_[1].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      lhs_lmo_pair_str_ += it->second;
    }
    for (const auto &til : lhs_pno_labels_) {
      auto it = symbol_table_.find(til.get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
       lhs_pno_str_.push_back(it->second + "_" + lhs_lmo_pair_str_);
      // lhs_pno_str_.push_back(it->second);
    }
  }

  DLPNORewriteVisitor(DLPNORewriteVisitor &&other) noexcept
      : DLPNORewriteVisitor{} {
    swap(*this, other);
  }

  DLPNORewriteVisitor &operator=(DLPNORewriteVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(DLPNORewriteVisitor &first,
                   DLPNORewriteVisitor &second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase &>(first), static_cast<VisitorBase &>(second));
    swap(first.lhs_pno_labels_, second.lhs_pno_labels_);
    swap(first.lhs_lmo_pair_labels_, second.lhs_lmo_pair_labels_);
    swap(first.symbol_table_, second.symbol_table_);
    swap(first.lhs_lmo_pair_str_, second.lhs_lmo_pair_str_);
    swap(first.lhs_lmo_pair_labels_, second.lhs_lmo_pair_labels_);
    swap(first.mo_v_space_, second.mo_v_space_);
  }

  void visit(MultOp &mop) override {
    init(mop);
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
    mop.set_attribute<DLPNOStringAttribute>(
        mop.coeff().to_string() + "*" +
        mop.lhs().get_attribute<DLPNOStringAttribute>().get() + "*" +
        mop.rhs().get_attribute<DLPNOStringAttribute>().get());
  }

  void visit(AddOp &aop) override {
    init(aop);
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    aop.set_attribute<DLPNOStringAttribute>(
        aop.lhs().get_attribute<DLPNOStringAttribute>().get() + "+\n" +
        aop.rhs().get_attribute<DLPNOStringAttribute>().get());
  }

  void visit(LTOp &ltop) override {
    init(ltop);
    auto it = symbol_table_.find(ltop.tensor().get_symbol_ptr());
    EXPECTS(it != symbol_table_.end());
    std::string name = it->second;
    auto new_label_map = ltop.get_attribute<DLPNOLabelsStringAttribute>().get();

    // add lhs pno maps to new_label_map
    for (size_t i = 0; i < lhs_pno_labels_.size(); i++) {
      auto it = symbol_table_.find(lhs_pno_labels_[i].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto pno_str = it->second;
      if (new_label_map.find(pno_str) == new_label_map.end()) {
        new_label_map[pno_str] = lhs_pno_str_[i];
      }
    }

    std::set<std::string> lmo_pairs;
    for(const auto& [key, value] : new_label_map) {
      lmo_pairs.insert(value.substr(value.rfind("_") + 1));
    }

    for(const auto& value : lhs_pno_str_) {
      lmo_pairs.insert(value.substr(value.rfind("_") + 1));
    }

    if (name == "t1") {
      it = symbol_table_.find(ltop.labels()[0].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto a = it->second;
      it = symbol_table_.find(ltop.labels()[1].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto i = it->second;
      auto ii = i + i;
      auto a_ii = a + "_" + ii;
      std::string transformed_str = tensor_cast_str_ + 
          "dlpno_t1(\"" + a_ii + "\",\""  + ii + "\")" + stransform(a, ii);

      ltop.set_attribute<DLPNOStringAttribute>(transformed_str);
    } else if (name == "t2") {
      it = symbol_table_.find(ltop.labels()[0].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto a = it->second;
      it = symbol_table_.find(ltop.labels()[1].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto b = it->second;
      it = symbol_table_.find(ltop.labels()[2].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto i = it->second;
      it = symbol_table_.find(ltop.labels()[3].get_symbol_ptr());
      EXPECTS(it != symbol_table_.end());
      auto j = it->second;

      auto ij = i + j;
      auto a_ij = a + "_" + ij;
      auto b_ij = b + "_" + ij;

      std::string transformed_str = tensor_cast_str_ + "dlpno_t2(\"" + a_ij + "\",\"" + b_ij + "\",\"" + ij + "\")" +
                                    stransform(a, ij) + stransform(b, ij);

      ltop.set_attribute<DLPNOStringAttribute>(transformed_str);
    } else if (name == "F") {
      EXPECTS(ltop.labels().size() == 2);
      auto labels = ltop.labels();
  
      if (is_mo_v_label(labels[0]) && is_mo_v_label(labels[1])){ // case for VV
        std::vector<std::string> rpaos{"f_mutp", "f_nutp"};
        std::string f_mu_nu = tensor_cast_str_ + R"|(dlpno_F("f_mutp","f_nutp"))|";
        std::string rest;
        for (size_t i = 0; i < labels.size(); i++) {
          auto lbl_str = symbol_table_[labels[i].get_symbol_ptr()];
          if (new_label_map.find(lbl_str)!= new_label_map.end()) {
            auto pno_str = new_label_map[lbl_str];
            auto loc = pno_str.rfind('_');
            auto lmo_pair_str = pno_str.substr(loc+1);
            rest += "*" + constructD(lmo_pair_str, rpaos[i], pno_str);
          }
          
        }
        ltop.set_attribute<DLPNOStringAttribute>(f_mu_nu + rest);
      } else if (!is_mo_v_label(labels[0]) && !is_mo_v_label(labels[1])) {
        std::string o_f_str = symbol_table_[labels[0].get_symbol_ptr()];
        std::string o_s_str = symbol_table_[labels[1].get_symbol_ptr()];


        auto oo_f_lbl = find_lmo_pair(lmo_pairs, o_f_str);
        auto oo_s_lbl = find_lmo_pair(lmo_pairs, o_s_str);

        std::string str = "Foo_xx_yy(" + oo_f_lbl + ", " + oo_s_lbl + ")";
        ltop.set_attribute<DLPNOStringAttribute>(str);
      } else {  // there is no OO use and VO, OV are 0
        // ltop.set_attribute<DLPNOStringAttribute>(
        //     "/*" + OpStringGenerator{symbol_table_}.toString(ltop) + "*/");
        std::string str = "0";
        if(lhs_lmo_pair_labels_.size() == 2) {
          str += " * " + tensor_cast_str_ + "dlpno_";

          if (lhs_lmo_pair_labels_[0] == lhs_lmo_pair_labels_[1]) {
            str += "t1(";
            it = symbol_table_.find(lhs_lmo_pair_labels_[0].get_symbol_ptr());
            EXPECTS(it != symbol_table_.end());

            auto i = it->second;
            auto ii = i + i;

            for(const auto &lbl : lhs_pno_labels_) {
              it = symbol_table_.find(lbl.get_symbol_ptr());
              EXPECTS(it != symbol_table_.end());
              str += "\"" + it->second + "_" + ii + "\",";
            }

            str += "\"" + ii + "\")";
            

          } else {
            str += "t2(";
            it = symbol_table_.find(lhs_lmo_pair_labels_[0].get_symbol_ptr());
            EXPECTS(it != symbol_table_.end());
            auto i = it->second;
            it = symbol_table_.find(lhs_lmo_pair_labels_[1].get_symbol_ptr());
            EXPECTS(it != symbol_table_.end());
            auto j = it->second;
            auto ij = i + j;

            for(const auto &lbl : lhs_pno_labels_) {
              it = symbol_table_.find(lbl.get_symbol_ptr());
              EXPECTS(it != symbol_table_.end());
              str += "\"" + it->second + "_" + ij + "\",";
            }
            str += "\"" + ij + "\")";
          }
        }
        ltop.set_attribute<DLPNOStringAttribute>(str);
      }
      
    } else if (name == "V") {
      EXPECTS(ltop.labels().size() == 4);
      auto labels = ltop.labels();
      auto v_1 = constructVpair({labels[0], labels[2]}, new_label_map, lmo_pairs);
      auto v_2 = constructVpair({labels[1], labels[3]}, new_label_map, lmo_pairs);
      std::string transformedV = v_1 + "*" + v_2;

      ltop.set_attribute<DLPNOStringAttribute>(transformedV);
    }
    else {
      ltop.set_attribute<DLPNOStringAttribute>(
          tensor_cast_str_ + OpStringGenerator{symbol_table_}.toString(ltop));
    }
  }

  void visit(EinSumOp &einsumop) override {}

  void visit(ReshapeOp &reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp &lambdaop) override {}

private:
  void init(Op &op) {
    if (!op.has_attribute<DLPNOLabelsStringAttribute>()) {
      DLPNOLabelsVisitor lbls_visitor{symbol_table_};
      op.accept(lbls_visitor);
    }
  }

  bool has_external_label(const IndexLabelVec &tilv) {
    for(const auto& pno_lbl : lhs_pno_labels_) {
      for(const auto& lbl : tilv) {
        if(pno_lbl == lbl) {
          return true;
        }
      }
    }
    return false;
  }

  bool is_mo_v_label(const TiledIndexLabel &lbl) {

    return lbl.tiled_index_space() == mo_v_space_;
  }

  std::string stransform(const std::string &pno, const std::string &lmo_pair) {
    std::string S_str;
    auto from = std::string{pno + "_" + lmo_pair};
    for (size_t i = 0; i < lhs_pno_str_.size(); i++) {
      const auto &lhs_pno_name = lhs_pno_str_[i];
      auto lhs_name = symbol_table_[lhs_pno_labels_[i].get_symbol_ptr()];
      if (lhs_name == pno && lhs_pno_name != from) {
        auto to = lhs_pno_name;
        //  S_str = "*" + constructS(lmo_pair, lhs_lmo_pair_str_, from, to);
        S_str = "*" + decomposedS(lmo_pair, lhs_lmo_pair_str_, from, to);
      }
    }

    return S_str;
  }

  std::string
  constructVpair(const std::pair<TiledIndexLabel, TiledIndexLabel> &label_pair,
                 const std::map<std::string, std::string> &pno_map,
                 const std::set<std::string>& lmo_pairs) {
    std::string result;
    if(is_mo_v_label(label_pair.first) && is_mo_v_label(label_pair.second)) {
      std::string first_label_str = symbol_table_[label_pair.first.get_symbol_ptr()];
      std::string second_label_str = symbol_table_[label_pair.second.get_symbol_ptr()];
      
      std::string first_rpao_str = "v_" + first_label_str + "_mutp";
      std::string second_rpao_str = "v_" + second_label_str + "_nutp";
      
      std::string first_pno_str = pno_map.at(first_label_str);
      std::string second_pno_str = pno_map.at(second_label_str);
      
      auto first_lmo_pair_str = first_pno_str.substr(first_pno_str.rfind("_")+1);
      auto second_lmo_pair_str = second_pno_str.substr(second_pno_str.rfind("_")+1);

      std::string te_vv = tensor_cast_str_ + "TEvv(\"" + first_rpao_str + "\",\"" + second_rpao_str + R"|(","K"))|";
      std::string rest = "*" + constructD(first_lmo_pair_str, first_rpao_str, first_pno_str);
      rest += "*" + constructD(second_lmo_pair_str, second_rpao_str, second_pno_str);
      result += te_vv + rest;
    } else if (!is_mo_v_label(label_pair.first) && !is_mo_v_label(label_pair.second)) {

      std::string first_label_str = find_lmo_pair(
          lmo_pairs, symbol_table_[label_pair.first.get_symbol_ptr()]);
      std::string second_label_str = find_lmo_pair(
          lmo_pairs, symbol_table_[label_pair.second.get_symbol_ptr()]);

      std::string te_oo = tensor_cast_str_ + "TEoo(\"" + first_label_str + "\",\"" + second_label_str + R"|(","K"))|";
      result += te_oo;
    } else {

      std::string oo_label_str, pno_str, rpao_str, o_label;

      if(!is_mo_v_label(label_pair.first)){
        o_label = symbol_table_[label_pair.first.get_symbol_ptr()];
        oo_label_str =  find_lmo_pair(
          lmo_pairs, o_label);
        pno_str = pno_map.at(symbol_table_[label_pair.second.get_symbol_ptr()]);
        rpao_str = "v_" + symbol_table_[label_pair.second.get_symbol_ptr()] + "_mutp";
      } else {
        o_label = symbol_table_[label_pair.second.get_symbol_ptr()];
        oo_label_str =  find_lmo_pair(
          lmo_pairs, o_label);
        pno_str = pno_map.at(symbol_table_[label_pair.first.get_symbol_ptr()]);
        rpao_str = "v_" + symbol_table_[label_pair.first.get_symbol_ptr()] + "_mutp";
      }

      auto lmo_pair_str = pno_str.substr(pno_str.rfind("_") + 1);
      std::string te_mix_str = "TEmix";
      auto o_lbl_loc = oo_label_str.find(o_label);
      if (o_lbl_loc != 0) {
        te_mix_str += "_1";
      }

      std::string te_mix = tensor_cast_str_ + te_mix_str + "(\"" + oo_label_str + "\",\"" + rpao_str + R"|(","K"))|";
      std::string rest = "*" + constructD(lmo_pair_str, rpao_str, pno_str);
      result += te_mix + rest;
    }

    return result;
  }

  std::string constructD(const std::string &lmo_pair, const std::string &rpao,
                         const std::string &pno) {
    return tensor_cast_str_ + "d(\"" + lmo_pair + "\",\"" + rpao + "\",\"" + pno + "\")";
  }

  std::string constructS(const std::string &ij, const std::string &lhs_ij,
                         const std::string &a_ij, const std::string &lhs_a_ij) {
    return tensor_cast_str_ + "S(\"" + ij + "\",\"" + lhs_ij + "\",\"" + a_ij + "\",\"" + lhs_a_ij + "\")";
  }

  std::string decomposedS(const std::string &ij, const std::string &lhs_ij,
                          const std::string &a_ij, const std::string &lhs_a_ij) {
    std::string a_str{a_ij[0]};
    std::string nutp_str = a_str + "nutp";
    std::string mutp_str = a_str + "mutp";
    std::string D1 = tensor_cast_str_ + "dinv(\"" + ij + "\",\"" + nutp_str + "\",\"" + a_ij + "\")";
    std::string D2 = tensor_cast_str_ + "d(\"" + lhs_ij + "\",\"" + mutp_str + "\",\"" + lhs_a_ij + "\")";
    std::string Stp = tensor_cast_str_ + "Stp(\"" + nutp_str + "\",\"" + mutp_str + "\")";

    return D1 + "*" + Stp + "*" + D2;
  }

  std::string find_lmo_pair(const std::set<std::string> &lmo_pairs,
                            const std::string &mo_lbl) {
    for (const auto &pair : lmo_pairs) {
      if (pair.find(mo_lbl) != std::string::npos) {
        return pair;
      }
    }

    return "";
  }

  IndexLabelVec lhs_pno_labels_;
  IndexLabelVec lhs_lmo_pair_labels_;
  SymbolTable symbol_table_;
  std::string lhs_lmo_pair_str_;
  std::vector<std::string> lhs_pno_str_;
  TiledIndexSpace mo_v_space_;
  std::string tensor_cast_str_ = "(tamm::new_ops::LTOp)";
};

class DLPNOGuardedGenVisitor : public VisitorBase {
  public:
  // DLPNOGuardedGenVisitor() = default;
    DLPNOGuardedGenVisitor(const std::string &guard_name = "")
        : guard_name_{guard_name} {}
    DLPNOGuardedGenVisitor(const DLPNOGuardedGenVisitor &) = default;

    DLPNOGuardedGenVisitor(DLPNOGuardedGenVisitor &&other) noexcept
        : DLPNOGuardedGenVisitor{} {
      swap(*this, other);
  }

  DLPNOGuardedGenVisitor& operator=(DLPNOGuardedGenVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(DLPNOGuardedGenVisitor& first,
                   DLPNOGuardedGenVisitor& second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase&>(first), static_cast<VisitorBase&>(second));
    swap(first.guard_num_, second.guard_num_);
  }

  void visit(MultOp& mop) override {
    std::string dense = mop.get_attribute<ToStringAttribute>("").get();
    std::string dlpno = mop.get_attribute<DLPNOStringAttribute>("").get();
    // guard_num_++;
    // std::string att = "if(" + guard_name(guard_num_++, guard_name_) + ") {\n" +
    //                     op_vector_name() + ".push_back((" + dlpno + ").clone());\n" +
    //                   "} else {\n" + 
    //                     op_vector_name() + ".push_back((" + dense + ").clone());\n" + 
    //                   "}\n";
    std::string att = "// " + std::to_string(guard_num_++) + "\n";
    att += dlpno_op_vector_name() + ".push_back((" + dlpno + ").clone());\n" +
           dense_op_vector_name() + ".push_back((" + dense + ").clone());\n\n";
    mop.set_attribute<DLPNOGuardedGenStringAttribute>(att);
  }

  void visit(AddOp& aop) override {
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    std::string guarded_lhs =
        aop.lhs().get_attribute<DLPNOGuardedGenStringAttribute>("").get();
    std::string guarded_rhs =
        aop.rhs().get_attribute<DLPNOGuardedGenStringAttribute>("").get();
    aop.set_attribute<DLPNOGuardedGenStringAttribute>(guarded_lhs +
                                                      guarded_rhs);
  }

  void visit(LTOp &ltop) override {
    std::string dense = ltop.get_attribute<ToStringAttribute>("").get();
    std::string dlpno = ltop.get_attribute<DLPNOStringAttribute>("").get();
    // std::string att = "if(" + guard_name(guard_num_++, guard_name_) + ") {\n";

    // if(dlpno == "0.0") {
    //   att += "// no-op\n";
    // } else {
    //   att += op_vector_name() + ".push_back((" + dlpno + ").clone());\n";
    // }
    // att += "} else {\n" + op_vector_name() +
    //                   ".push_back((" + dense + ").clone());\n" + "}\n";
    // guard_num_++;
    std::string att = "// " + std::to_string(guard_num_++) + "\n";

    if(dlpno == "0") {
      att += "// no-op for dlpno\n";
    } else {
      att += dlpno_op_vector_name() + ".push_back((" + dlpno + ").clone());\n";
    }
    att += dense_op_vector_name() + ".push_back((" + dense + ").clone());\n\n";

    ltop.set_attribute<DLPNOGuardedGenStringAttribute>(att);
  }

  void visit(EinSumOp& einsumop) override {}

  void visit(ReshapeOp& reshapeop) override {}

  void visit(ParForOp& parforop) override {}

  void visit(LambdaOp& lambdaop) override {}

  std::string guard_declarations(int num_guards = -1) {
    std::string ret;
    if(num_guards == -1) {
      num_guards = guard_num_;
    }

    for(int i=0; i<num_guards; i++) {

      auto g_name =  guard_name(i, guard_name_);
      std::string str = "std::getenv(\"";
      str += g_name + "\") ? std::stoi(std::getenv(\"";
      str += g_name + "\")) == 1 : false;\n";
      ret += "bool " + g_name + " = " + str;
    }

    ret += "\n";

    // for(int i=0; i<num_guards; i++) {
    //   ret += "is_dlpno_op.push_back(" + guard_name(i, guard_name_) + ");\n";
    // }
    // ret += "\n";

    return ret;
  }

  void reset_guard_num() {
    guard_num_ = 0;
  }

  static const std::string& dense_op_name() {
    static std::string name = "dense_op";
    return name;
  }
  static const std::string& dlpno_op_name() {
    static std::string name = "dlpno_op";
    return name;
  }

  static const std::string& op_vector_name() {
    static std::string name = "ops";
    return name; 
  }

  static const std::string& dense_op_vector_name() {
    static std::string name = "dense_ops";
    return name; 
  }

  static const std::string& dlpno_op_vector_name() {
    static std::string name = "dlpno_ops";
    return name; 
  }

  void set_guard_name(const std::string& guard_name) {
    guard_name_ = guard_name;
  }

  static std::string guard_name(unsigned id, const std::string& guard_name) {
    std::ostringstream oss;
    oss << std::fixed << std::setfill('0') << std::setw(4) << id;
    return guard_name + "_guard_enable_" + oss.str();
  }

 private:
  int guard_num_ = 0;
  std::string guard_name_;
};

class ReplaceLabelVisitor : public VisitorBase {
public:
  ReplaceLabelVisitor() = default;
  ReplaceLabelVisitor(const ReplaceLabelVisitor &) = default;

  ReplaceLabelVisitor(const TiledIndexLabel &from, const TiledIndexLabel &to)
      : from_{from}, to_{to} {}

  ReplaceLabelVisitor(ReplaceLabelVisitor &&other) noexcept
      : ReplaceLabelVisitor{} {
    swap(*this, other);
  }

  ReplaceLabelVisitor &operator=(ReplaceLabelVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(ReplaceLabelVisitor &first,
                   ReplaceLabelVisitor &second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase &>(first), static_cast<VisitorBase &>(second));
  }

  void visit(MultOp &mop) override {
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);
  }

  void visit(AddOp &aop) override {
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
  }

  void visit(LTOp &ltop) override { 
    auto tensor_variant = ltop.tensor();
    auto labels = ltop.labels();

    for (size_t i = 0; i < labels.size(); i++) {
      auto lbl = labels[i];
      if(lbl == from_) {
        labels[i] = to_;
      }
    }
    LTOp new_op{tensor_variant, labels};
    ltop = new_op;
  }

  void visit(EinSumOp &einsumop) override {}

  void visit(ReshapeOp &reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp &lambdaop) override {}

private:
  TiledIndexLabel from_;
  TiledIndexLabel to_;
};

class CanonicalizeVisitor : public VisitorBase {
public:
  using LabelPair = std::pair<TiledIndexLabel, TiledIndexLabel>;
  using OpLabelPair = std::pair<std::unique_ptr<Op>, LabelPair>;
  CanonicalizeVisitor() = default;
  CanonicalizeVisitor(const CanonicalizeVisitor &) = default;

  CanonicalizeVisitor(CanonicalizeVisitor &&other) noexcept
      : CanonicalizeVisitor{} {
    swap(*this, other);
  }

  CanonicalizeVisitor &operator=(CanonicalizeVisitor other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(CanonicalizeVisitor &first,
                   CanonicalizeVisitor &second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase &>(first), static_cast<VisitorBase &>(second));
  }

  void visit(MultOp &mop) override { canonicalize_binary_op(mop); }

  void visit(AddOp &aop) override { canonicalize_binary_op(aop); }

  void visit(LTOp &ltop) override { canonicalize_binary_op(ltop); }

  void visit(EinSumOp &einsumop) override {}

  void visit(ReshapeOp &reshapeop) override {}

  void visit(LambdaOp &lambdaop) override {}

  void visit(ParForOp &parforop) override { canonicalize_parfor_op(parforop); }

  static std::vector<OpLabelPair> canonicalize_ops(Op &op) {
    CanonicalizeVisitor c_visitor;
    op.accept(c_visitor);
    return std::move(c_visitor.canonicalized_ops_);
  }

private:
  std::vector<OpLabelPair> canonicalized_ops_;

  /// @bug this implementation won't work with multiple paralel labels as 
  /// it changes the labels one by one.
  void canonicalize_parfor_op(ParForOp &parforop) {
    
    bool no_slicing = true;
    auto labels = parforop.labels();
    for (const auto &lbl : labels) {
      if (lbl.tiled_index_space().num_tiles() == 1) {
        continue;
      }
      
      for (const auto &idx : lbl.tiled_index_space()) {
        TiledIndexSpace slice_TIS{lbl.tiled_index_space(), range(idx, idx + 1)};
        auto new_lbl = slice_TIS.label();
        ReplaceLabelVisitor rl_visitor{lbl, new_lbl};
        auto internal_op = parforop.op().clone();
        internal_op->accept(rl_visitor);
        canonicalized_ops_.push_back(
            {std::move(internal_op), LabelPair{lbl, new_lbl}});
        no_slicing = false;
      }
    }

    if(no_slicing) {
      canonicalized_ops_.push_back({std::move(parforop.op().clone()), LabelPair{}});
    }
  }

  void canonicalize_binary_op(Op &op) {
    canonicalized_ops_.push_back({op.clone(), LabelPair{}});
  }
};

class OpMemCostVisitor : public VisitorBase {
public:
  using size = OpMemCostAttribute::size;
  OpMemCostVisitor(const OpMemCostVisitor &) = default;
  OpMemCostVisitor() = default;

  OpMemCostVisitor(
      const SymbolTable &symbol_table,
      const std::map<std::string, TensorVariant> &inter_tensors = {})
      : symbol_table_{symbol_table}, inter_tensors_{inter_tensors} {}

  void visit(MultOp &mop) override {
    init(mop);
    mop.lhs().accept(*this);
    mop.rhs().accept(*this);

    size lcost = mop.lhs().get_attribute<OpMemCostAttribute>().get();
    size rcost = mop.rhs().get_attribute<OpMemCostAttribute>().get();
    size cost = getMemSize(mop);
    mop.set_attribute<OpMemCostAttribute>(lcost + rcost + cost);
  }

  void visit(AddOp &aop) override {
    init(aop);
    aop.lhs().accept(*this);
    aop.rhs().accept(*this);
    size lcost = aop.lhs().get_attribute<OpMemCostAttribute>().get();
    size rcost = aop.rhs().get_attribute<OpMemCostAttribute>().get();
    size cost = getMemSize(aop);
    aop.set_attribute<OpMemCostAttribute>(lcost + rcost + cost);
  }

  void visit(LTOp &ltop) override {
    init(ltop);
    ltop.set_attribute<OpMemCostAttribute>(0.0);
  }

  void visit(EinSumOp &einsumop) override {}

  void visit(ReshapeOp &reshapeop) override {}

  void visit(ParForOp &parforop) override {}

  void visit(LambdaOp &lambdaop) override {}

  OpMemCostVisitor(OpMemCostVisitor &&other) : OpMemCostVisitor{} {
    swap(*this, other);
  }

  OpMemCostVisitor &operator=(OpMemCostVisitor other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(OpMemCostVisitor &first, OpMemCostVisitor &second) noexcept {
    using std::swap;
    swap(static_cast<VisitorBase &>(first), static_cast<VisitorBase &>(second));
  }

private:
  SymbolTable symbol_table_;
  std::map<std::string, TensorVariant> inter_tensors_;

  size getMemSize(const Op& op) {
    TensorInfo info = op.get_attribute<TensorInfoAttribute>().get();

    if(info.is_intermediate_) {
      return info.tensor_.mem_size();
    }
    
    return 0;
  }

  void init(Op &op) {
    if(!op.has_attribute<TensorInfoAttribute>()) {
      TensorInfoVisitor tiv{symbol_table_};
      op.accept(tiv);
    }
  }
};

} // namespace new_ops
} // namespace tamm
