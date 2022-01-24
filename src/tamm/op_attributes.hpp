#pragma once

#include "tamm/interfaces.hpp"
#include "tamm/scalar.hpp"
#include "tamm/types.hpp"
#include "tamm/tensor_variant.hpp"

namespace tamm {

namespace new_ops {

enum class BinOpType { noop, multop, addop, setop };

struct TensorInfo {
    TensorInfo() = default;
    // Copy/Move Ctors and Assignment Operators
    TensorInfo(TensorInfo &&) = default;
    TensorInfo(const TensorInfo &) = default;
    TensorInfo &operator=(TensorInfo &&) = default;
    TensorInfo &operator=(const TensorInfo &) = default;

    TensorInfo(const std::string& name, TensorVariant tensor,
               const IndexLabelVec& ilv, ElType eltype, const Scalar& scale,
               bool is_intermediate) :
      name_{name},
      tensor_{tensor},
      ilv_{ilv},
      eltype_{eltype},
      scale_{scale},
      is_intermediate_{is_intermediate} {}

    std::string to_string() const {
        std::string result = "";
        result += "tensor_name = " + name_ + 
                  " tensor_type = " + tensor_.to_string() +
                  " el_type = " + eltype_to_string(eltype_) +
                  " lbl_count = " + std::to_string(ilv_.size()) + 
                  " scale = " + scale_.to_string();
        return result;
    }

    std::string op_string(SymbolTable& symbol_table, bool print_scale = true) const {
        std::string result = "";

        if (scale_.value() != Scalar{eltype_}.value() && print_scale) {
            result += scale_.to_string() + " * ";
        }
        if(name_ == "") {
          result += "TempTensor(";
        } else {
          result += name_ + "(";
        }

        for (const auto &lbl : ilv_) {
          auto it = symbol_table.find(lbl.get_symbol_ptr());
          if (it == symbol_table.end()) {
            std::string temp_lbl_str = "";
            if (lbl.label_str() == "") {
              temp_lbl_str = "_lbl_" + std::to_string(lbl.label());
            } else {
              temp_lbl_str = lbl.label_str();
            }
            result += temp_lbl_str;
            
          } else {
            result += symbol_table.at(lbl.get_symbol_ptr());
            if (!lbl.secondary_labels().empty()) {
              result += '(';
              for (const auto &dep_lbl : lbl.secondary_labels()) {
                EXPECTS(symbol_table.find(dep_lbl.get_symbol_ptr()) !=
                        symbol_table.end());
                result += symbol_table.at(dep_lbl.get_symbol_ptr()) + ", ";
              }
              result.pop_back();
              result.pop_back();
              result += ')';
            }
          }
          result += ", ";
        }

        if(!ilv_.empty()){
            result.pop_back();
            result.pop_back();
        }

        result += ')';

        return result;
    }

    std::string name_;
    TensorVariant tensor_;
    IndexLabelVec ilv_;
    ElType eltype_;
    Scalar scale_;
    bool is_intermediate_;
};

struct BinarizedOp {

    BinarizedOp() = default;
    // Copy/Move Ctors and Assignment Operators
    BinarizedOp(BinarizedOp &&) = default;
    BinarizedOp(const BinarizedOp &) = default;
    BinarizedOp &operator=(BinarizedOp &&) = default;
    BinarizedOp &operator=(const BinarizedOp &) = default;

    
    BinarizedOp(const TensorInfo& lhs, const TensorInfo& rhs1,
                const TensorInfo& rhs2, const Scalar& scale, BinOpType optype,
                bool is_assign = true) :
      lhs_{lhs},
      rhs1_{rhs1},
      rhs2_{rhs2},
      scale_{scale},
      optype_{optype},
      is_assign_{is_assign} {}
    
    std::string to_string() const {
        std::string result = "";
        result += "( lhs = " + lhs_.to_string() + " )";
        if(is_assign_) 
            result += " = ";
        else
            result += " += ";
        result += "( rhs1 = " + rhs1_.to_string() + " )";
        if(optype_ == BinOpType::addop) 
            result += " + ";
        else if (optype_ == BinOpType::multop)
            result += " * ";
        if(optype_ != BinOpType::setop) {
            result += "( rhs2 = " + rhs2_.to_string() +
                      " * l_scale = " + scale_.to_string() + " )";
        }

        return result;
    }

    std::string op_string(SymbolTable& symbol_table) const {
        std::string result = "";

        auto lhs_op_str  = lhs_.op_string(symbol_table, false);
        auto rhs1_op_str = rhs1_.op_string(symbol_table, !rhs1_.is_intermediate_);
        auto rhs2_op_str = rhs2_.op_string(symbol_table, !rhs2_.is_intermediate_);

        result += lhs_op_str + ' ';
        if(!is_assign_) { result += '+'; }
        result += "= ";

        if(scale_.value() != Scalar{lhs_.eltype_}.value()) {
            result += scale_.to_string() + " * ";
        }


        if(optype_ == BinOpType::setop) {
            result += rhs1_op_str + ";";
        }
        
        if(optype_ == BinOpType::addop) {
            result += rhs1_op_str + ";\n";
            result += lhs_op_str + " += " + rhs2_op_str + ';';

        
        }
        else if (optype_ == BinOpType::multop) {
            result += rhs1_op_str + " * " + rhs2_op_str + ';';
        }
            

        return result;
    }

    TensorInfo lhs_;
    TensorInfo rhs1_;
    TensorInfo rhs2_;
    Scalar scale_;
    BinOpType optype_;
    bool is_assign_;
};

template<typename T>
class AddAttributeId;

class OpAttribute : public Cloneable<OpAttribute> {
public:
    virtual ~OpAttribute() = default;

protected:
    static int create_attribute_id() {
        static int attribute_id = 0;
        return ++attribute_id;
    }
    template<typename T>
    friend class AddAttributeId;
};

template<typename T>
class AddAttributeId {
public:
    static int id() {
        static const int attr_id = OpAttribute::create_attribute_id();
        return attr_id;
    }
};

template<typename T>
class AttributeInfo {
public:
    using Typename = T;
    AttributeInfo(T info) {
        using std::swap;
        swap(info_, info);
    }

    const T& get() const { return info_; }

    void set(const T& info) { info_ = info; }

    friend void swap(AttributeInfo<T>& first,
                     AttributeInfo<T>& second) noexcept {
        using std::swap;
        swap(first.info_, second.info_);
    }

protected:
    T info_;
};

template<typename T>
class AttributeInfo<std::set<T>> {
public:
    using InfoType = std::set<T>;
    AttributeInfo(InfoType info) {
        using std::swap;
        swap(info_, info);
    }

    template<typename Container>
    AttributeInfo(const Container& info) : info_{info.begin(), info.end()} {}

    const InfoType& get() const { return info_; }

    void set(const InfoType& info) { info_ = info; }

    template<typename Container>
    void set(const Container& info) {
        info_.clear();
        info_.insert(info.begin(), info.end());
    }

    friend void swap(AttributeInfo<InfoType>& first,
                     AttributeInfo<InfoType>& second) noexcept {
        using std::swap;
        swap(first.info_, second.info_);
    }

protected:
    InfoType info_;
};

class ElTypeAttribute
  : public InheritWithCloneable<ElTypeAttribute, OpAttribute>,
    public AddAttributeId<ElTypeAttribute>,
    public AttributeInfo<ElType> {
public:
    using AttributeInfo<ElType>::AttributeInfo;
};

class NameAttribute : public InheritWithCloneable<NameAttribute, OpAttribute>,
                      public AddAttributeId<NameAttribute>,
                      public AttributeInfo<std::string> {
public:
    using AttributeInfo<std::string>::AttributeInfo;
};

class AvailableLabelsAttribute
  : public InheritWithCloneable<AvailableLabelsAttribute, OpAttribute>,
    public AddAttributeId<AvailableLabelsAttribute>,
    public AttributeInfo<std::set<TiledIndexLabel>> {
public:
    using AttributeInfo<std::set<TiledIndexLabel>>::AttributeInfo;
};

class NeededLabelsAttribute
  : public InheritWithCloneable<NeededLabelsAttribute, OpAttribute>,
    public AddAttributeId<NeededLabelsAttribute>,
    public AttributeInfo<std::set<TiledIndexLabel>> {
public:
    using AttributeInfo<std::set<TiledIndexLabel>>::AttributeInfo;
};

class AllocLabelsAttribute
  : public InheritWithCloneable<AllocLabelsAttribute, OpAttribute>,
    public AddAttributeId<AllocLabelsAttribute>,
    public AttributeInfo<std::set<TiledIndexLabel>> {
public:
    using AttributeInfo<std::set<TiledIndexLabel>>::AttributeInfo;
};

class BinarizedStringAttribute
  : public InheritWithCloneable<BinarizedStringAttribute, OpAttribute>,
    public AddAttributeId<BinarizedStringAttribute>,
    public AttributeInfo<std::vector<std::string>> {
public:
    using AttributeInfo<std::vector<std::string>>::AttributeInfo;
};

class ToStringAttribute
  : public InheritWithCloneable<ToStringAttribute, OpAttribute>,
    public AddAttributeId<ToStringAttribute>,
    public AttributeInfo<std::string> {
public:
    using AttributeInfo<std::string>::AttributeInfo;
};

class TensorInfoAttribute
  : public InheritWithCloneable<TensorInfoAttribute, OpAttribute>,
    public AddAttributeId<TensorInfoAttribute>,
    public AttributeInfo<TensorInfo> {
public:
    using AttributeInfo<TensorInfo>::AttributeInfo;
};

class UsedTensorInfoAttribute
  : public InheritWithCloneable<UsedTensorInfoAttribute, OpAttribute>,
    public AddAttributeId<UsedTensorInfoAttribute>,
    public AttributeInfo<std::vector<TensorInfo>> {
public:
    using AttributeInfo<std::vector<TensorInfo>>::AttributeInfo;
};

class SummationLabelsAttribute
    : public InheritWithCloneable<SummationLabelsAttribute, OpAttribute>,
      public AddAttributeId<SummationLabelsAttribute>,
      public AttributeInfo<std::set<TiledIndexLabel>> {
 public:
  using AttributeInfo<std::set<TiledIndexLabel>>::AttributeInfo;
};

class OpCostAttribute
    : public InheritWithCloneable<OpCostAttribute, OpAttribute>,
      public AddAttributeId<OpCostAttribute>,
      public AttributeInfo<double> {
 public:
  using size = double;
  using AttributeInfo<size>::AttributeInfo;
};

class DLPNOStringAttribute
    : public InheritWithCloneable<DLPNOStringAttribute, OpAttribute>,
      public AddAttributeId<DLPNOStringAttribute>,
      public AttributeInfo<std::string> {
 public:
  using AttributeInfo<std::string>::AttributeInfo;
};

class DLPNOGuardedGenStringAttribute
    : public InheritWithCloneable<DLPNOGuardedGenStringAttribute, OpAttribute>,
      public AddAttributeId<DLPNOGuardedGenStringAttribute>,
      public AttributeInfo<std::string> {
 public:
  using AttributeInfo<std::string>::AttributeInfo;
};

// class DLPNOGuardedGenCounterAttribute
//     : public InheritWithCloneable<DLPNOGuardedGenCounterAttribute, OpAttribute>,
//       public AddAttributeId<DLPNOGuardedGenCounterAttribute>,
//       public AttributeInfo<int> {
//  public:
//   using AttributeInfo<int>::AttributeInfo;
// };

class DLPNOLabelsStringAttribute
    : public InheritWithCloneable<DLPNOLabelsStringAttribute, OpAttribute>,
      public AddAttributeId<DLPNOLabelsStringAttribute>,
      public AttributeInfo<std::map<std::string, std::string>> {
 public:
  using AttributeInfo<std::map<std::string, std::string>>::AttributeInfo;
};

class OpMemCostAttribute
    : public InheritWithCloneable<OpMemCostAttribute, OpAttribute>,
      public AddAttributeId<OpMemCostAttribute>,
      public AttributeInfo<double> {
 public:
  using size = double;
  using AttributeInfo<size>::AttributeInfo;
};

} // namespace new_ops
} // namespace tamm

