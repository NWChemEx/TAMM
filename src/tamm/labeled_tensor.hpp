#ifndef TAMM_LABELED_TENSOR_HPP_
#define TAMM_LABELED_TENSOR_HPP_

#include <type_traits>
#include "tamm/ops.hpp"

namespace tamm {

#if __cplusplus >= 202303L

namespace internal {
template<typename>
struct is_tuple : std::false_type {};
template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};
template<typename T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;
} // namespace internal

template<typename T1, typename T2>
auto operator*(T1&& left, T2&& right) {
    using internal::is_tuple_v;
    if constexpr(is_tuple_v<T1>)
        return std::tuple_cat(left, std::forward_as_tuple(right));
    else
        return std::tuple_cat(std::forward_as_tuple(left),
                              std::forward_as_tuple(right));
}

template<typename T>
class Tensor;

template<typename T>
class LabeledTensor {
public:
    using element_type                  = T;
    LabeledTensor()                     = default;
    LabeledTensor(const LabeledTensor&) = default;

    // LabeledTensor(const Tensor<T>& tensor, const IndexLabelVec& ilv) :
    //   tensor_{tensor},
    //   ilv_{ilv} {}

    template<typename... Args>
    LabeledTensor(const Tensor<T>& tensor, Args... args) :
      tensor_{tensor},
      ilv_{IndexLabelVec(tensor_.num_modes())},
      slv_{StringLabelVec(tensor_.num_modes())},
      str_map_{std::vector<bool>(tensor_.num_modes())} {
        unpack(0, args...);
        validate();
    }

    Tensor<T> tensor() const { return tensor_; }
    const IndexLabelVec& labels() const { return ilv_; }
    const StringLabelVec& str_labels() const { return slv_; }
    const std::vector<bool>& str_map() const { return str_map_; }

    void set_labels(const IndexLabelVec& ilv) {
        EXPECTS(ilv_.size() == ilv.size());
        ilv_ = ilv;
        slv_.clear();
        slv_.resize(ilv_.size());
        str_map_ = std::vector<bool>(ilv_.size(), false);
    }

    using LTT = LabeledTensor<T>;

    template<typename T1>
    constexpr auto make_op(T1&& rhs, const bool is_assign,
                           const int sub_v = 1) {
        using internal::is_tuple_v;
        using std::get;
        using std::is_convertible_v;
        using std::is_same_v;
        using std::remove_reference;
        using std::tuple_size_v;

        // LT = alpha
        if constexpr(is_convertible_v<T1, T>)
            return SetOp{*this, T(sub_v * rhs), is_assign};

        // LT = LT
        else if constexpr(is_same_v<T1, LTT>)
            return AddOp{*this, T{sub_v * 1.0}, rhs, is_assign};

        else if constexpr(is_tuple_v<T1>) {
            static_assert(
              !(tuple_size_v<T1>> 3) && !(tuple_size_v<T1> < 2),
              "Operation can only be of the form c [+-]= [alpha] * a [* b]");
            using rhs0_t =
              typename remove_reference<decltype(get<0>(rhs))>::type;
            using rhs1_t =
              typename remove_reference<decltype(get<1>(rhs))>::type;

            if constexpr(tuple_size_v<T1> == 2) {
                // LT = alpha * LT
                if constexpr(is_convertible_v<rhs0_t, T> &&
                             is_same_v<rhs1_t, LTT>)
                    return AddOp{*this, sub_v * get<0>(rhs), get<1>(rhs),
                                 is_assign};
                //  LT = LT * LT
                else if constexpr(is_same_v<rhs0_t, LTT> &&
                                  is_same_v<rhs1_t, LTT>)
                    return MultOp{*this, T{sub_v * 1.0}, get<0>(rhs),
                                  get<1>(rhs), is_assign};
            }

            // LT = alpha * LT * LT
            else if constexpr(tuple_size_v<T1> == 3) {
                using rhs2_t =
                  typename remove_reference<decltype(get<2>(rhs))>::type;
                static_assert(
                  is_convertible_v<rhs0_t, T> && is_same_v<rhs1_t, LTT> &&
                    is_same_v<rhs2_t, LTT>,
                  "Operation can only be of the form c [+-] = alpha * a * b");
                return MultOp{*this, sub_v * get<0>(rhs), get<1>(rhs),
                              get<2>(rhs), is_assign};
            }
        }
    } // end make_op

    template<typename T1>
    auto operator=(T1&& rhs) {
        return make_op(std::move(rhs), true);
    } // operator =

    template<typename T1>
    auto operator+=(T1&& rhs) {
        return make_op(std::move(rhs), false);
    } // operator +=

    template<typename T1>
    auto operator-=(T1&& rhs) {
        return make_op(std::move(rhs), false, -1);
    } // operator -=

protected:
    Tensor<T> tensor_;
    IndexLabelVec ilv_;
    StringLabelVec slv_;
    std::vector<bool> str_map_;

private:
    /**
     * @brief Check that the labeled tensor is valid. Following conditions
     * need to be satisfied:
     *
     * 1. Number of labels in the label vector is equal to the tensor's rank/
     * mode
     *
     * 2. If any label is a dependent label, dependent on some other label l,
     * then l cannot have any key (dep labels)
     *
     * 3. The i-th label's tiled index space is compatible with the tiled
     *  index space corresponding to the tensor's i-th dimension
     *
     * 4. For each label, the number of key labels (dep labels) is either zero
     *  or equal to the number tiled index spaces the label's index space
     *  depends on
     *
     * 5. If any label 'a' is used as being dependent on another label 'l1'
     * (a(l1)), it cannot be a used as being dependent on another label 'l2'.
     * For example, this is not valid: T1(a(l1), a(l2)).
     *
     * 6. If two string labels (say at positions i and j) are identical, the
     *  tensor's index spaces at the same positions (i and j) are identical.
     *
     * 7. If a dimension of a tensor is a dependent dimension, with dependency
     * on dimensions i, j, etc., the corresponding label should be a dependent
     * label on the same dependent index space and should be dependent on the
     * same dimension positions. E.g., with a declaration T{i, a(i), k}, the use
     * T(x, y(z),z) is invalid. For now, we will check that the label is over
     * the same tiled index space as the dimension.
     * 
     * 8. No self dependences. e.g., label 'i(i)', are not allowed.
     *
     */
    void validate() {
        std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
        EXPECTS(tensor_.num_modes() == ilv_.size());
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                for(const auto& dlbl : ilv_[i].dep_labels()) {
                    EXPECTS(dlbl.dep_labels().size() == 0);
                }
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                EXPECTS(ilv_[i].tiled_index_space().is_compatible_with(
                  tensor_.tiled_index_spaces()[i]));
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                size_t sz = ilv_[i].dep_labels().size();
                EXPECTS(sz == 0 || sz == tensor_.tiled_index_spaces()[i]
                                           .index_space()
                                           .num_key_tiled_index_spaces());
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            const auto& ilbl = ilv_[i];
            for(size_t j = i + 1; j < ilv_.size(); j++) {
                if(!str_map_[i] && !str_map_[j]) {
                    const auto& jlbl = ilv_[j];
                    if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                       ilbl.get_label() == jlbl.get_label()) {
                        // EXPECTS(ilbl.dep_labels().size() == 0 ||
                        //         jlbl.dep_labels().size() == 0 ||
                        //         ilbl == jlbl);
                        EXPECTS(ilbl == jlbl);
                    }
                }
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            for(size_t j = i + 1; j < ilv_.size(); j++) {
                if(str_map_[i] && str_map_[j] && slv_[i] == slv_[j]) {
                    const auto& is = tensor_.tiled_index_spaces()[i];
                    const auto& js = tensor_.tiled_index_spaces()[j];
                    EXPECTS(is.is_identical(js));
                }
            }
        }

        const std::map<Index,IndexVector>& dep_map = tensor_.dep_map();
        for(auto itr = dep_map.begin(); itr!=dep_map.end(); ++itr){
            const auto& dep_iv = itr->second;
            auto dc_ = 0;
            for(auto &dlpos: dep_iv) {
                EXPECTS(str_map_[dlpos] == false);
                const auto& ltis = ilv_[dlpos].tiled_index_space();
                Label llbl = ilv_[dlpos].get_label();
                EXPECTS(ilv_[itr->first].dep_labels().size() > 0);
                const auto& rtis = ilv_[itr->first].dep_labels()[dc_].tiled_index_space();
                Label rlbl = ilv_[itr->first].dep_labels()[dc_].get_label();
                EXPECTS(ltis==rtis && llbl==rlbl);
                dc_++;
            }
        }

        for(const auto& lbl : ilv_) {
            for(const auto& dlbl : lbl.dep_labels()) {
                EXPECTS(lbl.tiled_index_space() != dlbl.tiled_index_space() ||
                        lbl.get_label() != dlbl.get_label());
            }
        }
    } // validate

    void unpack(size_t index) {
        if(index == 0) {
            int lc=0;
            for (size_t i=0;i < ilv_.size();i++) 
                ilv_[i] = tensor_.tiled_index_spaces()[i].label(--lc);
            for(size_t i = 0; i < ilv_.size(); i++) {
                auto dep_map = tensor_.dep_map();
                auto itr = dep_map.find(i);
                if(itr != dep_map.end()){
                    IndexLabelVec tempv;
                    for(auto idx: itr->second)
                        tempv.push_back(ilv_[idx]);  
                    ilv_[i] = TiledIndexLabel{ilv_[i],tempv};
                }
                str_map_[i] = false;
            }
        } else {
            EXPECTS(index == tensor_.num_modes());
        }
        EXPECTS(str_map_.size() == tensor_.num_modes());
        EXPECTS(ilv_.size() == tensor_.num_modes());
        EXPECTS(slv_.size() == tensor_.num_modes());
    }

    template<typename... Args>
    void unpack(size_t index, const std::string& str, Args... rest) {
        EXPECTS(index < tensor_.num_modes());
        slv_[index]     = str;
        str_map_[index] = true;
        unpack(++index, rest...);
    }

    template<typename... Args>
    void unpack(size_t index, const TiledIndexLabel& label, Args... rest) {
        EXPECTS(index < tensor_.num_modes());
        ilv_[index]     = label;
        str_map_[index] = false;
        unpack(++index, rest...);
    }
};
#endif

#if __cplusplus <= 201703L

template<typename T>
class Tensor;

// class LoopSpec {
//     public:
//     LoopSpec() : has_oll_{false}, has_ill_{false}, has_symm_factor_{false} {}

//     LoopSpec(const LoopSpec&) = default;

//     LoopSpec(const OuterLabeledLoop& oll) : LoopSpec{} { set_oll(oll); }

//     LoopSpec(const InnerLabeledLoop& ill) : LoopSpec{} { set_ill(ill); }

//     LoopSpec(const SymmFactor& sf) : LoopSpec{} { set_symm_factor(sf); }

//     LoopSpec& set_oll(const OuterLabeledLoop& oll) {
//         oll_     = oll;
//         has_oll_ = true;
//         return *this;
//     }

//     LoopSpec& set_ill(const InnerLabeledLoop& ill) {
//         ill_     = ill;
//         has_ill_ = true;
//         return *this;
//     }

//     LoopSpec& set_symm_factor(const SymmFactor& sf) {
//         symm_factor_     = sf;
//         has_symm_factor_ = true;
//         return *this;
//     }

//     bool has_oll() const { return has_oll_; }

//     bool has_ill() const { return has_ill_; }

//     bool has_symm_factor() const { return has_symm_factor_; }

//     OuterLabeledLoop oll() const { return oll_; }

//     InnerLabeledLoop ill() const { return ill_; }

//     SymmFactor symm_factor() const { return symm_factor_; }

//     private:
//     OuterLabeledLoop oll_;
//     InnerLabeledLoop ill_;
//     SymmFactor symm_factor_;

//     bool has_oll_;
//     bool has_ill_;
//     bool has_symm_factor_;
// };

template<typename T>
class LabeledTensor {
public:
    using element_type                  = T;
    LabeledTensor()                     = default;
    LabeledTensor(const LabeledTensor&) = default;

    template<typename... Args>
    LabeledTensor(const Tensor<T>& tensor, Args... args) :
      tensor_{tensor},
      ilv_{IndexLabelVec(tensor_.num_modes())},
      slv_{StringLabelVec(tensor_.num_modes())},
      str_map_{std::vector<bool>(tensor_.num_modes())} {
        unpack(0, args...);
        validate();
    }

    Tensor<T> tensor() const { return tensor_; }
    const IndexLabelVec& labels() const { return ilv_; }
    const StringLabelVec& str_labels() const { return slv_; }
    const std::vector<bool>& str_map() const { return str_map_; }
    void set_labels(const IndexLabelVec& ilv) {
        EXPECTS(ilv_.size() == ilv.size());
        ilv_ = ilv;
        slv_.clear();
        slv_.resize(ilv_.size());
        str_map_ = std::vector<bool>(ilv_.size(), false);
    }

    AddOp<T, LabeledTensor<T>> operator+=(const LabeledTensor<T> rhs) {
        return construct_addop(std::make_tuple((T)1.0, rhs), false);
    }

    SetOp<T, LabeledTensor<T>> operator+=(const T& rhs) {
        return construct_setop(rhs, false);
    }

    AddOp<T, LabeledTensor<T>> operator-=(const LabeledTensor<T> rhs) {
        return construct_addop(std::make_tuple((T)-1.0, rhs), false);
    }

    SetOp<T, LabeledTensor<T>> operator-=(const T& rhs) {
        return construct_setop(rhs, false);
    }

    SetOp<T, LabeledTensor<T>> operator=(const T& rhs) {
        return construct_setop(rhs, true);
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    AddOp<T, LabeledTensor<T>> operator+=(
      const std::tuple<T1, LabeledTensor<T>>& rhs) {
        return construct_addop(
          std::make_tuple((T)(std::get<0>(rhs)*1.0), std::get<1>(rhs)), false);
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    AddOp<T, LabeledTensor<T>> operator-=(
      const std::tuple<T1, LabeledTensor<T>>& rhs) {
        return construct_addop(
          std::make_tuple((T)(std::get<0>(rhs) * -1.0), std::get<1>(rhs)), false);
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    AddOp<T, LabeledTensor<T>> operator=(
      const std::tuple<T1, LabeledTensor<T>>& rhs) {
        return construct_addop(
          std::make_tuple((T)(std::get<0>(rhs)*1.0), std::get<1>(rhs)), true);
    }

    AddOp<T, LabeledTensor<T>> operator=(const LabeledTensor<T> rhs) {
        return construct_addop(std::make_tuple((T)1.0, rhs), true);
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    MultOp<T, LabeledTensor<T>> operator+=(
      const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(
          std::make_tuple((T)(std::get<0>(rhs)*1.0), std::get<1>(rhs), std::get<2>(rhs)),
          false);
    }

    // @to-do: implement.
    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    MultOp<T, LabeledTensor<T>> operator-=(
      const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(std::make_tuple((T)-1.0 * std::get<0>(rhs),
                                                std::get<1>(rhs),
                                                std::get<2>(rhs)),
                                false);
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    MultOp<T, LabeledTensor<T>> operator=(
      const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(
          std::make_tuple((T)(std::get<0>(rhs)*1.0), std::get<1>(rhs), std::get<2>(rhs)),
          true);
    }

    MultOp<T, LabeledTensor<T>> operator+=(
      const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(
          std::make_tuple((T)1.0, std::get<0>(rhs), std::get<1>(rhs)), false);
    }

    MultOp<T, LabeledTensor<T>> operator-=(
      const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(
          std::make_tuple((T)-1.0, std::get<0>(rhs), std::get<1>(rhs)), false);
    }

    MultOp<T, LabeledTensor<T>> operator=(
      const std::tuple<LabeledTensor<T>, LabeledTensor<T>>& rhs) {
        return construct_multop(
          std::make_tuple((T)1.0, std::get<0>(rhs), std::get<1>(rhs)), true);
    }

protected:
    Tensor<T> tensor_;
    IndexLabelVec ilv_;
    StringLabelVec slv_;
    std::vector<bool> str_map_;

    SetOp<T, LabeledTensor<T>> construct_setop(const T& rhs, bool is_assign) {
        return {*this, rhs, is_assign};
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    AddOp<T1, LabeledTensor<T>> construct_addop(
      const std::tuple<T1, LabeledTensor<T>>& rhs, bool is_assign) {
        addop_validate(*this,
                       std::make_tuple(std::get<0>(rhs), std::get<1>(rhs)));
        T1 alpha         = std::get<0>(rhs);
        auto& rhs_tensor = std::get<1>(rhs);
        return {*this, alpha, rhs_tensor, is_assign};
    }

    template<typename T1,
             typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
    MultOp<T1, LabeledTensor<T>> construct_multop(
      const std::tuple<T1, LabeledTensor<T>, LabeledTensor<T>>& rhs,
      bool is_assign) {
        multop_validate(*this,
                        std::make_tuple(std::get<0>(rhs), std::get<1>(rhs),
                                        std::get<2>(rhs)));

        return {*this, std::get<0>(rhs), std::get<1>(rhs), std::get<2>(rhs),
                is_assign};
    }

    // OuterLabeledLoop loop_nest() const {
    //     // return {labels(),
    //     tensor().perm_group().unique_loop_nest(labels())};
    //   return {};
    // }

    // template<typename T1>
    // static InnerLabeledLoop inner_loop_nest(const LabeledTensor<T1>&
    // ltensor1,
    //                                         const LabeledTensor<T1>&
    //                                         ltensor2) {
    //     using Itr = IndexSpace::Iterator;
    //     IndexLabelVec labels1{ltensor1.labels()};
    //     IndexLabelVec labels2{ltensor2.labels()};

    //     std::sort(labels1.begin(), labels1.end());
    //     std::sort(labels2.begin(), labels2.end());

    //     IndexLabelVec inner_labels;
    //     std::set_intersection(labels1.begin(), labels1.end(),
    //     labels2.begin(),
    //                           labels2.end(),
    //                           std::back_inserter(inner_labels));
    //     std::vector<Itr> begins, ends;
    //     // for(const auto& il : inner_labels) {
    //     //     begins.push_back(il.ir().begin());
    //     //     ends.push_back(il.ir().end());
    //     // }
    //     return InnerLabeledLoop{inner_labels, begins, ends, {}};
    // }

private:
    /**
     * @brief Check that the labeled tensor is valid. Following conditions
     * need to be satisfied:
     *
     * 1. Number of labels in the label vector is equal to the tensor's rank/
     * mode
     *
     * 2. If any label is a dependent label, dependent on some other label l,
     * then l cannot have any key (dep labels)
     *
     * 3. The i-th label's tiled index space is compatible with the tiled
     *  index space corresponding to the tensor's i-th dimension
     *
     * 4. For each label, the number of key labels (dep labels) is either zero
     *  or equal to the number tiled index spaces the label's index space
     *  depends on
     *
     * 5. If any label 'a' is used as being dependent on another label 'l1'
     * (a(l1)), it cannot be a used as being dependent on another label 'l2'.
     * For example, this is not valid: T1(a(l1), a(l2)).
     *
     * 6. If two string labels (say at positions i and j) are identical, the
     *  tensor's index spaces at the same positions (i and j) are identical.
     *
     * 7. If a dimension of a tensor is a dependent dimension, with dependency
     * on dimensions i, j, etc., the corresponding label should be a dependent
     * label on the same dependent index space and should be dependent on the
     * same dimension positions. E.g., with a declaration T{i, a(i), k}, the use
     * T(x, y(z),z) is invalid. For now, we will check that the label is over
     * the same tiled index space as the dimension.
     * 
     * 8. No self dependences. e.g., label 'i(i)', are not allowed.
     *
     */
    void validate() {
        EXPECTS(tensor_.num_modes() == ilv_.size());
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                for(const auto& dlbl : ilv_[i].dep_labels()) {
                    EXPECTS(dlbl.dep_labels().size() == 0);
                }
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                EXPECTS(ilv_[i].tiled_index_space().is_compatible_with(
                  tensor_.tiled_index_spaces()[i]));
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                size_t sz = ilv_[i].dep_labels().size();
                EXPECTS(sz == 0 || sz == tensor_.tiled_index_spaces()[i]
                                           .index_space()
                                           .num_key_tiled_index_spaces());
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            const auto& ilbl = ilv_[i];
            for(size_t j = i + 1; j < ilv_.size(); j++) {
                if(!str_map_[i] && !str_map_[j]) {
                    const auto& jlbl = ilv_[j];
                    if(ilbl.tiled_index_space() == jlbl.tiled_index_space() &&
                       ilbl.get_label() == jlbl.get_label()) {
                        // EXPECTS(ilbl.dep_labels().size() == 0 ||
                        //         jlbl.dep_labels().size() == 0 ||
                        //         ilbl == jlbl);
                        EXPECTS(ilbl == jlbl);
                    }
                }
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            for(size_t j = i + 1; j < ilv_.size(); j++) {
                if(str_map_[i] && str_map_[j] && slv_[i] == slv_[j]) {
                    const auto& is = tensor_.tiled_index_spaces()[i];
                    const auto& js = tensor_.tiled_index_spaces()[j];
                    EXPECTS(is.is_identical(js));
                }
            }
        }

        const std::map<size_t,std::vector<size_t>>& dep_map = tensor_.dep_map();
        for(auto itr = dep_map.begin(); itr!=dep_map.end(); ++itr){
            const auto& dep_iv = itr->second;
            auto dc_ = 0;
            for(auto &dlpos: dep_iv) {
                EXPECTS(str_map_[dlpos] == false);
                const auto& ltis = ilv_[dlpos].tiled_index_space();
                Label llbl = ilv_[dlpos].get_label();
                EXPECTS(ilv_[itr->first].dep_labels().size() > 0);
                const auto& rtis = ilv_[itr->first].dep_labels()[dc_].tiled_index_space();
                Label rlbl = ilv_[itr->first].dep_labels()[dc_].get_label();
                EXPECTS(ltis==rtis && llbl==rlbl);
                dc_++;
            }
        }

        for(const auto& lbl : ilv_) {
            for(const auto& dlbl : lbl.dep_labels()) {
                EXPECTS(lbl.tiled_index_space() != dlbl.tiled_index_space() ||
                        lbl.get_label() != dlbl.get_label());
            }
        }
    } // validate

    void unpack(size_t index) {
        if(index == 0) {
            int lc=0;
            for (size_t i=0;i < ilv_.size();i++) 
                ilv_[i] = tensor_.tiled_index_spaces()[i].label(--lc);
            for(size_t i = 0; i < ilv_.size(); i++) {
                auto dep_map = tensor_.dep_map();
                auto itr = dep_map.find(i);
                if(itr != dep_map.end()){
                    IndexLabelVec tempv;
                    for(auto idx: itr->second)
                        tempv.push_back(ilv_[idx]);  
                    ilv_[i] = TiledIndexLabel{ilv_[i],tempv};
                }
                str_map_[i] = false;
            }
        } else {
            EXPECTS(index == tensor_.num_modes());
        }
        EXPECTS(str_map_.size() == tensor_.num_modes());
        EXPECTS(ilv_.size() == tensor_.num_modes());
        EXPECTS(slv_.size() == tensor_.num_modes());
    }

    template<typename... Args>
    void unpack(size_t index, const std::string& str, Args... rest) {
        EXPECTS(index < tensor_.num_modes());
        slv_[index]     = str;
        str_map_[index] = true;
        unpack(++index, rest...);
    }

    template<typename... Args>
    void unpack(size_t index, const TiledIndexLabel& label, Args... rest) {
        EXPECTS(index < tensor_.num_modes());
        ilv_[index]     = label;
        str_map_[index] = false;
        unpack(++index, rest...);
    }
};

// inline LoopSpec operator*(LoopSpec ls, const InnerLabeledLoop& ill) {
//     return ls.set_ill(ill);
// }

// inline LoopSpec operator*(LoopSpec ls, const SymmFactor& sf) {
//     return ls.set_symm_factor(sf);
// }

// template<typename T>
// inline std::tuple<LoopSpec, T> operator*(LoopSpec ls, T rhs) {
//     return {ls, rhs};
// }

template<typename... Types, typename T>
inline std::tuple<Types..., T> operator*(std::tuple<Types...> lhs, T rhs) {
    return std::tuple_cat(lhs, std::forward_as_tuple(rhs));
}

// @to-do: implement properly
template<typename T>
inline std::tuple<LabeledTensor<T>, LabeledTensor<T>> operator-(
  LabeledTensor<T> lhs, LabeledTensor<T> rhs) {
    return {lhs, rhs};
}

template<typename T1, typename T2,
         typename = std::enable_if_t<std::is_arithmetic<T1>::value>>
inline std::tuple<T1, LabeledTensor<T2>> operator*(
  T1 val, const LabeledTensor<T2>& rhs) {
    return {val, rhs};
}

template<typename T>
inline std::tuple<LabeledTensor<T>, LabeledTensor<T>> operator*(
  const LabeledTensor<T>& rhs1, const LabeledTensor<T>& rhs2) {
    return {rhs1, rhs2};
}

// inline void validate_slicing(const TensorVec<IndexRange>& index_ranges,
//                              const IndexLabelVec& label) {
//     for(size_t i = 0; i < index_ranges.size(); i++) {
//         EXPECTS(index_ranges[i].is_superset_of(label[i].ir()));
//     }
// }

#endif

/**
 * @brief Check if the setop operation is valid. A set operation is valid if:
 *
 * 1. The scalar is convertible to to the tensor element type
 *
 * 2. All labels used as keys in some tensor label are the primary labels in
 * other some other position.
 *
 * @tparam LabeledTensorType type of labeled tensor LHS
 * @tparam T Type of scalar being assigned to the tensor
 * @param ltc Tensor being set
 * @param alpha Scalar used to set the tensor
 *
 * @pre ltc.validate() has been invoked
 */
template<typename LabeledTensorType, typename T>
inline void setop_validate(const LabeledTensorType& ltc, T alpha) {
    using tensor_el_type = typename LabeledTensorType::element_type;

    static_assert(std::is_convertible<T, tensor_el_type>(),
                  "Error setop: mismatch between scalar type and tensor type");

    size_t rank         = ltc.tensor().num_modes();
    const auto& lbl_vec = ltc.labels();
    for(size_t i = 0; i < rank; i++) {
        if(!ltc.str_map()[i] && lbl_vec[i].dep_labels().size() > 0) {
            for(const auto& dlbl : lbl_vec[i].dep_labels()) {
                size_t j = 0;
                for(; j < rank; j++) {
                    if(dlbl.primary_label() == lbl_vec[j].primary_label()) {
                        break;
                    }
                }
                EXPECTS(j < rank);
            }
        }
    }
}

/**
 * @brief Check if the parameters forma valid add operation. The parameters
 * (ltc, tuple(alpha,lta)) form a valid add operation if:
 *
 * 1. Every label depended on by another label (i.e., all 'd' such that there
 *  exists label 'l(d)') is bound at least once
 *
 * 2. There are no conflicting dependent label specifications. That if 'a(i)'
 * is a label in either lta or ltc,
 * there is no label 'a(j)' (i!=j) in either lta or ltc.
 *
 * @tparam LabeledTensorType Type RHS labeled tensor
 * @tparam T Type of scaling factor (alpha)
 * @param ltc LHS tensor being added to
 * @param rhs RHS (scaling factor and labeled tensor)
 *
 * @pre ltc.validate() has been invoked
 * @pre lta.validate() has been invoked
 */
template<typename LabeledTensorType, typename T>
inline void addop_validate(const LabeledTensorType& ltc,
                           const std::tuple<T, LabeledTensorType>& rhs) {
    T alpha         = std::get<0>(rhs);
    const auto& lta = std::get<1>(rhs);
#if 0
    auto lta = rhs1_t;
    // EXPECTS(ltc.tensor() != nullptr);
    // EXPECTS(lta.tensor() != nullptr);
    const auto& tc = ltc.tensor();
    const auto& ta = lta.tensor();

    // tensors should have same rank
    EXPECTS(tc.rank() == ta.rank());

    IndexLabelVec clabel = ltc.labels();
    IndexLabelVec alabel = lta.labels();

    // index range underlying an index label is the same or a subset of the
    // tensor’s index range along that dimension
    validate_slicing(tc.dim_ranges(), ltc.labels());
    validate_slicing(ta.dim_ranges(), lta.labels());

    // length of the index label vector matches the rank (number of indices) in
    // the tensor
    EXPECTS(alabel.size() == ta.rank());
    EXPECTS(clabel.size() == tc.rank());

#if 0
  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    EXPECTS(is_range_subset(ta.flindices()[i], alabel[i].rt()));
  }
  for(int i=0; i<clabel.size(); i++) {
    EXPECTS(is_range_subset(tc.flindices()[i], clabel[i].rt()));
  }
#endif

    std::sort(alabel.begin(), alabel.end());
    std::sort(clabel.begin(), clabel.end());

    // all labels are unique
    EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
    EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

    // all labels in ta are in tb
    for(auto& al : alabel) {
        EXPECTS(std::find(clabel.begin(), clabel.end(), al) != clabel.end());
    }
#endif
}

template<typename LabeledTensorType, typename T>
inline void multop_validate(
  const LabeledTensorType& ltc,
  const std::tuple<T, LabeledTensorType, LabeledTensorType>& rhs) {
#if 0
    auto& lta = rhs1_t;
    auto& ltb = get<2>(rhs);
    // EXPECTS(ltc.tensor_ != nullptr);
    // EXPECTS(lta.tensor_ != nullptr);
    // EXPECTS(ltb.tensor_ != nullptr);
    const auto& tc = ltc.tensor();
    const auto& ta = lta.tensor();
    const auto& tb = ltb.tensor();

    IndexLabelVec clabel = ltc.labels();
    IndexLabelVec alabel = lta.labels();
    IndexLabelVec blabel = ltb.labels();

    // length of the index label vector matches the rank (number of indices) in
    // the tensor
    EXPECTS(clabel.size() == tc.rank());
    EXPECTS(alabel.size() == ta.rank());
    EXPECTS(blabel.size() == tb.rank());

    // index range underlying an index label is the same or a subset of the
    // tensor’s index range along that dimension
    validate_slicing(tc.dim_ranges(), ltc.labels());
    validate_slicing(ta.dim_ranges(), lta.labels());
    validate_slicing(tb.dim_ranges(), ltb.labels());

#if 0
  //all labels are of compatible type
  for(int i=0; i<alabel.size(); i++) {
    EXPECTS(is_range_subset(ta.flindices()[i], alabel[i].rt()));
  }
  for(int i=0; i<blabel.size(); i++) {
    EXPECTS(is_range_subset(tb.flindices()[i], blabel[i].rt()));
  }
  for(int i=0; i<clabel.size(); i++) {
    EXPECTS(is_range_subset(tc.flindices()[i], clabel[i].rt()));
  }
#endif

    std::sort(alabel.begin(), alabel.end());
    std::sort(blabel.begin(), blabel.end());
    std::sort(clabel.begin(), clabel.end());

    // all labels are unique
    EXPECTS(std::adjacent_find(alabel.begin(), alabel.end()) == alabel.end());
    EXPECTS(std::adjacent_find(blabel.begin(), blabel.end()) == blabel.end());
    EXPECTS(std::adjacent_find(clabel.begin(), clabel.end()) == clabel.end());

    IndexLabelVec rhs_labels;
    std::set_union(alabel.begin(), alabel.end(), blabel.begin(), blabel.end(),
                   std::back_inserter(rhs_labels));

    IndexLabelVec inner_labels;
    std::set_difference(rhs_labels.begin(), rhs_labels.end(), clabel.begin(),
                        clabel.end(), std::back_inserter(inner_labels));

    IndexLabelVec slabel;
    std::set_intersection(alabel.begin(), alabel.end(), blabel.begin(),
                          blabel.end(), std::back_inserter(slabel));

    // Every outer index label (clabel) appears in exactly one RHS tensor
    for(auto& ol : clabel) {
        EXPECTS(std::find(slabel.begin(), slabel.end(), ol) == slabel.end() &&
                std::find(rhs_labels.begin(), rhs_labels.end(), ol) !=
                  rhs_labels.end());
    }

    // Every inner index label appears exactly once in both RHS tensors
    for(auto& il : inner_labels) {
        EXPECTS(std::find(slabel.begin(), slabel.end(), il) != slabel.end());
    }

    // //summation index is not in the output
    // for(auto &sl: slabel) {
    //   EXPECTS(std::find(clabel.begin(), clabel.end(), sl) == clabel.end());
    // }
    // //every label in A/B is either in slabel or clabel
    // for(auto &al : alabel) {
    //   EXPECTS(std::find(slabel.begin(), slabel.end(), al) != slabel.end()
    //           || std::find(clabel.begin(), clabel.end(), al) !=
    //           clabel.end());
    // }
    // for(auto &bl : blabel) {
    //   EXPECTS(std::find(slabel.begin(), slabel.end(), bl) != slabel.end()
    //           || std::find(clabel.begin(), clabel.end(), bl) !=
    //           clabel.end());
    // }

    EXPECTS(clabel.size() == alabel.size() + blabel.size() - 2 * slabel.size());
#endif
}

} // namespace tamm
#endif // LABELED_TENSOR_HPP_
