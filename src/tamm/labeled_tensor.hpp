#ifndef TAMM_LABELED_TENSOR_HPP_
#define TAMM_LABELED_TENSOR_HPP_

#include "tamm/ops.hpp"
#include <type_traits>

namespace tamm {

namespace internal {
template<typename>
struct is_tuple : std::false_type {};
template<typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};
template<typename T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

template<typename> struct is_complex : std::false_type {};
template<typename T> struct is_complex<std::complex<T>> : std::true_type {};
template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;


} // namespace internal

template<typename T>
class Tensor;

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

    LabeledTensor(const Tensor<T>& tensor, const IndexLabelVec& labels) :
      tensor_{tensor},
      ilv_{IndexLabelVec(tensor_.num_modes())},
      slv_{StringLabelVec(tensor_.num_modes())},
      str_map_{std::vector<bool>(tensor_.num_modes())} {
        unpack(0, labels);
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

    using LTT = LabeledTensor<T>; //this is LHS, has to be complex if one of RHS is complex
    using LTT_int = LabeledTensor<int>;
    using LTT_float = LabeledTensor<float>;
    using LTT_double = LabeledTensor<double>;

    template<typename T1>
    constexpr auto make_op(T1&& rhs, const bool is_assign,
                           const int sub_v = 1) {
        using internal::is_tuple_v;
        using internal::is_complex_v;
        using std::get;
        using std::is_convertible_v;
        using std::is_same_v;
        using std::remove_reference;
        using std::tuple_size_v;

        // LT = alpha
        if constexpr(is_convertible_v<T1, T>)
            return SetOp<T, LTT>{*this, static_cast<T>(sub_v * rhs), is_assign};

        // LT = LT
        else if constexpr(is_same_v<T1, LTT>)
            return AddOp<T, LTT, T1>{*this, static_cast<T>(sub_v), rhs, is_assign};
        else if constexpr(is_complex_v<T> && 
                          (is_same_v<T1,LTT_int>
                          ||is_same_v<T1,LTT_float>
                          ||is_same_v<T1,LTT_double>))
            return AddOp<int, LTT, T1>{*this, sub_v, rhs, is_assign};            

        else if constexpr(is_tuple_v<T1>) {
            static_assert(
              !(tuple_size_v<T1> > 3) && !(tuple_size_v<T1> < 2),
              "Operation can only be of the form c [+-]= [alpha *] a [* b]");
            using rhs0_t =
              typename remove_reference<decltype(get<0>(rhs))>::type;
            using rhs1_t =
              typename remove_reference<decltype(get<1>(rhs))>::type;

            if constexpr(tuple_size_v<T1> == 2) {
                // LT = alpha * LT
                if constexpr((is_convertible_v<rhs0_t, T>)&&is_same_v<rhs1_t,
                                                                      LTT>)
                    return AddOp<T, LTT, rhs1_t>{*this,
                                         static_cast<T>(sub_v * get<0>(rhs)),
                                         get<1>(rhs), is_assign};
                else if constexpr( is_convertible_v<rhs0_t, T> && 
                                    is_complex_v<T> && 
                                    (is_same_v<rhs1_t,LTT_int>
                                    ||is_same_v<rhs1_t,LTT_float>
                                    ||is_same_v<rhs1_t,LTT_double>)                     
                                 )
                    return AddOp<double, LTT, rhs1_t>{*this,
                                         static_cast<double>(sub_v * get<0>(rhs)),
                                         get<1>(rhs), is_assign};

                //  LT = LT * LT
                else if constexpr(is_same_v<rhs0_t, LTT> &&
                                  is_same_v<rhs1_t, LTT>)
                    return MultOp<T, LTT, LTT, LTT>{*this, static_cast<T>(sub_v),
                                          get<0>(rhs), get<1>(rhs), is_assign};
                else if constexpr(is_complex_v<T> && 
                                    (
                                      (is_same_v<rhs0_t, LTT> &&
                                        (is_same_v<rhs1_t,LTT_int>
                                        ||is_same_v<rhs1_t,LTT_float>
                                        ||is_same_v<rhs1_t,LTT_double>)
                                      )
                                      ||
                                      (is_same_v<rhs1_t, LTT> &&
                                        (is_same_v<rhs0_t,LTT_int>
                                        ||is_same_v<rhs0_t,LTT_float>
                                        ||is_same_v<rhs0_t,LTT_double>)
                                      )
                                    )
                                 ) 
                    return MultOp<int, LTT, rhs0_t, rhs1_t>{*this, sub_v,
                                    get<0>(rhs), get<1>(rhs), is_assign};                                          
            }

            // LT = alpha * LT * LT
            else if constexpr(tuple_size_v<T1> == 3) {
                using rhs2_t =
                  typename remove_reference<decltype(get<2>(rhs))>::type;

                if constexpr(is_same_v<rhs1_t, LTT> &&
                             is_same_v<rhs2_t, LTT>)
                    return MultOp<T, LTT, LTT, LTT>{*this,
                                      static_cast<T>(sub_v * get<0>(rhs)),
                                      get<1>(rhs), get<2>(rhs), is_assign};
                else if constexpr(is_complex_v<T> && 
                                    (
                                      (is_same_v<rhs1_t, LTT> &&
                                        (is_same_v<rhs2_t,LTT_int>
                                        ||is_same_v<rhs2_t,LTT_float>
                                        ||is_same_v<rhs2_t,LTT_double>)
                                      )
                                      ||
                                      (is_same_v<rhs2_t, LTT> &&
                                        (is_same_v<rhs1_t,LTT_int>
                                        ||is_same_v<rhs1_t,LTT_float>
                                        ||is_same_v<rhs1_t,LTT_double>)
                                      )
                                    )
                                 ) 
                    return MultOp<rhs0_t, LTT, rhs1_t, rhs2_t>{*this, 
                                    static_cast<rhs0_t>(sub_v * get<0>(rhs)),
                                    get<1>(rhs), get<2>(rhs), is_assign};
                // static_assert(
                //   (is_convertible_v<rhs0_t, T>)&&is_same_v<rhs1_t, LTT> &&
                //     is_same_v<rhs2_t, LTT>,
                //   "Operation can only be of the form c [+-] = [alpha *] a * b");
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

    TensorBase* base_ptr() const {
        return tensor_.base_ptr();
    }
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
     * then l cannot have any key (dep labels) (REMOVED)
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
        // for(size_t i = 0; i < ilv_.size(); i++) {
        //     if(!str_map_[i]) {
        //         for(const auto& dlbl : ilv_[i].secondary_labels()) {
        //             EXPECTS(dlbl.secondary_labels().size() == 0);
        //         }
        //     }
        // }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                EXPECTS(ilv_[i].tiled_index_space().is_compatible_with(
                  tensor_.tiled_index_spaces()[i]));
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            if(!str_map_[i]) {
                size_t sz = ilv_[i].secondary_labels().size();
                EXPECTS(sz == 0 || sz == tensor_.tiled_index_spaces()[i]
                                           .num_key_tiled_index_spaces());
            }
        }
        for(size_t i = 0; i < ilv_.size(); i++) {
            const auto& ilbl = ilv_[i];
            for(size_t j = i + 1; j < ilv_.size(); j++) {
                if(!str_map_[i] && !str_map_[j]) {
                    const auto& jlbl = ilv_[j];
                    if(ilbl.primary_label() == jlbl.primary_label()) {
                        // EXPECTS(ilbl.secondary_labels().size() == 0 ||
                        //         jlbl.secondary_labels().size() == 0 ||
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

        const std::map<size_t, std::vector<size_t>>& dep_map =
          tensor_.dep_map();
        for(auto itr = dep_map.begin(); itr != dep_map.end(); ++itr) {
            const auto& dep_iv = itr->second;
            auto dc_           = 0;
            for(auto& dlpos : dep_iv) {
                EXPECTS(str_map_[dlpos] == false);
                const auto& ltis = ilv_[dlpos].tiled_index_space();
                Label llbl       = ilv_[dlpos].label();
                EXPECTS(ilv_[itr->first].secondary_labels().size() > 0);
                const auto& rtis =
                  ilv_[itr->first].secondary_labels()[dc_].tiled_index_space();
                Label rlbl = ilv_[itr->first].secondary_labels()[dc_].label();
                EXPECTS(ltis == rtis && llbl == rlbl);
                dc_++;
            }
        }

        for(const auto& lbl : ilv_) {
            for(const auto& dlbl : lbl.secondary_labels()) {
                EXPECTS(lbl.primary_label() != dlbl);
            }
        }
    } // validate

    void unpack(size_t index) {
        if(index == 0) {
            int lc = 0;
            for(size_t i = 0; i < ilv_.size(); i++)
                ilv_[i] = tensor_.tiled_index_spaces()[i].label(--lc);
            for(size_t i = 0; i < ilv_.size(); i++) {
                auto dep_map = tensor_.dep_map();
                auto itr     = dep_map.find(i);
                if(itr != dep_map.end()) {
                    IndexLabelVec tempv;
                    for(auto idx : itr->second) tempv.push_back(ilv_[idx]);
                    ilv_[i] = TiledIndexLabel{ilv_[i], tempv};
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

    /// @todo: Implement 
    template<typename... Args>
    void unpack(size_t index, const Index& idx, Args... rest) {
        // EXPECTS(index < tensor_.num_modes());
        // ilv_[index]     = label;
        // str_map_[index] = false;
        // unpack(++index, rest...);
    }

    void unpack(size_t index, const IndexLabelVec& labels) {
        EXPECTS(index < tensor_.num_modes());
        for(auto label : labels) {
            ilv_[index]     = label;
            str_map_[index] = false;
            ++index;
        }
    }
};

template<typename... Types, typename T>
inline std::tuple<Types..., T> operator*(std::tuple<Types...> lhs, T rhs) {
    return std::tuple_cat(lhs, std::forward_as_tuple(rhs));
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

template<typename T1, typename T2>
inline std::tuple<LabeledTensor<T1>, LabeledTensor<T2>> operator*(
  const LabeledTensor<T1>& rhs1, const LabeledTensor<T2>& rhs2) {
    return {rhs1, rhs2};
}

} // namespace tamm
#endif // LABELED_TENSOR_HPP_
