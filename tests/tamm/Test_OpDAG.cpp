#include <chrono>
#include <random>
#include <tamm/block_plan.hpp>
#include <tamm/op_dag.hpp>
#include <tamm/op_executor.hpp>
#include <tamm/op_visitors.hpp>
#include <tamm/tamm.hpp>

using namespace tamm;
using namespace tamm::new_ops;

#if 0
void test_new_op_construction() {
    using namespace std::complex_literals;
    using namespace std::string_literals;

    TiledIndexSpace AO{IndexSpace{range(10)}};
    auto[i, j, k, l] = AO.labels<4>("all");
    Tensor<double> A{i, j};
    Tensor<float> B{i, k, j};
    Tensor<int> C{i, j};
    Tensor<std::complex<double>> D{i, j, k};
    Tensor<std::complex<float>> E{i, j, k, l};

    SymbolTable symbol_table;
    TAMM_REGISTER_SYMBOLS(symbol_table, A, B, C, D, E, i, j , k, l);

    OpStringGenerator opstr{symbol_table};

    // Testing Scalar Ops
    std::complex<float> z1{2.0, -3.2};
    std::complex<double> z2{-2.0, 3.2};
    float t = 2.2;
    // std::cout << z2 * t << "\n";

    Scalar s1{2. - 3i}, s2{-3.0}, s3{2.0}, s4{z1}, s5{z2};

    std::cout << s5 * t << "\n";

    auto sc_ops = {s1, s2, s3, s4, s5};

    for(auto& sc : sc_ops) { std::cout << sc.to_string() << std::endl; }

    // Testing Scalar Ops
    LTOp lt_A{A, {i, j}};
    LTOp lt_B = B(i, k, j);
    LTOp lt_C{C(i, j)};
    LTOp lt_D = D(i, j, k);
    LTOp lt_E = E(i, j, k, l);
    LTOp lt_F = lt_E * 3.9;

    auto lt_ops = {lt_A, lt_B, lt_C, lt_D, lt_E, lt_F};
    // print tensor types for LTOps
    for(auto& lt : lt_ops) {
        std::cout << "LTOp: " << opstr.toString(lt) << std::endl;
    }

    // Testing AddOp
    // auto add_1 = s1 + s2;
    // auto add_2 = s1 + 3.0;
    // auto add_3 = 5.0 - s3;

    // auto add_4  = lt_A + s1;
    // auto add_5  = lt_A + s3;
    // auto add_6  = lt_B - 4.0;
    // auto add_7  = lt_D - s3;
    auto add_8  = lt_A * 2 + lt_C * 3.0;
    auto add_9  = lt_B + lt_D;
    auto add_10 = lt_C + lt_E;

    std::cout << opstr.toString(add_8) << std::endl;
    std::cout << opstr.toString(add_9) << std::endl;
    std::cout << opstr.toString(add_10) << std::endl;

#if 0
    auto m_adds_1 = s1 + s2 - s3 + s4;
    auto m_adds_2 = add_1 + s3;
    auto m_adds_3 = add_3 + 4.0;

    auto m_adds_4 = lt_A + lt_C - s4;
    auto m_adds_5 = lt_A + lt_C + lt_D;
    auto m_adds_6 = add_9 + lt_D;
    auto m_adds_7 = add_10 + add_5 - add_6;

    auto m_add_ops = {m_adds_1, m_adds_2, m_adds_3, m_adds_4,
                    m_adds_5, m_adds_6, m_adds_7};
    for(auto& add : m_add_ops) {
        std::cout << "Multiple AddOps: " << opstr.toString(add) << std::endl;
    }
#endif
    // Testing MultOp
    auto mult_1 = s1 * s2;
    auto mult_2 = 3.0 * s4;
    auto mult_3 = s4 * -2.0;
    auto mult_4 = lt_A * 4.0;
    auto mult_5 = 4.0 * lt_C;
    auto mult_6 = lt_A * lt_B;
    auto mult_7 = lt_E * lt_C;

    std::cout << mult_1 << std::endl;
    std::cout << mult_2 << std::endl;
    std::cout << mult_3 << std::endl;
    std::cout << opstr.toString(mult_4) << std::endl;
    std::cout << opstr.toString(mult_5) << std::endl;
    std::cout << opstr.toString(mult_6) << std::endl;

    auto m_mult_1 = s1 * s2 * s3 * s4;
    auto m_mult_2 = s1 * 4.0 * s5;
    auto m_mult_3 = mult_3 * mult_2 * 4.0;

    auto m_mult_4 = lt_A * lt_B * lt_C;
    auto m_mult_5 = lt_A * 3.0 * s1 * lt_B;
    auto m_mult_6 = mult_6 * mult_1 * 4.0;
    auto m_mult_7 = lt_E * mult_7 * s1;

    std::cout << m_mult_1 << std::endl;
    std::cout << m_mult_2 << std::endl;
    std::cout << m_mult_3 << std::endl;
    std::cout << opstr.toString(m_mult_4) << std::endl;
    std::cout << opstr.toString(m_mult_5) << std::endl;
    std::cout << opstr.toString(m_mult_6) << std::endl;
    std::cout << opstr.toString(m_mult_7) << std::endl;
}

void test_visitors() {
    TiledIndexSpace AO{IndexSpace{range(10)}};
    auto[i, j, k, l] = AO.labels<4>("all");
    Tensor<double> A{i, j};
    Tensor<double> B{j, k};
    Tensor<double> C{k, l};
    Tensor<double> D{i, k};
    // Tensor<std::complex<double>> D{i, j, k};
    // Tensor<std::complex<float>> E{i, j, k, l};

    SymbolTable symbol_table;
    TAMM_REGISTER_SYMBOLS(symbol_table, A, B, C, D,  i, j , k, l);

    LTOp lbl_A = A(i, j);
    LTOp lbl_B = B(j, k);
    LTOp lbl_C = C(k, l);
    LTOp lbl_D = D(i, l);

    auto multop  = 3.2 * lbl_A * 5.0 * (lbl_B * lbl_C) + (lbl_A + lbl_B);
    auto multop2 = lbl_A * lbl_B * lbl_C;
    OpStringGenerator opstr{symbol_table};

    AvailableLabelsVisitor alv;
    NeededLabelsVisitor nlv;
    NameVisitor nv{symbol_table};
    AllocLabelsVisitor allv;
    BinarizedPrintVisitor binv{symbol_table};

    multop.accept(alv);
    multop.accept(nlv);
    multop.accept(nv);
    multop.accept(allv);

    multop.accept(binv);

    std::cout << "Printing binarized multop" << std::endl;
    for(const auto& str :
        multop.get_attribute<BinarizedStringAttribute>().get()) {
        std::cout << str << std::endl;
    }

    D(i, l).set(multop);
    D(i, l).set(lbl_A * 5.0 * (lbl_B * lbl_C) + (lbl_A + lbl_B));

    std::cout << opstr.toString(multop) << std::endl;
}

void test_scalar_mult_operator() {
    std::complex<float> cf{2.9, 2};
    std::complex<double> cd{2.2, 7};
    int int_       = 2;
    int64_t int64_ = 2;
    float float_   = 2.0;
    double double_ = 2.0;

    std::cout << "cf * T" << std::endl;
    std::cout << cf * int_ << std::endl;
    std::cout << cf * int64_ << std::endl;
    std::cout << cf * float_ << std::endl;
    std::cout << cf * double_ << std::endl;
    std::cout << "T * cf" << std::endl;
    std::cout << int_ * cf << std::endl;
    std::cout << int64_ * cf << std::endl;
    std::cout << float_ * cf << std::endl;
    std::cout << double_ * cf << std::endl;
    std::cout << "cd * T" << std::endl;
    std::cout << cd * int_ << std::endl;
    std::cout << cd * int64_ << std::endl;
    std::cout << cd * float_ << std::endl;
    std::cout << cd * double_ << std::endl;
    std::cout << "T * cd" << std::endl;
    std::cout << int_ * cd << std::endl;
    std::cout << int64_ * cd << std::endl;
    std::cout << float_ * cd << std::endl;
    std::cout << double_ * cd << std::endl;
    std::cout << "cf * cd" << std::endl;
    std::cout << cf * cd << std::endl;
    std::cout << "cd * cf" << std::endl;
    std::cout << cd * cf << std::endl;

    Scalar scf{cf}, scd{cd}, sint_{int_}, sint64_{int64_}, sfloat_{float_},
      sdouble_{double_};

    std::cout << "scf * T" << std::endl;
    std::cout << scf * int_ << std::endl;
    std::cout << scf * int64_ << std::endl;
    std::cout << scf * float_ << std::endl;
    std::cout << scf * double_ << std::endl;
    std::cout << scf * cf << std::endl;
    std::cout << scf * cd << std::endl;
    std::cout << "T * scf" << std::endl;
    std::cout << int_ * scf << std::endl;
    std::cout << int64_ * scf << std::endl;
    std::cout << float_ * scf << std::endl;
    std::cout << double_ * scf << std::endl;
    std::cout << cf * scf << std::endl;
    std::cout << cd * scf << std::endl;
    std::cout << "scd * T" << std::endl;
    std::cout << scd * int_ << std::endl;
    std::cout << scd * int64_ << std::endl;
    std::cout << scd * float_ << std::endl;
    std::cout << scd * double_ << std::endl;
    std::cout << scd * cf << std::endl;
    std::cout << scd * cd << std::endl;
    std::cout << "T * scd" << std::endl;
    std::cout << int_ * scd << std::endl;
    std::cout << int64_ * scd << std::endl;
    std::cout << float_ * scd << std::endl;
    std::cout << double_ * scd << std::endl;
    std::cout << cf * scd << std::endl;
    std::cout << cd * scd << std::endl;

    std::cout << "scf * T" << std::endl;
    std::cout << scf * sint_ << std::endl;
    std::cout << scf * sint64_ << std::endl;
    std::cout << scf * sfloat_ << std::endl;
    std::cout << scf * sdouble_ << std::endl;
    std::cout << scf * cf << std::endl;
    std::cout << scf * cd << std::endl;
    std::cout << "T * scf" << std::endl;
    std::cout << sint_ * scf << std::endl;
    std::cout << sint64_ * scf << std::endl;
    std::cout << sfloat_ * scf << std::endl;
    std::cout << sdouble_ * scf << std::endl;
    std::cout << cf * scf << std::endl;
    std::cout << cd * scf << std::endl;
    std::cout << "scd * T" << std::endl;
    std::cout << scd * sint_ << std::endl;
    std::cout << scd * sint64_ << std::endl;
    std::cout << scd * sfloat_ << std::endl;
    std::cout << scd * sdouble_ << std::endl;
    std::cout << scd * cf << std::endl;
    std::cout << scd * cd << std::endl;
    std::cout << "T * scd" << std::endl;
    std::cout << sint_ * scd << std::endl;
    std::cout << sint64_ * scd << std::endl;
    std::cout << sfloat_ * scd << std::endl;
    std::cout << sdouble_ * scd << std::endl;
    std::cout << cf * scd << std::endl;
    std::cout << cd * scd << std::endl;
    std::cout << "scf * scd" << std::endl;
    std::cout << scf * scd << std::endl;
    std::cout << "scd * scf" << std::endl;
    std::cout << scd * scf << std::endl;
}

void test_scalar_add_operator() {
    std::complex<float> cf{2.9, 2};
    std::complex<double> cd{2.2, 7};
    int int_       = 2;
    int64_t int64_ = 2;
    float float_   = 2.0;
    double double_ = 2.0;

    std::cout << "cf + T" << std::endl;
    std::cout << cf + int_ << std::endl;
    std::cout << cf + int64_ << std::endl;
    std::cout << cf + float_ << std::endl;
    std::cout << cf + double_ << std::endl;
    std::cout << "T + cf" << std::endl;
    std::cout << int_ + cf << std::endl;
    std::cout << int64_ + cf << std::endl;
    std::cout << float_ + cf << std::endl;
    std::cout << double_ + cf << std::endl;
    std::cout << "cd + T" << std::endl;
    std::cout << cd + int_ << std::endl;
    std::cout << cd + int64_ << std::endl;
    std::cout << cd + float_ << std::endl;
    std::cout << cd + double_ << std::endl;
    std::cout << "T + cd" << std::endl;
    std::cout << int_ + cd << std::endl;
    std::cout << int64_ + cd << std::endl;
    std::cout << float_ + cd << std::endl;
    std::cout << double_ + cd << std::endl;
    std::cout << "cf + cd" << std::endl;
    std::cout << cf + cd << std::endl;
    std::cout << "cd + cf" << std::endl;
    std::cout << cd + cf << std::endl;

    Scalar scf{cf}, scd{cd}, sint_{int_}, sint64_{int64_}, sfloat_{float_},
      sdouble_{double_};

    std::cout << "scf + T" << std::endl;
    std::cout << scf + int_ << std::endl;
    std::cout << scf + int64_ << std::endl;
    std::cout << scf + float_ << std::endl;
    std::cout << scf + double_ << std::endl;
    std::cout << scf + cf << std::endl;
    std::cout << scf + cd << std::endl;
    std::cout << "T + scf" << std::endl;
    std::cout << int_ + scf << std::endl;
    std::cout << int64_ + scf << std::endl;
    std::cout << float_ + scf << std::endl;
    std::cout << double_ + scf << std::endl;
    std::cout << cf + scf << std::endl;
    std::cout << cd + scf << std::endl;
    std::cout << "scd + T" << std::endl;
    std::cout << scd + int_ << std::endl;
    std::cout << scd + int64_ << std::endl;
    std::cout << scd + float_ << std::endl;
    std::cout << scd + double_ << std::endl;
    std::cout << scd + cf << std::endl;
    std::cout << scd + cd << std::endl;
    std::cout << "T + scd" << std::endl;
    std::cout << int_ + scd << std::endl;
    std::cout << int64_ + scd << std::endl;
    std::cout << float_ + scd << std::endl;
    std::cout << double_ + scd << std::endl;
    std::cout << cf + scd << std::endl;
    std::cout << cd + scd << std::endl;

    std::cout << "scf + T" << std::endl;
    std::cout << scf + sint_ << std::endl;
    std::cout << scf + sint64_ << std::endl;
    std::cout << scf + sfloat_ << std::endl;
    std::cout << scf + sdouble_ << std::endl;
    std::cout << scf + cf << std::endl;
    std::cout << scf + cd << std::endl;
    std::cout << "T + scf" << std::endl;
    std::cout << sint_ + scf << std::endl;
    std::cout << sint64_ + scf << std::endl;
    std::cout << sfloat_ + scf << std::endl;
    std::cout << sdouble_ + scf << std::endl;
    std::cout << cf + scf << std::endl;
    std::cout << cd + scf << std::endl;
    std::cout << "scd + T" << std::endl;
    std::cout << scd + sint_ << std::endl;
    std::cout << scd + sint64_ << std::endl;
    std::cout << scd + sfloat_ << std::endl;
    std::cout << scd + sdouble_ << std::endl;
    std::cout << scd + cf << std::endl;
    std::cout << scd + cd << std::endl;
    std::cout << "T + scd" << std::endl;
    std::cout << sint_ + scd << std::endl;
    std::cout << sint64_ + scd << std::endl;
    std::cout << sfloat_ + scd << std::endl;
    std::cout << sdouble_ + scd << std::endl;
    std::cout << cf + scd << std::endl;
    std::cout << cd + scd << std::endl;
    std::cout << "scf + scd" << std::endl;
    std::cout << scf + scd << std::endl;
    std::cout << "scd + scf" << std::endl;
    std::cout << scd + scf << std::endl;
}

void test_scalar_sub_operator() {
    std::complex<float> cf{2.9, 2};
    std::complex<double> cd{2.2, 7};
    int int_       = 2;
    int64_t int64_ = 2;
    float float_   = 2.0;
    double double_ = 2.0;

    std::cout << "cf - T" << std::endl;
    std::cout << cf - int_ << std::endl;
    std::cout << cf - int64_ << std::endl;
    std::cout << cf - float_ << std::endl;
    std::cout << cf - double_ << std::endl;
    std::cout << "T - cf" << std::endl;
    std::cout << int_ - cf << std::endl;
    std::cout << int64_ - cf << std::endl;
    std::cout << float_ - cf << std::endl;
    std::cout << double_ - cf << std::endl;
    std::cout << "cd - T" << std::endl;
    std::cout << cd - int_ << std::endl;
    std::cout << cd - int64_ << std::endl;
    std::cout << cd - float_ << std::endl;
    std::cout << cd - double_ << std::endl;
    std::cout << "T - cd" << std::endl;
    std::cout << int_ - cd << std::endl;
    std::cout << int64_ - cd << std::endl;
    std::cout << float_ - cd << std::endl;
    std::cout << double_ - cd << std::endl;
    std::cout << "cf - cd" << std::endl;
    std::cout << cf - cd << std::endl;
    std::cout << "cd - cf" << std::endl;
    std::cout << cd - cf << std::endl;

    Scalar scf{cf}, scd{cd}, sint_{int_}, sint64_{int64_}, sfloat_{float_},
      sdouble_{double_};

    std::cout << "scf - T" << std::endl;
    std::cout << scf - int_ << std::endl;
    std::cout << scf - int64_ << std::endl;
    std::cout << scf - float_ << std::endl;
    std::cout << scf - double_ << std::endl;
    std::cout << scf - cf << std::endl;
    std::cout << scf - cd << std::endl;
    std::cout << "T - scf" << std::endl;
    std::cout << int_ - scf << std::endl;
    std::cout << int64_ - scf << std::endl;
    std::cout << float_ - scf << std::endl;
    std::cout << double_ - scf << std::endl;
    std::cout << cf - scf << std::endl;
    std::cout << cd - scf << std::endl;
    std::cout << "scd - T" << std::endl;
    std::cout << scd - int_ << std::endl;
    std::cout << scd - int64_ << std::endl;
    std::cout << scd - float_ << std::endl;
    std::cout << scd - double_ << std::endl;
    std::cout << scd - cf << std::endl;
    std::cout << scd - cd << std::endl;
    std::cout << "T - scd" << std::endl;
    std::cout << int_ - scd << std::endl;
    std::cout << int64_ - scd << std::endl;
    std::cout << float_ - scd << std::endl;
    std::cout << double_ - scd << std::endl;
    std::cout << cf - scd << std::endl;
    std::cout << cd - scd << std::endl;

    std::cout << "scf - T" << std::endl;
    std::cout << scf - sint_ << std::endl;
    std::cout << scf - sint64_ << std::endl;
    std::cout << scf - sfloat_ << std::endl;
    std::cout << scf - sdouble_ << std::endl;
    std::cout << scf - cf << std::endl;
    std::cout << scf - cd << std::endl;
    std::cout << "T - scf" << std::endl;
    std::cout << sint_ - scf << std::endl;
    std::cout << sint64_ - scf << std::endl;
    std::cout << sfloat_ - scf << std::endl;
    std::cout << sdouble_ - scf << std::endl;
    std::cout << cf - scf << std::endl;
    std::cout << cd - scf << std::endl;
    std::cout << "scd - T" << std::endl;
    std::cout << scd - sint_ << std::endl;
    std::cout << scd - sint64_ << std::endl;
    std::cout << scd - sfloat_ << std::endl;
    std::cout << scd - sdouble_ << std::endl;
    std::cout << scd - cf << std::endl;
    std::cout << scd - cd << std::endl;
    std::cout << "T - scd" << std::endl;
    std::cout << sint_ - scd << std::endl;
    std::cout << sint64_ - scd << std::endl;
    std::cout << sfloat_ - scd << std::endl;
    std::cout << sdouble_ - scd << std::endl;
    std::cout << cf - scd << std::endl;
    std::cout << cd - scd << std::endl;
    std::cout << "scf - scd" << std::endl;
    std::cout << scf - scd << std::endl;
    std::cout << "scd - scf" << std::endl;
    std::cout << scd - scf << std::endl;
}

void test_scalar_div_operator() {
    std::complex<float> cf{2.9, 2};
    std::complex<double> cd{2.2, 7};
    int int_       = 2;
    int64_t int64_ = 2;
    float float_   = 2.0;
    double double_ = 2.0;

    std::cout << "cf / T" << std::endl;
    std::cout << cf / int_ << std::endl;
    std::cout << cf / int64_ << std::endl;
    std::cout << cf / float_ << std::endl;
    std::cout << cf / double_ << std::endl;
    std::cout << "T / cf" << std::endl;
    std::cout << int_ / cf << std::endl;
    std::cout << int64_ / cf << std::endl;
    std::cout << float_ / cf << std::endl;
    std::cout << double_ / cf << std::endl;
    std::cout << "cd / T" << std::endl;
    std::cout << cd / int_ << std::endl;
    std::cout << cd / int64_ << std::endl;
    std::cout << cd / float_ << std::endl;
    std::cout << cd / double_ << std::endl;
    std::cout << "T / cd" << std::endl;
    std::cout << int_ / cd << std::endl;
    std::cout << int64_ / cd << std::endl;
    std::cout << float_ / cd << std::endl;
    std::cout << double_ / cd << std::endl;
    std::cout << "cf / cd" << std::endl;
    std::cout << cf / cd << std::endl;
    std::cout << "cd / cf" << std::endl;
    std::cout << cd / cf << std::endl;

    Scalar scf{cf}, scd{cd}, sint_{int_}, sint64_{int64_}, sfloat_{float_},
      sdouble_{double_};

    std::cout << "scf / T" << std::endl;
    std::cout << scf / int_ << std::endl;
    std::cout << scf / int64_ << std::endl;
    std::cout << scf / float_ << std::endl;
    std::cout << scf / double_ << std::endl;
    std::cout << scf / cf << std::endl;
    std::cout << scf / cd << std::endl;
    std::cout << "T / scf" << std::endl;
    std::cout << int_ / scf << std::endl;
    std::cout << int64_ / scf << std::endl;
    std::cout << float_ / scf << std::endl;
    std::cout << double_ / scf << std::endl;
    std::cout << cf / scf << std::endl;
    std::cout << cd / scf << std::endl;
    std::cout << "scd / T" << std::endl;
    std::cout << scd / int_ << std::endl;
    std::cout << scd / int64_ << std::endl;
    std::cout << scd / float_ << std::endl;
    std::cout << scd / double_ << std::endl;
    std::cout << scd / cf << std::endl;
    std::cout << scd / cd << std::endl;
    std::cout << "T / scd" << std::endl;
    std::cout << int_ / scd << std::endl;
    std::cout << int64_ / scd << std::endl;
    std::cout << float_ / scd << std::endl;
    std::cout << double_ / scd << std::endl;
    std::cout << cf / scd << std::endl;
    std::cout << cd / scd << std::endl;

    std::cout << "scf / T" << std::endl;
    std::cout << scf / sint_ << std::endl;
    std::cout << scf / sint64_ << std::endl;
    std::cout << scf / sfloat_ << std::endl;
    std::cout << scf / sdouble_ << std::endl;
    std::cout << scf / cf << std::endl;
    std::cout << scf / cd << std::endl;
    std::cout << "T / scf" << std::endl;
    std::cout << sint_ / scf << std::endl;
    std::cout << sint64_ / scf << std::endl;
    std::cout << sfloat_ / scf << std::endl;
    std::cout << sdouble_ / scf << std::endl;
    std::cout << cf / scf << std::endl;
    std::cout << cd / scf << std::endl;
    std::cout << "scd / T" << std::endl;
    std::cout << scd / sint_ << std::endl;
    std::cout << scd / sint64_ << std::endl;
    std::cout << scd / sfloat_ << std::endl;
    std::cout << scd / sdouble_ << std::endl;
    std::cout << scd / cf << std::endl;
    std::cout << scd / cd << std::endl;
    std::cout << "T / scd" << std::endl;
    std::cout << sint_ / scd << std::endl;
    std::cout << sint64_ / scd << std::endl;
    std::cout << sfloat_ / scd << std::endl;
    std::cout << sdouble_ / scd << std::endl;
    std::cout << cf / scd << std::endl;
    std::cout << cd / scd << std::endl;
    std::cout << "scf / scd" << std::endl;
    std::cout << scf / scd << std::endl;
    std::cout << "scd / scf" << std::endl;
    std::cout << scd / scf << std::endl;
}

void test_block_set() {
    std::complex<double> c{5};
    std::complex<float> cf6{6.1};
    int i3 = 3;
    Scalar v3{i3}, v4{1.4}, v5{c}, v6{cf6};

    int buf[3][2], buf2[3][2];
    BlockSpan<int> dbs(&buf[0][0], {3, 2});
    BlockSpan<int> dbs2(&buf2[0][0], {3, 2});
    blockops::cpu::set(dbs, 4.0);

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 2; j++) {
        std::cout << buf[i][j] << "\n";
      }
    }

    BlockSetPlan{BlockSetPlan::OpType::set}.apply(dbs, Scalar{2});

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 2; j++) {
        std::cout << buf[i][j] << "\n";
      }
    }
    

    // blockops::cpu::set(dbs, Scalar{int(3)});
    // blockops::cpu::update(dbs, Scalar{int(3)});
    // blockops::cpu::update(4.5, dbs, Scalar{i3});
    // blockops::cpu::flat_assign(dbs, Scalar{1.0}, dbs2);
}

void test_binarized_execution(){
  TiledIndexSpace AO{IndexSpace{range(10)}};
  auto[i, j, k, l, a, b] = AO.labels<6>("all");
  Tensor<double> A{i, j};
  Tensor<double> B{j, k};
  Tensor<double> C{k, l};
  Tensor<double> D{i, l};
  
  // Tensor<std::complex<double>> D{i, j, k};
  // Tensor<std::complex<float>> E{i, j, k, l};
  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, A, B, C, D, i, j, k, l, a, b);



  ProcGroup pg = ProcGroup::create_world_coll();
  auto mgr     = MemoryManagerGA::create_coll(pg);
  Distribution_NW distribution;
  ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
  Scheduler sch{*ec};

  sch.allocate(A, B, C, D)
  (A() = 1.0)
  (B() = 2.0)
  (C() = 3.0)
  (D() = 0.0)
  .execute();
  std::cout << "Printing Tensor D before new op execute" << "\n";
  print_tensor(D);

  LTOp lbl_A = A(i, j);
  LTOp lbl_B = B(j, k);
  LTOp lbl_C = C(k, l);
  LTOp lbl_D = D(i, l);

  auto multop = lbl_A * 4.0 * lbl_B * 8.2 * lbl_C;
  auto multop2 = lbl_A * 5.0 * (lbl_B * lbl_C) + (3.0 * lbl_A + lbl_B * 4.2);

  D(i, l).set(multop);
  D(i, k).update(lbl_A * lbl_B);
  D(i, l).update(multop2);


  OpExecutor op_executor{sch, symbol_table};
  // op_executor.print_binarized(D);
  op_executor.pretty_print_binarized(D);
  op_executor.execute(D);

  std::cout << "Printing Tensor D after new op execute" << "\n";
  print_tensor(D);

  sch.deallocate(A, B, C, D);

}

void new_ops_ccsd_e() {
  using T = double;

  IndexSpace MO_IS{range(0, 20),
                   {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
  TiledIndexSpace MO{MO_IS, 1};

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  Tensor<T> de{};
  Tensor<T> t1{V,O};
  Tensor<T> f1{O, V};
  Tensor<T> v2{O, O, V, V};
  Tensor<T> t2{V, V, O, O};

  TiledIndexLabel p1, p2, p3, p4, p5;
  TiledIndexLabel h3, h4, h5, h6;

  std::tie(p1, p2, p3, p4, p5) = V.labels<5>("all");
  std::tie(h3, h4, h5, h6)     = O.labels<4>("all");

  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, de, t1, f1, v2, t2, p1, p2, p3, p4, p5, h3, h4, h5, h6);

  LTOp t1_p5_h6       = t1(p5, h6);
  LTOp f1_h6_p5       = f1(h6, p5);
  LTOp t1_p3_h4       = t1(p3, h4);
  LTOp v2_h4_h6_p3_p5 = v2(h4, h6, p3, p5);
  LTOp t2_p1_p2_h3_h4 = t2(p1, p2, h3, h4);
  LTOp v2_h3_h4_p1_p2 = v2(h3, h4, p1, p2);

  ProcGroup pg = ProcGroup::create_world_coll();
  auto mgr     = MemoryManagerGA::create_coll(pg);
  Distribution_NW distribution;
  ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
  Scheduler sch{*ec};
  
  BlockSetPlan plan{IndexLabelVec{p1, p2}, BlockSetPlan::OpType::set};

  sch.allocate(de, t1, f1, v2, t2)
  (t1() = 1.0)
  (f1() = 2.0)
  (v2() = 3.0)
  (t2() = 4.0)
  .execute();

  de().set((t1_p5_h6 * (f1_h6_p5 + (0.5 * t1_p3_h4 * v2_h4_h6_p3_p5))) +
           (0.25 * t2_p1_p2_h3_h4 * v2_h3_h4_p1_p2));

  OpExecutor op_executor{sch, symbol_table};
  op_executor.execute(de);

  std::cout << "Printing Tensor de after new op execute" << "\n";
  std::cout << get_scalar(de) << "\n";

  sch.deallocate(de, t1, f1, v2, t2);

}
#endif

template<typename T>
void lambda_function(const IndexVector& blockid, span<T> buff) {
  for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
}

void test_new_ops() {
  using T                  = double;
  size_t          aux_size = 5;
  IndexSpace      MO_IS{range(0, 20), {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
  TiledIndexSpace MO{MO_IS, {2, 3, 2, 3, 2, 3, 2, 3}};
  TiledIndexSpace AUX{IndexSpace{range(aux_size)}};

  ProcGroup         pg  = ProcGroup::create_world_coll();
  auto              mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW   distribution;
  ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
  Scheduler         sch{*ec};

  auto [x, y] = MO.labels<2>("all");
  auto [i, j] = MO.labels<2>("occ");
  auto [a, b] = MO.labels<2>("virt");
  auto [K, L] = AUX.labels<2>("all");
  Tensor<T> lambdaT{{MO, MO}, lambda_function<T>};

  Tensor<T>               A{x, y};
  Tensor<std::complex<T>> B{x, y};
  Tensor<std::complex<T>> C{x, y};
  Tensor<std::complex<T>> D{x, y, K};

  std::complex<T> cnst{-10.1, 4.0};
  std::complex<T> cnst2{1.1, 3.0};

  sch.allocate(A, B, C, D).execute();

  sch(A() = 1.0)(B() = cnst)(C() = cnst2).execute();
  print_tensor(A);
  print_tensor(B);
  print_tensor(C);

  sch(C(x, y) = B(x, y))(A(x, y) = B(x, y))(B(x, y) = A(x, y)).execute();

  print_tensor(A);
  print_tensor(B);
  print_tensor(C);

  sch(A() = 1.0)(B() = cnst)(C() = cnst2).execute();

  sch(C(x, y) += B(x, y))(A(x, y) += B(x, y))(B(x, y) += A(x, y)).execute();

  print_tensor(A);
  print_tensor(B);
  print_tensor(C);

  sch(A() = 1.0)(B() = cnst)(C() = cnst2).execute();

  sch(C(x, y) -= B(x, y))(A(x, y) -= B(x, y))(B(x, y) -= A(x, y)).execute();

  print_tensor(A);
  print_tensor(B);
  print_tensor(C);

  sch(A() = 1.0)(B() = cnst)(C() = cnst2)(D() = 0.0).execute();

  print_tensor(A);
  print_tensor(B);
  print_tensor(C);
  std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
  for(size_t i = 0; i < aux_size; i++) {
    TiledIndexSpace tsc{AUX, range(i, i + 1)};
    auto [sc] = tsc.labels<1>("all");
    sch(D(x, y, sc) = C(x, y)).execute();
    print_tensor(D);
  }

  // sch
  // (A(x,x) = 10.0)
  // // (A(x, y) += -0.1 * B(x, y))
  // // (A(x, y) -= B(y, x))
  // // (A(x, y) += B(y, x))
  // .execute();
}

void test_utility_methods() {
  std::vector<int> vec1{0, 2, 5, 4, 3};
  std::vector<int> vec2{4, 1, 6, 8};
  std::vector<int> vec3{9, 0, 2, 3, 7};

  auto merge_vec = internal::merge_vector<std::vector<int>>(vec1, vec2, vec3);

  EXPECTS((merge_vec == std::vector<int>{0, 2, 5, 4, 3, 4, 1, 6, 8, 9, 0, 2, 3, 7}));
  auto [new_vec1, new_vec2, new_vec3] =
    internal::split_vector<std::vector<int>, 3>(merge_vec, {5, 4, 5});
  EXPECTS(vec1 == new_vec1);
  EXPECTS(vec2 == new_vec2);
  EXPECTS(vec3 == new_vec3);
}

void test_gfcc_failed_case() {
  using T                  = double;
  size_t          aux_size = 5;
  IndexSpace      MO_IS{range(0, 20), {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
  TiledIndexSpace MO{MO_IS, {2, 3, 2, 3, 2, 3, 2, 3}};
  TiledIndexSpace AUX{IndexSpace{range(aux_size)}};

  ProcGroup         pg  = ProcGroup::create_world_coll();
  auto              mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW   distribution;
  ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
  Scheduler         sch{*ec};

  int nsranks = /* sys_data.nbf */ 45 / 15;
  int ga_cnn  = ec->nnodes();
  if(nsranks > ga_cnn) nsranks = ga_cnn;
  nsranks = nsranks * ec->ppn();
  int subranks[nsranks];
  for(int i = 0; i < nsranks; i++) subranks[i] = i;
  auto      world_comm = ec->pg().comm();
  MPI_Group world_group;
  MPI_Comm_group(world_comm, &world_group);
  MPI_Group subgroup;
  MPI_Group_incl(world_group, nsranks, subranks, &subgroup);
  MPI_Comm subcomm;
  MPI_Comm_create(world_comm, subgroup, &subcomm);

  MPI_Group_free(&world_group);
  MPI_Group_free(&subgroup);

  ProcGroup         sub_pg           = ProcGroup::create_coll(subcomm);
  MemoryManagerGA*  sub_mgr          = MemoryManagerGA::create_coll(sub_pg);
  Distribution_NW*  sub_distribution = new Distribution_NW();
  RuntimeEngine*    sub_re           = new RuntimeEngine();
  ExecutionContext* sub_ec = new ExecutionContext(sub_pg, sub_distribution, sub_mgr, sub_re);

  Scheduler sub_sch{*sub_ec};

  auto [x, y] = MO.labels<2>("all");
  auto [i, j] = MO.labels<2>("occ");
  auto [a, b] = MO.labels<2>("virt");
  auto [K, L] = AUX.labels<2>("all");
  Tensor<T> lambdaT{{MO, MO}, lambda_function<T>};

  Tensor<T>               A{x, y};
  Tensor<std::complex<T>> B{x, y};
  Tensor<std::complex<T>> C{x, y};
  Tensor<std::complex<T>> D{x, y, K};

  std::complex<T> cnst{-10.1, 4.0};
  std::complex<T> cnst2{1.1, 3.0};

  sch.allocate(A, B, C, D)(A() = 1.0)(B() = cnst)(C() = cnst2)(D() = 0.0).execute();
  std::cout << "Execute on sub_sch"
            << "\n";
  for(size_t i = 0; i < aux_size; i++) {
    TiledIndexSpace tsc{AUX, range(i, i + 1)};
    auto [sc] = tsc.labels<1>("all");
    sub_sch(D(x, y, sc) = C(x, y)).execute();
  }
  std::cout << "Printing Tensor D"
            << "\n";
  print_tensor_all(D);

  std::cout << "Execute on sch"
            << "\n";
  for(size_t i = 0; i < aux_size; i++) {
    TiledIndexSpace tsc{AUX, range(i, i + 1)};
    auto [sc] = tsc.labels<1>("all");
    sch(D(x, y, sc) = C(x, y)).execute();
  }
  std::cout << "Printing Tensor D"
            << "\n";
  print_tensor_all(D);
}

template<typename T>
void cs_ccsd_t1() {
  IndexSpace IS{range(10), {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};

  TiledIndexSpace MO{IS};

  auto [i, j, m, n] = MO.labels<4>("occ");
  auto [a, e, f]    = MO.labels<3>("virt");

  Tensor<T> i0{a, i};
  Tensor<T> t1{a, m};
  Tensor<T> t2{a, e, m, n};
  Tensor<T> F{MO, MO};
  Tensor<T> V{MO, MO, MO, MO};

  SymbolTable symbol_table;
  TAMM_REGISTER_SYMBOLS(symbol_table, F, V, t1, t2, i0);
  TAMM_REGISTER_SYMBOLS(symbol_table, i, j, m, n, a, e, f);

  auto singles =
    (LTOp) F(a, i) + (-2.0 * (LTOp) F(m, e) * (LTOp) t1(a, m) * (LTOp) t1(e, i)) +
    ((LTOp) F(a, e) * (LTOp) t1(e, i)) +
    (-2.0 * (LTOp) V(m, n, e, f) * (LTOp) t2(a, f, m, n) * (LTOp) t1(e, i)) +
    (-2.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(a, m) * (LTOp) t1(f, n) * (LTOp) t1(e, i)) +
    ((LTOp) V(n, m, e, f) * (LTOp) t2(a, f, m, n) * (LTOp) t1(e, i)) +
    ((LTOp) V(n, m, e, f) * (LTOp) t1(a, m) * (LTOp) t1(f, n) * (LTOp) t1(e, i)) +
    (-1.0 * (LTOp) F(m, i) * (LTOp) t1(a, m)) +
    (-2.0 * (LTOp) V(m, n, e, f) * (LTOp) t2(e, f, i, n) * (LTOp) t1(a, m)) +
    (-2.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(e, i) * (LTOp) t1(f, n) * (LTOp) t1(a, m)) +
    ((LTOp) V(m, n, f, e) * (LTOp) t2(e, f, i, n) * (LTOp) t1(a, m)) +
    ((LTOp) V(m, n, f, e) * (LTOp) t1(e, i) * (LTOp) t1(f, n) * (LTOp) t1(a, m)) +
    (2.0 * (LTOp) F(m, e) * (LTOp) t2(e, a, m, i)) +
    (-1.0 * (LTOp) F(m, e) * (LTOp) t2(e, a, i, m)) +
    ((LTOp) F(m, e) * (LTOp) t1(e, i) * (LTOp) t1(a, m)) +
    (+4.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(f, n) * (LTOp) t2(e, a, m, i)) +
    (-2.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(f, n) * (LTOp) t2(e, a, i, m)) +
    (2.0 * (LTOp) V(m, n, e, f) * (LTOp) t1(f, n) * (LTOp) t1(e, i) * (LTOp) t1(a, m)) +
    (-2.0 * (LTOp) V(m, n, f, e) * (LTOp) t1(f, n) * (LTOp) t2(e, a, m, i)) +
    ((LTOp) V(m, n, f, e) * (LTOp) t1(f, n) * (LTOp) t2(e, a, i, m)) +
    (-1.0 * (LTOp) V(m, n, f, e) * (LTOp) t1(f, n) * (LTOp) t1(e, i) * (LTOp) t1(a, m)) +
    (2.0 * (LTOp) V(m, a, e, i) * (LTOp) t1(e, m)) +
    (-1.0 * (LTOp) V(m, a, i, e) * (LTOp) t1(e, m)) +
    (2.0 * (LTOp) V(m, a, e, f) * (LTOp) t2(e, f, m, i)) +
    (2.0 * (LTOp) V(m, a, e, f) * (LTOp) t1(e, m) * (LTOp) t1(f, i)) +
    (-1.0 * (LTOp) V(m, a, f, e) * (LTOp) t2(e, f, m, i)) +
    (-1.0 * (LTOp) V(m, a, f, e) * (LTOp) t1(e, m) * (LTOp) t1(f, i)) +
    (-2.0 * (LTOp) V(m, n, e, i) * (LTOp) t2(e, a, m, n)) +
    (-2.0 * (LTOp) V(m, n, e, i) * (LTOp) t1(e, m) * (LTOp) t1(a, n)) +
    ((LTOp) V(n, m, e, i) * (LTOp) t2(e, a, m, n)) +
    ((LTOp) V(n, m, e, i) * (LTOp) t1(e, m) * (LTOp) t1(a, n));

  OpStringGenerator str_generator{symbol_table};
  auto              op_str = str_generator.toString(singles);

  UsedTensorInfoVisitor tensor_info{symbol_table};
  singles.accept(tensor_info);

  std::cout << "op: \n" << op_str << "\n";
  SeparateSumOpsVisitor sum_visitor;

  auto sum_ops = sum_visitor.sum_vectors(singles);
  std::cout << "total mult ops : " << sum_ops.size() << "\n";
  for(auto& var: sum_ops) {
    std::cout << "op: \n" << str_generator.toString(*var) << "\n";
    auto tensors = var->get_attribute<UsedTensorInfoAttribute>().get();
    for(auto tensor: tensors) { std::cout << tensor.to_string() << "\n"; }
  }
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  // test_scalar_add_operator();
  // test_scalar_sub_operator();
  // test_scalar_mult_operator();
  // test_scalar_div_operator();

  // test_new_op_construction();
  // test_visitors();

  // test_block_set();

  // test_binarized_execution();
  // new_ops_ccsd_e();
  // test_new_ops();
  // test_utility_methods();
  // test_gfcc_failed_case();
  cs_ccsd_t1<double>();

  tamm::finalize();
  return 0;
}