#include <chrono>

#include <tamm/op_executor.hpp>
#include <tamm/opmin.hpp>
#include <tamm/tamm.hpp>

using namespace tamm;
using namespace tamm::new_ops;

template<typename T>
void cs_ccsd_t1(Scheduler& sch) {
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

  sch.allocate(F, V, t1, t2, i0);
  sch(F() = 1.0)(V() = 1.0)(t1() = 1.0)(t2() = 1.0)(i0() = 1.0).execute();

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

  i0(a, i).update(singles);

  OpExecutor op_exec{sch, symbol_table};
  // op_exec.pretty_print_binarized(i0);
  op_exec.opmin_execute(i0);
  print_tensor_all(i0);

  sch(i0() = 1.0).execute();
  i0(a, i).update(singles);
  op_exec.execute(i0);
  print_tensor_all(i0);

  // OpStringGenerator str_generator{symbol_table};
  // auto op_str = str_generator.toString(singles);

  // LTOp new_ltop{i0(a, i)};
  // TensorInfo new_lhs{symbol_table[i0.get_symbol_ptr()],
  //                    new_ltop.tensor(),
  //                    new_ltop.labels(),
  //                    new_ltop.tensor_type(),
  //                    new_ltop.coeff(),
  //                    false};

  // UsedTensorInfoVisitor tensor_info{symbol_table};
  // singles.accept(tensor_info);

  // std::cout << "op: \n" << op_str << "\n";
  // SeparateSumOpsVisitor sum_visitor;
  // std::map<TiledIndexLabel, std::string> label_names;
  // for(auto lbl : {i, j, m, n, a, e, f}) {
  //   label_names[lbl] = symbol_table[lbl.get_symbol_ptr()];
  // }

  // auto sum_ops = sum_visitor.sum_vectors(singles);
  // std::cout << "total mult ops : " << sum_ops.size() << "\n";
  // for(auto& op : sum_ops) {
  //   std::cout << "op: \n" << str_generator.toString(*op) << "\n";
  //   auto tensors = op->get_attribute<UsedTensorInfoAttribute>().get();
  //   opmin::OpStmt stmt{new_lhs, opmin::OpType::plus_equal, op->coeff(), tensors};
  //   opmin::Optimizer optimizer{stmt, label_names};
  //   auto optimized_op = optimizer.optimize();
  //   std::cout << "optimized_op: \n" << str_generator.toString(*optimized_op) << "\n";
  // }
}

template<typename T>
void dlpno_test(Scheduler& sch) {
  IndexSpace      IS{range(13), {{"occ", {range(0, 4)}}, {"virt", {range(4, 13)}}}};
  TiledIndexSpace AO{IndexSpace{range(13)}};
  TiledIndexSpace AO_DF{IndexSpace{range(375)}};
  TiledIndexSpace MO{IS};
  TiledIndexSpace PAO = AO("all");
  TiledIndexSpace PNO = MO("virt");
  TiledIndexSpace LMOP{IndexSpace{range(16)}};

  Tensor<T> dTEvv{PAO, PAO, AO_DF};
  Tensor<T> dTEov_00{LMOP, PAO, AO_DF};
  Tensor<T> dT1{PNO, LMOP};
  Tensor<T> dT2{PNO, PNO, LMOP};
  Tensor<T> dT2_out{PNO, PNO, LMOP};

  Tensor<T> d{LMOP, PAO, PNO};
  Tensor<T> Siikl{LMOP, LMOP, PNO, PNO};

  SymbolTable symbol_table;

  TAMM_REGISTER_SYMBOLS(symbol_table, dTEvv, dTEov_00, dT1, dT2, dT2_out, d, Siikl);

  sch.allocate(dTEvv, dTEov_00, dT1, dT2, dT2_out, d, Siikl);
  sch(dTEvv() = 1.0)(dTEov_00() = 1.0)(dT1() = 1.0)(dT2() = 1.0)(dT2_out() =
                                                                   0.0)(d() = 1.0)(Siikl() = 1.0)
    .execute();

  auto dlpno_doubles_12 =
    (-1.0 * (LTOp) dTEvv("a_mu", "e_mu", "K") * (LTOp) dTEov_00("mm", "f_mu", "K") *
     (LTOp) dT1("b_mm", "mm") * (LTOp) dT2("e_ij", "f_ij", "ij") * (LTOp) d("ij", "f_mu", "f_ij") *
     (LTOp) d("ij", "e_mu", "e_ij") * (LTOp) d("ij", "a_mu", "a_ij") *
     (LTOp) Siikl("mm", "ij", "b_mm", "b_ij"));

  OpCostCalculator op_cost{symbol_table};

  LTOp lhs_ltop = (LTOp) dT2_out("a_ij", "b_ij", "ij");

  std::cout << "Print original binarized op\n";
  op_cost.print_op_binarized(lhs_ltop, dlpno_doubles_12.clone(), std::cout);
  auto original_op_cost =
    op_cost.get_op_cost(dlpno_doubles_12.clone(), dT2_out("a_ij", "b_ij", "ij"));
  std::cout << "Original op cost: " << original_op_cost << "\n";

  OpMin opmin{symbol_table};
  auto  optimized_dlpno_doubles_12 = opmin.optimize_all(lhs_ltop, dlpno_doubles_12);

  std::cout << "Print opmined binarized op\n";
  op_cost.print_op_binarized((LTOp) dT2_out("a_ij", "b_ij", "ij"), optimized_dlpno_doubles_12, std::cout);
  auto opmined_op_cost =
    op_cost.get_op_cost(optimized_dlpno_doubles_12->clone(), dT2_out("a_ij", "b_ij", "ij"));
  std::cout << "Opmined op cost: " << opmined_op_cost << "\n";
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec};

  // cs_ccsd_t1<double>(sch);
  dlpno_test<double>(sch);

  tamm::finalize();

  return 0;
}
