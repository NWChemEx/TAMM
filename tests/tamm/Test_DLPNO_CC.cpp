#include <filesystem>
#include <nlohmann/json.hpp>

#include <tamm/op_executor.hpp>
#include <tamm/tamm.hpp>

using namespace tamm;
using namespace tamm::new_ops;

namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

template<typename T>
void dlpno_test(const json& params) {
  // Get dimension sizes and tile sizes from input JSON
  auto   dim_sizes = params.at("dim_sizes");
  size_t ao_size   = dim_sizes.at("AO");
  size_t mo_size   = dim_sizes.at("MO");
  size_t occ_size  = dim_sizes.at("MO_occ");
  size_t virt_size = dim_sizes.at("MO_virt");
  if((occ_size + virt_size) != mo_size) {
    std::ostringstream os;
    os << "[TAMM ERROR] MO(" << mo_size << ") size should be equal "
       << "to Virt(" << virt_size << ") + Occ(" << occ_size << ")!\n"
       << __FILE__ << ":L" << __LINE__;
    tamm_terminate(os.str());
  }

  size_t df_size   = dim_sizes.at("DF");
  size_t lmop_size = dim_sizes.at("LMOP");
  size_t pao_size  = dim_sizes.at("PAO");
  size_t pno_size  = dim_sizes.at("PNO");

  tamm::Tile tile_size = params.at("tile_size");

  bool use_opmin        = params.at("use_opmin");
  bool do_profile       = params.at("do_profile");
  auto ex_hw            = params.at("use_gpu") ? ExecutionHW::GPU : ExecutionHW::CPU;
  bool print_debug_info = params.at("print_debug_info");

  // Initialize TiledIndexSpaces
  TiledIndexSpace AO{IndexSpace{range(ao_size)}, tile_size};
  TiledIndexSpace MO{
    IndexSpace{range(mo_size), {{"occ", {range(occ_size)}}, {"virt", {range(occ_size, mo_size)}}}},
    tile_size};
  TiledIndexSpace DF{IndexSpace{range(df_size)}, tile_size};
  TiledIndexSpace LMOP{IndexSpace{range(lmop_size)}, tile_size};
  TiledIndexSpace PAO{IndexSpace{range(pao_size)}, tile_size};
  TiledIndexSpace PNO{IndexSpace{range(pno_size)}, tile_size};

  // Constructing EC and Scheduler
  ProcGroup         pg  = ProcGroup::create_world_coll();
  auto              mgr = MemoryManagerGA::create_coll(pg);
  Distribution_NW   distribution;
  ExecutionContext* ec   = new ExecutionContext{pg, &distribution, mgr};
  auto              rank = pg.rank();
  Scheduler         sch{*ec};

  // Constructs for new execution
  SymbolTable symbol_table;
  OpExecutor  op_executor{sch, symbol_table};

  // Amplitude tensors
  Tensor<T> dT1{PNO, LMOP};
  Tensor<T> dT2{PNO, PNO, LMOP};
  Tensor<T> dr1{PNO, LMOP};
  Tensor<T> dr2{PNO, PNO, LMOP};

  // 3 Center Integrals
  Tensor<T> dTEoo{LMOP, LMOP, DF};
  Tensor<T> dTEov{LMOP, PAO, DF};
  Tensor<T> dTEvv{PAO, PAO, DF};

  // Transformation tensors
  Tensor<T> d{LMOP, PAO, PNO};
  Tensor<T> Sijkl{LMOP, LMOP, PNO, PNO};

  // Expand O to LMOP
  Tensor<T> expand{MO("occ"), LMOP};

  // Register tensor names to the symbol table (required for printing operations)
  TAMM_REGISTER_SYMBOLS(symbol_table, dT1, dT2, dr1, dr2, dTEoo, dTEov, dTEvv, d, Sijkl, expand);

  // clang-format off
  // Allocate all tensors
  sch.allocate(dT1, dT2, dr1, dr2, dTEoo, dTEov, dTEvv, d, Sijkl, expand)
  (dr1() = 0.0)
  (dr2() = 0.0)
    .execute();
  // clang-format on
  if(rank == 0)
    std::cout << "Allocated all tensors."
              << "\n";

  // Initialize tensors with random values
  random_ip(dT1);
  random_ip(dT2);
  random_ip(dTEoo);
  random_ip(dTEov);
  random_ip(dTEvv);
  random_ip(d);
  random_ip(Sijkl);
  random_ip(expand);

  if(rank == 0)
    std::cout << "Initialized all input tensors with random values."
              << "\n";

  // most expensive doubles
  auto doubles_5 = (LTOp) dTEov("mn", "e_mu", "K") * (LTOp) dTEov("mn", "f_mu", "K") *
                   (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_jj", "jj") *
                   (LTOp) dT2("a_mn", "b_mn", "mn") * (LTOp) d("ii", "e_mu", "e_ii") *
                   (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("mn", "ij", "a_mn", "a_ij") *
                   (LTOp) Sijkl("mn", "ij", "b_mn", "b_ij") * (LTOp) expand("j", "jj") *
                   (LTOp) expand("i", "ii") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "ij");

  auto doubles_10 = (LTOp) dTEov("mm", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_jj", "jj") * (LTOp) dT1("a_mm", "mm") *
                    (LTOp) dT1("b_nn", "nn") * (LTOp) d("ii", "e_mu", "e_ii") *
                    (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("nn", "ij", "b_nn", "b_ij") *
                    (LTOp) Sijkl("mm", "ij", "a_mm", "a_ij") * (LTOp) expand("j", "jj") *
                    (LTOp) expand("i", "ii") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "ij");

  auto doubles_15 = -1.0 * (LTOp) dTEvv("a_mu", "e_mu", "K") * (LTOp) dTEov("mm", "f_mu", "K") *
                    (LTOp) dT1("b_mm", "mm") * (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_jj", "jj") *
                    (LTOp) d("ii", "e_mu", "e_ii") * (LTOp) d("jj", "f_mu", "f_jj") *
                    (LTOp) d("ij", "a_mu", "a_ij") * (LTOp) Sijkl("mm", "ij", "b_mm", "b_ij") *
                    (LTOp) expand("j", "jj") * (LTOp) expand("i", "ii") * (LTOp) expand("i", "ij") *
                    (LTOp) expand("j", "ij");

  auto doubles_16 = -1.0 * (LTOp) dTEov("mm", "e_mu", "K") * (LTOp) dTEvv("b_mu", "f_mu", "K") *
                    (LTOp) dT1("a_mm", "mm") * (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_jj", "jj") *
                    (LTOp) d("ii", "e_mu", "e_ii") * (LTOp) d("jj", "f_mu", "f_jj") *
                    (LTOp) d("ij", "b_mu", "b_ij") * (LTOp) Sijkl("mm", "ij", "a_mm", "a_ij") *
                    (LTOp) expand("j", "jj") * (LTOp) expand("i", "ii") * (LTOp) expand("i", "ij") *
                    (LTOp) expand("j", "ij");

  auto doubles_37 = -2.0 * (LTOp) dTEov("mj", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_nn", "nn") *
                    (LTOp) dT2("a_mj", "b_mj", "mj") * (LTOp) d("ii", "e_mu", "e_ii") *
                    (LTOp) d("nn", "f_mu", "f_nn") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                    (LTOp) Sijkl("mj", "ij", "a_mj", "a_ij") * (LTOp) expand("i", "ii") *
                    (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_38 = -2.0 * (LTOp) dTEov("mi", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("e_jj", "jj") * (LTOp) dT1("f_nn", "nn") *
                    (LTOp) dT2("b_mi", "a_mi", "mi") * (LTOp) d("jj", "e_mu", "e_jj") *
                    (LTOp) d("nn", "f_mu", "f_nn") * (LTOp) Sijkl("mi", "ij", "b_mi", "b_ij") *
                    (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") * (LTOp) expand("j", "ij") *
                    (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  auto doubles_41 = (LTOp) dTEov("mj", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                    (LTOp) dT1("e_ii", "ii") * (LTOp) dT1("f_nn", "nn") *
                    (LTOp) dT2("a_mj", "b_mj", "mj") * (LTOp) d("ii", "e_mu", "e_ii") *
                    (LTOp) d("nn", "f_mu", "f_nn") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                    (LTOp) Sijkl("mj", "ij", "a_mj", "a_ij") * (LTOp) expand("i", "ii") *
                    (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_42 = (LTOp) dTEov("mi", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                    (LTOp) dT1("e_jj", "jj") * (LTOp) dT1("f_nn", "nn") *
                    (LTOp) dT2("b_mi", "a_mi", "mi") * (LTOp) d("jj", "e_mu", "e_jj") *
                    (LTOp) d("nn", "f_mu", "f_nn") * (LTOp) Sijkl("mi", "ij", "b_mi", "b_ij") *
                    (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") * (LTOp) expand("j", "ij") *
                    (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  auto doubles_65 = -2.0 * (LTOp) dTEov("mj", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("f_ii", "ii") * (LTOp) dT1("a_nn", "nn") *
                    (LTOp) dT2("e_mj", "b_mj", "mj") * (LTOp) d("mj", "e_mu", "e_mj") *
                    (LTOp) d("ii", "f_mu", "f_ii") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                    (LTOp) Sijkl("nn", "ij", "a_nn", "a_ij") * (LTOp) expand("i", "ii") *
                    (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_66 = -2.0 * (LTOp) dTEov("mi", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("f_jj", "jj") * (LTOp) dT1("b_nn", "nn") *
                    (LTOp) dT2("e_mi", "a_mi", "mi") * (LTOp) d("mi", "e_mu", "e_mi") *
                    (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") *
                    (LTOp) Sijkl("nn", "ij", "b_nn", "b_ij") * (LTOp) expand("j", "ij") *
                    (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  auto doubles_79 = (LTOp) dTEov("mj", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("f_ii", "ii") * (LTOp) dT1("a_nn", "nn") *
                    (LTOp) dT2("b_mj", "e_mj", "mj") * (LTOp) d("mj", "e_mu", "e_mj") *
                    (LTOp) d("ii", "f_mu", "f_ii") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                    (LTOp) Sijkl("nn", "ij", "a_nn", "a_ij") * (LTOp) expand("i", "ii") *
                    (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_80 = (LTOp) dTEov("mi", "e_mu", "K") * (LTOp) dTEov("nn", "f_mu", "K") *
                    (LTOp) dT1("f_jj", "jj") * (LTOp) dT1("b_nn", "nn") *
                    (LTOp) dT2("a_mi", "e_mi", "mi") * (LTOp) d("mi", "e_mu", "e_mi") *
                    (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") *
                    (LTOp) Sijkl("nn", "ij", "b_nn", "b_ij") * (LTOp) expand("j", "ij") *
                    (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  auto doubles_93 = (LTOp) dTEov("mj", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                    (LTOp) dT1("f_ii", "ii") * (LTOp) dT1("a_nn", "nn") *
                    (LTOp) dT2("e_mj", "b_mj", "mj") * (LTOp) d("mj", "e_mu", "e_mj") *
                    (LTOp) d("ii", "f_mu", "f_ii") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                    (LTOp) Sijkl("nn", "ij", "a_nn", "a_ij") * (LTOp) expand("i", "ii") *
                    (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_94 = (LTOp) dTEov("mi", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                    (LTOp) dT1("f_jj", "jj") * (LTOp) dT1("b_nn", "nn") *
                    (LTOp) dT2("e_mi", "a_mi", "mi") * (LTOp) d("mi", "e_mu", "e_mi") *
                    (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") *
                    (LTOp) Sijkl("nn", "ij", "b_nn", "b_ij") * (LTOp) expand("j", "ij") *
                    (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  auto doubles_103 = -0.5 * (LTOp) dTEov("mj", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                     (LTOp) dT1("f_ii", "ii") * (LTOp) dT1("a_nn", "nn") *
                     (LTOp) dT2("b_mj", "e_mj", "mj") * (LTOp) d("mj", "e_mu", "e_mj") *
                     (LTOp) d("ii", "f_mu", "f_ii") * (LTOp) Sijkl("mj", "ij", "b_mj", "b_ij") *
                     (LTOp) Sijkl("nn", "ij", "a_nn", "a_ij") * (LTOp) expand("i", "ii") *
                     (LTOp) expand("i", "ij") * (LTOp) expand("j", "mj") * (LTOp) expand("j", "ij");

  auto doubles_104 = -0.5 * (LTOp) dTEov("mi", "f_mu", "K") * (LTOp) dTEov("nn", "e_mu", "K") *
                     (LTOp) dT1("f_jj", "jj") * (LTOp) dT1("b_nn", "nn") *
                     (LTOp) dT2("a_mi", "e_mi", "mi") * (LTOp) d("mi", "e_mu", "e_mi") *
                     (LTOp) d("jj", "f_mu", "f_jj") * (LTOp) Sijkl("mi", "ij", "a_mi", "a_ij") *
                     (LTOp) Sijkl("nn", "ij", "b_nn", "b_ij") * (LTOp) expand("j", "ij") *
                     (LTOp) expand("i", "mi") * (LTOp) expand("i", "ij") * (LTOp) expand("j", "jj");

  dr2("a_ij", "b_ij", "ij").set(doubles_5);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_5.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_5." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_10);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_10.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_10." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_15);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_15.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_15." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_16);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_16.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_16." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_37);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_37.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_37." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_38);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_38.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_38." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_41);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_41.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_41." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_42);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_42.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_42." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_65);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_65.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_65." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_66);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_66.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_66." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_79);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_79.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_79." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_80);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_80.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_80." << std::endl;

  dr2("a_ij", "b_ij", "ij").set(doubles_93);
  if(rank == 0 && print_debug_info)
    op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij", "ij"), doubles_93.clone(), std::cout, use_opmin);
  op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  if(rank == 0) std::cout << "Finished executing doubles_93." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_94);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_94.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_94." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_103);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_103.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_103." << std::endl;

  // dr2("a_ij", "b_ij", "ij").set(doubles_104);
  // if(rank == 0 && print_debug_info) op_executor.print_op_binarized((LTOp) dr2("a_ij", "b_ij",
  // "ij"), doubles_104.clone(), use_opmin); op_executor.execute(dr2, use_opmin, ex_hw, do_profile);
  // if(rank == 0) std::cout << "Finished executing doubles_104." << std::endl;
}

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cout << "Please provide a JSON input file!" << std::endl;
    return 1;
  }

  std::string filename = std::string(argv[1]);
  if(fs::path(filename).extension() != ".json") {
    std::cout << "Only JSON format is supported. Please provide a JSON input file!" << std::endl;
    return 1;
  }
  std::ifstream input_params(filename);

  if(!input_params) {
    std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
    return 1;
  }
  std::string input_filestem = fs::path(filename).stem();
  json        params;

  input_params >> params;

  tamm::initialize(argc, argv);

  dlpno_test<double>(params);

  tamm::finalize();

  return 0;
}
