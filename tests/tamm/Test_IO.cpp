#include <chrono>
#include <filesystem>
#include <fstream>
#include <tamm/tamm.hpp>

using namespace tamm;

using T             = double;
using ComplexTensor = Tensor<std::complex<T>>;
bool   tammio       = true;
bool   profileio    = true;
double init_value   = 21.0;

template<typename TensorType>
void io_stats(ExecutionContext& gec, Tensor<TensorType>& tensor) {
  int rank = gec.pg().rank().value();

  long double nelements = 1;
  // Heuristic: Use 1 agg for every 14 GiB
  const long double ne_mb = 131072 * 14.0;
  const int         ndims = tensor.num_modes();
  for(auto i = 0; i < ndims; i++)
    nelements *= tensor.tiled_index_spaces()[i].index_space().num_indices();
  // nelements = tensor.size();
  int nagg = (nelements / (ne_mb * 1024)) + 1;

  const std::string nppn = std::to_string(nagg) + " nodes";

  if(rank == 0 && profileio)
    std::cout << "tensor size: " << std::fixed << std::setprecision(2)
              << (nelements * 8.0) / (1024 * 1024 * 1024.0)
              << "GiB, can write to disk using upto: " << nppn << std::endl;
}

std::tuple<TiledIndexSpace, TiledIndexSpace, TAMM_SIZE> setupTIS(TAMM_SIZE noa, TAMM_SIZE nva) {
  TAMM_SIZE n_occ_alpha    = noa;
  TAMM_SIZE n_occ_beta     = noa;
  TAMM_SIZE freeze_core    = 0;
  TAMM_SIZE freeze_virtual = 0;

  TAMM_SIZE nbf         = noa + nva;
  TAMM_SIZE nmo         = noa * 2 + nva * 2;
  TAMM_SIZE n_vir_alpha = nva;
  TAMM_SIZE n_vir_beta  = nva;
  TAMM_SIZE nocc        = n_occ_alpha * 2;

  std::vector<TAMM_SIZE> sizes = {n_occ_alpha - freeze_core, n_occ_beta - freeze_core,
                                  n_vir_alpha - freeze_virtual, n_vir_beta - freeze_virtual};

  const TAMM_SIZE total_orbitals = nmo - 2 * freeze_core - 2 * freeze_virtual;

  // Construction of tiled index space MO
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {
      {"occ", {range(0, nocc)}},
      {"occ_alpha", {range(0, n_occ_alpha)}},
      {"occ_beta", {range(n_occ_alpha, nocc)}},
      {"virt", {range(nocc, total_orbitals)}},
      {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
      {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}},
    },
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  Tile tce_tile = static_cast<Tile>(nbf / 10);
  if(tce_tile < 50 || tce_tile > 100) {
    if(tce_tile < 50) tce_tile = 50;   // 50 is the default tilesize for CCSD.
    if(tce_tile > 100) tce_tile = 100; // 100 is the max tilesize for CCSD.
    if(ProcGroup::world_rank() == 0)
      std::cout << std::endl << "Resetting tilesize to: " << tce_tile << std::endl;
  }

  if(ProcGroup::world_rank() == 0) {
    std::cout << "nbf = " << nbf << std::endl;
    std::cout << "nmo = " << nmo << std::endl;
    std::cout << "nocc = " << nocc << std::endl;

    std::cout << "n_occ_alpha = " << n_occ_alpha << std::endl;
    std::cout << "n_vir_alpha = " << n_vir_alpha << std::endl;
    std::cout << "n_occ_beta = " << n_occ_beta << std::endl;
    std::cout << "n_vir_beta = " << n_vir_beta << std::endl;
    std::cout << "tilesize   = " << tce_tile << std::endl;
  }

  std::vector<Tile> mo_tiles;

  tamm::Tile est_nt    = n_occ_alpha / tce_tile;
  tamm::Tile last_tile = n_occ_alpha % tce_tile;
  for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
  if(last_tile > 0) mo_tiles.push_back(last_tile);
  est_nt    = n_occ_beta / tce_tile;
  last_tile = n_occ_beta % tce_tile;
  for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
  if(last_tile > 0) mo_tiles.push_back(last_tile);

  est_nt    = n_vir_alpha / tce_tile;
  last_tile = n_vir_alpha % tce_tile;
  for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
  if(last_tile > 0) mo_tiles.push_back(last_tile);
  est_nt    = n_vir_beta / tce_tile;
  last_tile = n_vir_beta % tce_tile;
  for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
  if(last_tile > 0) mo_tiles.push_back(last_tile);

  TiledIndexSpace MO{MO_IS, mo_tiles}; //{ova,ova,ovb,ovb}};

  TiledIndexSpace tis_i{IndexSpace{range(6 * nva)}, tce_tile};

  return std::make_tuple(MO, tis_i, total_orbitals);
}

template<typename T>
void read_write(Tensor<T> tensor, std::string tstring) {
  std::string hdf5_str  = tstring + "_hdf5";
  std::string mpiio_str = tstring + "_mpiio";

  auto* ec = tensor.execution_context();

  // Roundtrip verification: fill with non-uniform data, hash it, write, zero the
  // in-memory tensor, read back, and require the hash to match.  Previously this
  // helper wrote+read without verifying the read-back values at all.
  random_ip(tensor, /*seed=*/12345u);
  ec->pg().barrier();
  const size_t hash_before = hash_tensor(tensor);

  write_to_disk(tensor, hdf5_str, tammio, profileio);

  // Clobber the in-memory data so a no-op read would be detected.
  Scheduler{*ec}(tensor() = T{0}).execute();
  ec->pg().barrier();

  read_from_disk(tensor, hdf5_str, tammio, {}, profileio);
  ec->pg().barrier();

  const size_t hash_after = hash_tensor(tensor);
  EXPECTS(hash_before == hash_after);
  // write_to_disk_mpiio(tensor,mpiio_str,tammio,profileio);
  // read_from_disk_mpiio(tensor,mpiio_str,tammio,{},profileio);
}

template<typename T>
void test_io_2d(Scheduler& sch, TiledIndexSpace tis, TiledIndexSpace tis_i) {
  TiledIndexSpace N = tis("all");
  TiledIndexSpace O = tis("occ");
  TiledIndexSpace V = tis("virt");

  TiledIndexSpace K = tis_i("all");

  Tensor<T> t2_oo{O, O};
  Tensor<T> t2_ov{O, V};
  Tensor<T> t2_vv{V, V};

  sch
    .allocate(t2_oo, t2_ov, t2_vv)(t2_oo() = init_value)(t2_ov() = init_value)(t2_vv() = init_value)
    .execute();

  read_write(t2_oo, "t2_oo");
  read_write(t2_ov, "t2_ov");
  read_write(t2_vv, "t2_vv");

  sch.deallocate(t2_oo, t2_ov, t2_vv).execute();
}

template<typename T>
void test_io_3d(Scheduler& sch, TiledIndexSpace tis, TiledIndexSpace tis_i) {
  TiledIndexSpace N = tis("all");
  TiledIndexSpace O = tis("occ");
  TiledIndexSpace V = tis("virt");

  TiledIndexSpace K = tis_i("all");

  Tensor<T> t3_ook{O, O, K};
  Tensor<T> t3_ovk{O, V, K};
  Tensor<T> t3_vvk{V, V, K};

  Tensor<T> t3_ooo{O, O, O};
  Tensor<T> t3_oov{O, O, V};
  Tensor<T> t3_ovv{O, V, V};
  Tensor<T> t3_vvv{V, V, V};

  sch.allocate(t3_ook, t3_ovk, t3_vvk)
    .allocate(t3_ooo, t3_oov, t3_ovv,
              t3_vvv)(t3_ook() = init_value)(t3_ovk() = init_value)(t3_vvk() = init_value)

      (t3_ooo() = init_value)(t3_oov() = init_value)(t3_ovv() = init_value)(t3_vvv() = init_value)
    .execute();

  read_write(t3_ook, "t3_ook");
  read_write(t3_ovk, "t3_ovk");
  read_write(t3_vvk, "t3_vvk");

  read_write(t3_ooo, "t3_ooo");
  read_write(t3_oov, "t3_oov");
  read_write(t3_ovv, "t3_ovv");
  read_write(t3_vvv, "t3_vvv");

  sch.deallocate(t3_ook, t3_ovk, t3_vvk).execute();
  sch.deallocate(t3_ooo, t3_oov, t3_ovv, t3_vvv).execute();
}

template<typename T>
void test_io_4d(Scheduler& sch, TiledIndexSpace tis, TiledIndexSpace tis_i) {
  TiledIndexSpace N = tis("all");
  TiledIndexSpace O = tis("occ");
  TiledIndexSpace V = tis("virt");

  TiledIndexSpace K = tis_i("all");

  Tensor<T> t_oooo{{O, O, O, O}, {2, 2}}; // OOOO
  Tensor<T> t_ooov{{O, O, O, V}, {2, 2}}; // OOOV
  Tensor<T> t_oovv{{O, O, V, V}, {2, 2}}; // OOVV
  Tensor<T> t_ovvv{{O, V, V, V}, {2, 2}}; // OVVV

  sch.allocate(t_oooo)(t_oooo() = init_value).execute();
  sch.allocate(t_ooov)(t_ooov() = init_value).execute();
  sch.allocate(t_oovv)(t_oovv() = init_value).execute();
  sch.allocate(t_ovvv)(t_ovvv() = init_value).execute();

  read_write(t_oooo, "t_oooo");
  read_write(t_ooov, "t_ooov");
  read_write(t_oovv, "t_oovv");
  read_write(t_ovvv, "t_ovvv");

  sch.deallocate(t_oooo).execute();
  sch.deallocate(t_ooov).execute();
  sch.deallocate(t_oovv).execute();
  sch.deallocate(t_ovvv).execute();
}

namespace fs = std::filesystem;

// All files written by this test go into this directory.
const std::string io_dir = "test_io/";

// Rank 0 performs a filesystem action; all ranks wait for it.
template<typename Func>
void on_rank0(ExecutionContext& ec, Func&& func) {
  if(ec.pg().rank() == 0) func();
  ec.pg().barrier();
}

// Prints what the test is about to do, so its output (including expected warnings) can be
// followed.
void step(ExecutionContext& ec, const std::string& what) {
  ec.pg().barrier();
  if(ec.print()) std::cout << "  - " << what << std::endl;
}

// A failed check is recorded rather than thrown, so all ranks stay in step through the test's
// collective calls and the test reports [FAILED] instead of aborting the program.
bool test_failed = false;

#define IO_CHECK(cond)                                                                   \
  do {                                                                                   \
    if(!(cond)) {                                                                        \
      test_failed = true;                                                                \
      std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " #cond << std::endl; \
    }                                                                                    \
  } while(0)

// Runs one test: a dashed line, the test's own output (including expected warnings), then
// [PASSED] or [FAILED] with its description. Returns whether it passed on all ranks.
template<typename Func>
bool run_test(ExecutionContext& ec, const std::string& description, Func&& test) {
  if(ec.print()) std::cout << std::string(80, '-') << std::endl;
  ec.pg().barrier();

  test_failed = false;
  test();

  const int failed     = test_failed ? 1 : 0;
  const int any_failed = ec.pg().allreduce(&failed, ReduceOp::max);
  if(ec.print()) std::cout << (any_failed ? "[FAILED] " : "[PASSED] ") << description << std::endl;
  return !any_failed;
}

// A copy of t to compare against later. (hash_tensor only covers the blocks a rank picks up
// through dynamic load balancing, so its value is not comparable across calls.)
Tensor<T> snapshot(ExecutionContext& ec, Tensor<T> t) {
  Tensor<T> ref{t.tiled_index_spaces()};
  Scheduler{ec}.allocate(ref)(ref() = t()).execute();
  return ref;
}

bool same_values(ExecutionContext& ec, Tensor<T> t, Tensor<T> ref) {
  Tensor<T> diff{t.tiled_index_spaces()};
  Scheduler{ec}.allocate(diff)(diff() = t())(diff() -= ref()).execute();
  const T diff_norm = norm(diff);
  Scheduler{ec}.deallocate(diff).execute();
  return diff_norm == T{0};
}

// A completed write is committed: the tensor file exists, its staging file does not, and it
// reads back what was written.
void test_commit(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_a.h5";
  on_rank0(ec, [&] { fs::remove(f); });

  Tensor<T> t{tis, tis};
  Scheduler{ec}.allocate(t).execute();
  random_ip(t, 101u);
  Tensor<T> written = snapshot(ec, t);

  step(ec, "writing tensor A to " + f);
  write_to_disk(t, f);
  step(ec, "checking " + f + " exists and " + f + ".tmp does not");
  on_rank0(ec, [&] {
    IO_CHECK(fs::exists(f));
    IO_CHECK(!fs::exists(f + ".tmp"));
  });

  step(ec, "zeroing tensor A, reading it back from " + f + " and comparing with what was written");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(t, f);
  IO_CHECK(same_values(ec, t, written));

  Scheduler{ec}.deallocate(t, written).execute();
  on_rank0(ec, [&] { fs::remove(f); });
}

// A staging file left behind by a killed job is overwritten by the next write.
void test_stale_staging_file(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_b.h5";
  step(ec, "creating a stale " + f + ".tmp, as a job killed while writing would leave");
  on_rank0(ec, [&] {
    fs::remove(f);
    std::ofstream(f + ".tmp") << "partial file left by a killed job";
  });

  Tensor<T> t{tis, tis};
  Scheduler{ec}.allocate(t).execute();
  random_ip(t, 102u);
  Tensor<T> written = snapshot(ec, t);

  step(ec, "writing tensor B to " + f);
  write_to_disk(t, f);
  step(ec, "checking " + f + ".tmp is gone");
  on_rank0(ec, [&] { IO_CHECK(!fs::exists(f + ".tmp")); });

  step(ec, "zeroing tensor B, reading it back from " + f + " and comparing with what was written");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(t, f);
  IO_CHECK(same_values(ec, t, written));

  Scheduler{ec}.deallocate(t, written).execute();
  on_rank0(ec, [&] { fs::remove(f); });
}

// A tensor file stores the tensor description, and a read accepts the file only for a tensor with
// the same description. The description is checked here directly with the same check
// read_from_disk performs, so a mismatch is reported without terminating the program.
void test_tensor_description(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_d.h5";
  on_rank0(ec, [&] { fs::remove(f); });

  Tensor<T> t{tis, tis};
  Scheduler{ec}.allocate(t).execute();
  random_ip(t, 106u);
  step(ec, "writing tensor D to " + f);
  write_to_disk(t, f);

  const Tile      n = tis.index_space().num_indices();
  TiledIndexSpace other{tis.index_space(), std::vector<Tile>{n / 2, n - n / 2}};
  Tensor<T>       u{other, other};
  const auto      desc_t = internal::describe_tensor(t);
  const auto      desc_u = internal::describe_tensor(u);

  on_rank0(ec, [&] {
    const hid_t file    = H5Fopen(f.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    const hid_t dataset = H5Dopen(file, "tensor", H5P_DEFAULT);

    internal::H5Check h5{f, "ERROR", true};
    const auto        format =
      internal::read_attribute<int64_t>(dataset, "tamm_format", H5T_NATIVE_INT64, h5);
    const auto tiles0 =
      internal::read_attribute<int64_t>(dataset, "tile_sizes_0", H5T_NATIVE_INT64, h5);
    const auto tiles1 =
      internal::read_attribute<int64_t>(dataset, "tile_sizes_1", H5T_NATIVE_INT64, h5);
    const auto blocks =
      internal::read_attribute<int64_t>(dataset, "nonzero_blocks", H5T_NATIVE_INT64, h5);
    const auto hash =
      internal::read_attribute<uint64_t>(dataset, "nonzero_hash", H5T_NATIVE_UINT64, h5);
    IO_CHECK(!h5.failed);
    if(!h5.failed) {
      std::cout << "  - description stored in " << f << ": tamm_format = " << format[0]
                << ", tile_sizes_0 = " << internal::join(tiles0)
                << ", tile_sizes_1 = " << internal::join(tiles1)
                << ", nonzero_blocks = " << blocks[0] << ", nonzero_hash = " << hash[0]
                << std::endl;
      IO_CHECK(tiles0 == desc_t.tile_sizes[0] && tiles1 == desc_t.tile_sizes[1]);
      IO_CHECK(blocks[0] == desc_t.nonzero_blocks && hash[0] == desc_t.nonzero_hash);
    }

    std::cout << "  - checking the description against tensor D (same tiling): expected to match"
              << std::endl;
    internal::H5Check same{f, "ERROR", true};
    internal::check_description(dataset, H5T_NATIVE_DOUBLE, desc_t, same);
    IO_CHECK(!same.failed);
    std::cout << "    " << (same.failed ? "mismatch" : "match") << std::endl;

    std::cout << "  - checking the description against a tensor with tiles " << n / 2 << ","
              << n - n / 2 << ": expected to report a mismatch" << std::endl;
    internal::H5Check tiling{f, "ERROR", true};
    internal::check_description(dataset, H5T_NATIVE_DOUBLE, desc_u, tiling);
    IO_CHECK(tiling.failed);

    std::cout << "  - checking the description against a float tensor: expected to report a "
                 "mismatch"
              << std::endl;
    internal::H5Check type{f, "ERROR", true};
    internal::check_description(dataset, H5T_NATIVE_FLOAT, desc_t, type);
    IO_CHECK(type.failed);

    H5Dclose(dataset);
    H5Fclose(file);
  });

  Scheduler{ec}.deallocate(t).execute();
  on_rank0(ec, [&] { fs::remove(f); });
}

// A failed write is not fatal and keeps the previous tensor file. The failure is forced by a
// directory occupying the staging file's name, which makes H5Fcreate fail.
void test_failed_write_keeps_previous(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_c.h5";
  on_rank0(ec, [&] {
    fs::remove(f);
    fs::remove_all(f + ".tmp");
  });

  Tensor<T> t{tis, tis};
  Scheduler{ec}.allocate(t).execute();

  step(ec, "[1] no " + f + " yet; creating directory " + f + ".tmp so the next write fails");
  on_rank0(ec, [&] { fs::create_directory(f + ".tmp"); });
  random_ip(t, 105u);
  step(ec, "[1] writing tensor C to " + f + " (expected to fail with a warning)");
  write_to_disk(t, f);
  step(ec, "[1] checking no " + f + " was created; removing directory " + f + ".tmp");
  on_rank0(ec, [&] {
    IO_CHECK(!fs::exists(f));
    fs::remove_all(f + ".tmp");
  });

  random_ip(t, 103u);
  Tensor<T> previous = snapshot(ec, t);
  step(ec, "[2] writing tensor C (version 1) to " + f + " (expected to succeed)");
  write_to_disk(t, f);

  step(ec, "[2] creating directory " + f + ".tmp so the next write fails");
  on_rank0(ec, [&] { fs::create_directory(f + ".tmp"); });
  random_ip(t, 104u);
  step(ec, "[2] writing tensor C (version 2) to " + f + " (expected to fail with a warning)");
  write_to_disk(t, f);

  step(ec, "[2] reading " + f + " back and checking it still holds version 1");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(t, f);
  IO_CHECK(same_values(ec, t, previous));

  Scheduler{ec}.deallocate(t, previous).execute();
  on_rank0(ec, [&] {
    fs::remove(f);
    fs::remove_all(f + ".tmp");
  });
}

// A group write is committed all-or-nothing: if one tensor file fails, none is committed and
// every tensor file keeps its previous version. The tensor files are left on disk afterwards.
void test_group_write_one_failure(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::vector<std::string> files{io_dir + "group_tensor_0.h5", io_dir + "group_tensor_1.h5",
                                       io_dir + "group_tensor_2.h5"};
  const size_t                   nt = files.size();
  on_rank0(ec, [&] {
    for(const auto& f: files) {
      fs::remove(f);
      fs::remove_all(f + ".tmp");
    }
  });

  std::vector<Tensor<T>> ts(nt);
  for(auto& t: ts) {
    t = Tensor<T>{tis, tis};
    Scheduler{ec}.allocate(t).execute();
  }

  std::vector<Tensor<T>> previous(nt);
  for(size_t i = 0; i < nt; i++) {
    random_ip(ts[i], 200u + i);
    previous[i] = snapshot(ec, ts[i]);
  }
  step(ec, "writing tensors G0, G1, G2 (version 1) as a group to " + io_dir +
             "group_tensor_{0,1,2}.h5 "
             "(expected to succeed)");
  write_to_disk_group(ec, ts, files);

  step(ec, "creating directory " + files[1] + ".tmp so writing G1 fails");
  on_rank0(ec, [&] { fs::create_directory(files[1] + ".tmp"); });
  for(size_t i = 0; i < nt; i++) random_ip(ts[i], 300u + i);
  step(ec, "writing tensors G0, G1, G2 (version 2) as a group (expected to fail with a warning; "
           "G0 and G2 write fine but must not be committed)");
  write_to_disk_group(ec, ts, files);

  step(ec, "checking the staging files of G0 and G2 were removed");
  on_rank0(ec, [&] {
    IO_CHECK(!fs::exists(files[0] + ".tmp"));
    IO_CHECK(!fs::exists(files[2] + ".tmp"));
  });

  step(ec, "reading the group back and checking all three tensors still hold version 1");
  for(auto& t: ts) Scheduler{ec}(t() = T{0}).execute();
  read_from_disk_group(ec, ts, files);
  for(size_t i = 0; i < nt; i++) IO_CHECK(same_values(ec, ts[i], previous[i]));

  for(size_t i = 0; i < nt; i++) Scheduler{ec}.deallocate(ts[i], previous[i]).execute();
  step(ec, "removing directory " + files[1] + ".tmp; " + io_dir +
             "group_tensor_{0,1,2}.h5 are left on disk");
  on_rank0(ec, [&] { fs::remove_all(files[1] + ".tmp"); });
}

// Reading a tensor file into a tensor with a different tiling is fatal and reports what differs.
// tamm_terminate ends the program, so this must be the last test.
void test_read_mismatch_fatal(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_d.h5";

  Tensor<T> t{tis, tis};
  Scheduler{ec}.allocate(t).execute();
  random_ip(t, 500u);
  step(ec, "writing tensor D to " + f);
  write_to_disk(t, f);

  const Tile      n = tis.index_space().num_indices();
  TiledIndexSpace other{tis.index_space(), std::vector<Tile>{n / 2, n - n / 2}};
  Tensor<T>       u{other, other};
  Scheduler{ec}.allocate(u).execute();
  step(ec, "reading " + f + " into a tensor with tiles " + std::to_string(n / 2) + "," +
             std::to_string(n - n / 2) + " (expected to be fatal: tiling differs)");
  read_from_disk(u, f);

  if(ec.print()) std::cout << "FAILED: the read did not terminate" << std::endl;
}

// A group read with unreadable tensor files is fatal and reports all of them together.
// tamm_terminate ends the program, so this must be the last test; ctest checks the message
// (PASS_REGULAR_EXPRESSION in test_tamm.cmake).
void test_group_read_fatal(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::vector<std::string> files{io_dir + "read_tensor_0.h5", io_dir + "read_tensor_1.h5",
                                       io_dir + "read_tensor_2.h5", io_dir + "read_tensor_3.h5"};
  std::vector<Tensor<T>>         ts(files.size());
  for(auto& t: ts) {
    t = Tensor<T>{tis, tis};
    Scheduler{ec}.allocate(t).execute();
    random_ip(t, 400u);
  }
  write_to_disk_group(ec, ts, files);

  on_rank0(ec, [&] {
    std::ofstream(files[1], std::ios::trunc) << "not an HDF5 file"; // garbage
    fs::resize_file(files[3], fs::file_size(files[3]) / 2);         // truncated
  });

  if(ec.print()) std::cout << "expecting a fatal read error" << std::endl;
  read_from_disk_group(ec, ts, files);

  if(ec.print()) std::cout << "FAILED: the read did not terminate" << std::endl;
}

int main(int argc, char* argv[]) {
  // argv[1]: N for the 3D tensor (NxNx12N) written to disk first.
  // argv[2]: optional dimension length for all remaining tests (default 100).
  if(argc < 2) {
    std::cout << "Usage: Test_IO <N for the NxNx12N tensor> [dimension length for the tests "
                 "(default 100)]\n";
    return 0;
  }

  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};

  Scheduler sch{ec_dense};

  if(ec.pg().rank() == 0) fs::create_directories(io_dir);
  ec.pg().barrier();
  Tile io_dim1 = atoi(argv[1]); // N of the NxNx12N tensor
  Tile io_dim2 = argc > 2 ? atoi(argv[2]) : 100;

  // Tiles of a dimension of length n: tiles of max(30, 5% of n) plus the remainder.
  auto make_tiles = [](Tile n) {
    const Tile        ts = std::max(30, (int) (n * 0.05));
    std::vector<Tile> tiles(n / ts, ts);
    if(n % ts > 0) tiles.push_back(n % ts);
    return tiles;
  };

  // Tile ts_ = std::max(30, (int) (io_dim1 * 0.05));
  //  auto [TIS, TIS_I, total_orbitals] = setupTIS(io_dim1, ts_);

  // test_io_2d<T>(sch, TIS, TIS_I);
  // test_io_3d<T>(sch, TIS, TIS_I);
  // test_io_4d<T>(sch, TIS, TIS_I);

  std::vector<Tile> gc_tiles = make_tiles(io_dim1);

  TiledIndexSpace tc_ij{IndexSpace{range(io_dim1)}, gc_tiles};
  TiledIndexSpace tci{IndexSpace{range(12 * io_dim1)}, 12 * io_dim1};
  Tensor<double>  gc{tc_ij, tc_ij, tci};
  gc.set_dense();
  io_stats(ec_dense, gc);

  sch.allocate(gc).execute();
  if(ec.print()) std::cout << "Writing a 3D tensor of size (NxNx12N) to disk ... " << std::endl;
  write_to_disk(gc, io_dir + "tensor3d.h5", true, true);

  sch.deallocate(gc).execute();

  TiledIndexSpace tis_io{IndexSpace{range(io_dim2)}, make_tiles(io_dim2)};
  if(ec.print())
    std::cout << "Remaining tests use tensors of dimension length " << io_dim2 << std::endl;
  bool all_passed = true;
  all_passed &= run_test(ec, "Write and read back a tensor file; no staging (.tmp) file is left",
                         [&] { test_commit(ec, tis_io); });
  all_passed &= run_test(ec, "Write over a staging (.tmp) file left behind by a killed job",
                         [&] { test_stale_staging_file(ec, tis_io); });
  all_passed &= run_test(ec, "A tensor file stores its tensor description and a read checks it",
                         [&] { test_tensor_description(ec, tis_io); });
  all_passed &= run_test(ec,
                         "A failed write is not fatal and never replaces an existing tensor file",
                         [&] { test_failed_write_keeps_previous(ec, tis_io); });
  all_passed &= run_test(ec,
                         "A group write where one tensor fails commits none of the tensor files",
                         [&] { test_group_write_one_failure(ec, tis_io); });
  // last: terminates the program
  // run_test(ec, "A group read with unreadable tensor files is fatal and lists all of them",
  //          [&] { test_group_read_fatal(ec, tis_io); });
  // run_test(ec, "Reading into a tensor with a different tiling is fatal and says what differs",
  //          [&] { test_read_mismatch_fatal(ec, tis_io); });

  tamm::finalize();

  return all_passed ? 0 : 1;
}
