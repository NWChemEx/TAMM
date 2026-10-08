#include <chrono>
#include <filesystem>
#include <fstream>
#include <tamm/tamm.hpp>

using namespace tamm;

using T = double;

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

namespace fs = std::filesystem;

// All files written by this test go into this directory.
const std::string io_dir = "io_test/";

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
  write_to_disk(ec, t, f);
  step(ec, "checking " + f + " exists and " + f + ".tmp does not");
  on_rank0(ec, [&] {
    IO_CHECK(fs::exists(f));
    IO_CHECK(!fs::exists(f + ".tmp"));
  });

  step(ec, "zeroing tensor A, reading it back from " + f + " and comparing with what was written");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, t, f);
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
  write_to_disk(ec, t, f);
  step(ec, "checking " + f + ".tmp is gone");
  on_rank0(ec, [&] { IO_CHECK(!fs::exists(f + ".tmp")); });

  step(ec, "zeroing tensor B, reading it back from " + f + " and comparing with what was written");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, t, f);
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
  write_to_disk(ec, t, f);

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

// A subgroup of the ranks writes a tensor allocated on all ranks (blocks owned outside the
// subgroup are fetched), and all ranks read it back, i.e. a different number of ranks than wrote
// it; then the reverse. Covers a plain tensor and a spin tensor whose zero blocks are skipped.
void test_subgroup_io(ExecutionContext& ec, TiledIndexSpace tis) {
  // A spin-blocked space: occupied and virtual parts, each with alpha and beta halves.
  const Tile      n  = tis.index_space().num_indices();
  const Tile      no = n / 4, nv = n / 2 - no; // per spin
  IndexSpace      mo_is{range(0, 2 * (no + nv)),
                        {{"occ", {range(0, 2 * no)}}, {"virt", {range(2 * no, 2 * (no + nv))}}},
                        {{Spin{1}, {range(0, no), range(2 * no, 2 * no + nv)}},
                         {Spin{2}, {range(no, 2 * no), range(2 * no + nv, 2 * (no + nv))}}}};
  const Tile      ts = std::max<Tile>(1, std::min(no, nv) / 2);
  TiledIndexSpace mo{mo_is, ts};
  TiledIndexSpace O = mo("occ"), V = mo("virt");

  const int         nsub   = std::max(1, static_cast<int>(ec.pg().size().value() / 2));
  ProcGroup         sub_pg = ProcGroup::create_subgroup(ec.pg(), nsub);
  ExecutionContext* sub_ec =
    sub_pg.is_valid() ? new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga)
                      : nullptr;

  struct Case {
    std::string name;
    Tensor<T>   tensor;
  };
  std::vector<Case> cases{{"plain tensor", Tensor<T>{tis, tis}},
                          {"spin tensor", Tensor<T>{{O, O, V, V}, {2, 2}}}};

  for(auto& c: cases) {
    Tensor<T>& t = c.tensor;
    Scheduler{ec}.allocate(t).execute();
    random_ip(t, 107u);
    Tensor<T> written = snapshot(ec, t);

    const std::string f1 = io_dir + "subgroup_write.h5", f2 = io_dir + "subgroup_read.h5";
    step(ec, c.name + " allocated on all " + std::to_string(ec.pg().size().value()) +
               " ranks: the first " + std::to_string(nsub) +
               " rank(s) write it, all ranks read it");
    if(sub_ec) write_to_disk(*sub_ec, t, f1);
    ec.pg().barrier();
    Scheduler{ec}(t() = T{0}).execute();
    read_from_disk(ec, t, f1);
    IO_CHECK(same_values(ec, t, written));

    step(ec,
         c.name + ": all ranks write it, the first " + std::to_string(nsub) + " rank(s) read it");
    write_to_disk(ec, t, f2);
    Scheduler{ec}(t() = T{0}).execute();
    if(sub_ec) read_from_disk(*sub_ec, t, f2);
    ec.pg().barrier();
    IO_CHECK(same_values(ec, t, written));

    Scheduler{ec}.deallocate(t, written).execute();
    on_rank0(ec, [&] {
      fs::remove(f1);
      fs::remove(f2);
    });
  }

  if(sub_ec) {
    sub_ec->flush_and_sync();
    delete sub_ec;
    sub_pg.destroy_coll();
  }
}

// A named tensor written to its own tensor file.
struct NamedTensor {
  std::string name;
  Tensor<T>   tensor;
};

// 2D tensors over the MO space: occupied-occupied, occupied-virtual and virtual-virtual blocks.
std::vector<NamedTensor> tensors_2d(TiledIndexSpace mo) {
  TiledIndexSpace O = mo("occ"), V = mo("virt");
  return {{"t2_oo", {O, O}}, {"t2_ov", {O, V}}, {"t2_vv", {V, V}}};
}

// 3D tensors: MO pairs with an auxiliary index (Cholesky-vector-like), and MO triples.
std::vector<NamedTensor> tensors_3d(TiledIndexSpace mo, TiledIndexSpace aux) {
  TiledIndexSpace O = mo("occ"), V = mo("virt"), K = aux("all");
  return {{"t3_ook", {O, O, K}}, {"t3_ovk", {O, V, K}}, {"t3_vvk", {V, V, K}},
          {"t3_ooo", {O, O, O}}, {"t3_oov", {O, O, V}}, {"t3_ovv", {O, V, V}},
          {"t3_vvv", {V, V, V}}};
}

// Spin-blocked 4D tensors (two-electron-integral-like), whose zero blocks are skipped.
std::vector<NamedTensor> tensors_4d(TiledIndexSpace mo) {
  TiledIndexSpace O = mo("occ"), V = mo("virt");
  return {{"t_oooo", {{O, O, O, O}, {2, 2}}},
          {"t_ooov", {{O, O, O, V}, {2, 2}}},
          {"t_oovv", {{O, O, V, V}, {2, 2}}},
          {"t_ovvv", {{O, V, V, V}, {2, 2}}}};
}

// 2D, 3D and 4D tensors over spin-blocked molecular-orbital spaces and an auxiliary space, as
// used in coupled-cluster methods, filled with random values, written as one list of tensor files,
// zeroed, read back as one list, and each checked against a copy of what was written. The n
// molecular orbitals are split evenly between the two spins, and per spin 10% are occupied and
// 90% virtual.
void test_spin_blocked_tensors(ExecutionContext& ec, Tile n) {
  const TAMM_SIZE per_spin         = std::max<TAMM_SIZE>(2, n / 2);
  const TAMM_SIZE noa              = std::max<TAMM_SIZE>(1, per_spin / 10);
  const TAMM_SIZE nva              = per_spin - noa;
  auto [MO, tis_i, total_orbitals] = setupTIS(noa, nva);

  std::vector<NamedTensor> named = tensors_2d(MO);
  for(auto& t: tensors_3d(MO, tis_i)) named.push_back(t);
  for(auto& t: tensors_4d(MO)) named.push_back(t);

  std::vector<Tensor<T>>   tensors, written;
  std::vector<std::string> files;
  for(size_t i = 0; i < named.size(); i++) {
    Tensor<T>& t = named[i].tensor;
    Scheduler{ec}.allocate(t).execute();
    random_ip(t, 600u + i);
    tensors.push_back(t);
    written.push_back(snapshot(ec, t));
    files.push_back(io_dir + named[i].name + ".h5");
  }

  step(ec, "writing " + std::to_string(tensors.size()) +
             " tensors (3 2D, 7 3D, 4 4D) as one list to " + io_dir + "t*.h5");
  write_to_disk(ec, tensors, files, true);

  step(ec, "zeroing them, reading them back as one list and checking each");
  for(auto& t: tensors) Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, tensors, files, true);
  for(size_t i = 0; i < tensors.size(); i++) IO_CHECK(same_values(ec, tensors[i], written[i]));

  for(size_t i = 0; i < tensors.size(); i++)
    Scheduler{ec}.deallocate(tensors[i], written[i]).execute();
  on_rank0(ec, [&] {
    for(const auto& f: files) fs::remove(f);
  });
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
  write_to_disk(ec, t, f);
  step(ec, "[1] checking no " + f + " was created; removing directory " + f + ".tmp");
  on_rank0(ec, [&] {
    IO_CHECK(!fs::exists(f));
    fs::remove_all(f + ".tmp");
  });

  random_ip(t, 103u);
  Tensor<T> previous = snapshot(ec, t);
  step(ec, "[2] writing tensor C (version 1) to " + f + " (expected to succeed)");
  write_to_disk(ec, t, f);

  step(ec, "[2] creating directory " + f + ".tmp so the next write fails");
  on_rank0(ec, [&] { fs::create_directory(f + ".tmp"); });
  random_ip(t, 104u);
  step(ec, "[2] writing tensor C (version 2) to " + f + " (expected to fail with a warning)");
  write_to_disk(ec, t, f);

  step(ec, "[2] reading " + f + " back and checking it still holds version 1");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, t, f);
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
  write_to_disk(ec, ts, files);

  step(ec, "creating directory " + files[1] + ".tmp so writing G1 fails");
  on_rank0(ec, [&] { fs::create_directory(files[1] + ".tmp"); });
  for(size_t i = 0; i < nt; i++) random_ip(ts[i], 300u + i);
  step(ec, "writing tensors G0, G1, G2 (version 2) as a group (expected to fail with a warning; "
           "G0 and G2 write fine but must not be committed)");
  write_to_disk(ec, ts, files);

  step(ec, "checking the staging files of G0 and G2 were removed");
  on_rank0(ec, [&] {
    IO_CHECK(!fs::exists(files[0] + ".tmp"));
    IO_CHECK(!fs::exists(files[2] + ".tmp"));
  });

  step(ec, "reading the group back and checking all three tensors still hold version 1");
  for(auto& t: ts) Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, ts, files);
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
  write_to_disk(ec, t, f);

  const Tile      n = tis.index_space().num_indices();
  TiledIndexSpace other{tis.index_space(), std::vector<Tile>{n / 2, n - n / 2}};
  Tensor<T>       u{other, other};
  Scheduler{ec}.allocate(u).execute();
  step(ec, "reading " + f + " into a tensor with tiles " + std::to_string(n / 2) + "," +
             std::to_string(n - n / 2) + " (expected to be fatal: tiling differs)");
  read_from_disk(ec, u, f);

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
  write_to_disk(ec, ts, files);

  on_rank0(ec, [&] {
    std::ofstream(files[1], std::ios::trunc) << "not an HDF5 file"; // garbage
    fs::resize_file(files[3], fs::file_size(files[3]) / 2);         // truncated
  });

  if(ec.print()) std::cout << "expecting a fatal read error" << std::endl;
  read_from_disk(ec, ts, files);

  if(ec.print()) std::cout << "FAILED: the read did not terminate" << std::endl;
}

// A dense tensor (an N-D Global Array, whose blocks are not each stored on one rank) is written and
// read back. Dense tensors are allocated on an execution context with a dense distribution.
void test_dense_tensor(ExecutionContext& ec, TiledIndexSpace tis) {
  const std::string f = io_dir + "tensor_dense.h5";
  on_rank0(ec, [&] { fs::remove(f); });

  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
  Tensor<T>        t{tis, tis};
  t.set_dense();
  Scheduler{ec_dense}.allocate(t).execute();
  random_ip(t, 104u);
  Tensor<T> written = snapshot(ec, t);

  step(ec, "writing dense tensor E to " + f);
  write_to_disk(ec, t, f);
  step(ec, "zeroing tensor E, reading it back from " + f + " and comparing with what was written");
  Scheduler{ec}(t() = T{0}).execute();
  read_from_disk(ec, t, f);
  IO_CHECK(same_values(ec, t, written));

  Scheduler{ec}.deallocate(written).execute();
  Scheduler{ec_dense}.deallocate(t).execute();
  on_rank0(ec, [&] { fs::remove(f); });
}

int main(int argc, char* argv[]) {
  // argv[1]: N for the 2D tensor (100N x 100N) and the 4D tensor (NxNxNxN), written and read first.
  // argv[2]: optional tile size as a percentage of a dimension's length (default 5).
  // argv[3]: optional; 0 runs only the 2D and 4D tensor tests, any other value (default 1) also
  //          runs the remaining tests.
  if(argc < 2) {
    std::cout << "Usage: Test_IO <N for the 100N x 100N and NxNxNxN tensors> [tile size as % of a "
                 "dimension's length (default 5)] [0: only the 2D and 4D tensor tests; 1: all "
                 "tests (default 1)]\n";
    return 0;
  }

  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  ExecutionContext ec_dense{ec.pg(), DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec_dense};

  if(ec.pg().rank() == 0) fs::create_directories(io_dir);
  ec.pg().barrier();
  Tile       io_dim1  = atoi(argv[1]); // N of the NxNxNxN tensor
  const int  tile_pct = argc > 2 ? atoi(argv[2]) : 5;
  const bool run_all  = argc > 3 ? atoi(argv[3]) != 0 : true;
  const Tile io_dim2  = 100; // dimension length of the tensors in the remaining tests
  if(ec.print())
    std::cout << "Nodes: " << ec.nnodes() << ", ranks: " << ec.pg().size().value()
              << ", ranks per node: " << ec.ppn() << ", tile size: " << tile_pct
              << "% of each dimension" << std::endl;

  // Tiles of a dimension of length n: tiles of max(30, tile_pct% of n) plus the remainder.
  auto make_tiles = [tile_pct](Tile n) {
    const Tile        ts = std::clamp((int) (n * tile_pct / 100), 30, 2000);
    std::vector<Tile> tiles(n / ts, ts);
    if(n % ts > 0) tiles.push_back(n % ts);
    return tiles;
  };

  // Fills t with random values, writes it to file, zeroes it, reads it back and compares the norms.
  auto write_read = [&](auto t, const std::string& file, unsigned int seed) {
    using E = decltype(norm(t)); // the element type
    if(ec.print()) std::cout << "GiB per I/O node: " << internal::io_gib_per_node() << std::endl;
    random_ip(t, seed);
    const E written = norm(t);
    write_to_disk(ec, t, file, true);
    sch(t() = E{0}).execute();
    read_from_disk(ec, t, file, true);
    const E read = norm(t);
    if(ec.print()) {
      std::cout << "Norm written: " << written << ", norm read: " << read << std::endl;
      // the norms may be summed in a different order
      if(std::abs(read - written) > 1e-12 * std::abs(written))
        std::cout << "The norms of the tensor written and read back do not match" << std::endl;
    }
  };

  const Tile              dim_2d = 100 * io_dim1;
  TiledIndexSpace         tis_2d{IndexSpace{range(dim_2d)}, make_tiles(dim_2d)};
  Tensor<std::complex<T>> t2d{tis_2d, tis_2d};

  sch.allocate(t2d).execute();
  if(ec.print()) {
    const size_t ntiles = tis_2d.num_tiles();
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Writing a complex 2D tensor of size (100N x 100N), N = " << io_dim1
              << ", tile size " << tis_2d.tile_size(0) << ", " << ntiles << " tiles per dimension, "
              << ntiles * ntiles << " tiles in total, to disk and reading it back ... "
              << std::endl;
  }
  write_read(t2d, io_dir + "tensor2d.h5", 201u);

  sch.deallocate(t2d).execute();

  TiledIndexSpace tis_n{IndexSpace{range(io_dim1)}, make_tiles(io_dim1)};
  Tensor<T>       t4d{tis_n, tis_n, tis_n, tis_n};

  sch.allocate(t4d).execute();
  if(ec.print()) {
    const size_t ntiles = tis_n.num_tiles();
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Writing a 4D tensor of size (NxNxNxN), N = " << io_dim1 << ", tile size "
              << tis_n.tile_size(0) << ", " << ntiles << " tiles per dimension, "
              << ntiles * ntiles * ntiles * ntiles
              << " tiles in total, to disk and reading it back ... " << std::endl;
  }
  write_read(t4d, io_dir + "tensor4d.h5", 202u);
  sch.deallocate(t4d).execute();

  if(!run_all) {
    if(ec.print()) std::cout << std::string(80, '-') << std::endl;
    tamm::finalize();
    return 0;
  }

  bool all_passed = true;
  all_passed &= run_test(ec, "2D, 3D and 4D spin-blocked tensors written and read as one list",
                         [&] { test_spin_blocked_tensors(ec, io_dim1); });

  TiledIndexSpace tis_io{IndexSpace{range(io_dim2)}, make_tiles(io_dim2)};
  if(ec.print())
    std::cout << "Remaining tests use tensors of dimension length " << io_dim2 << std::endl;
  all_passed &= run_test(ec, "Write and read back a tensor file; no staging (.tmp) file is left",
                         [&] { test_commit(ec, tis_io); });
  all_passed &= run_test(ec, "Write over a staging (.tmp) file left behind by a killed job",
                         [&] { test_stale_staging_file(ec, tis_io); });
  all_passed &= run_test(ec, "A tensor file stores its tensor description and a read checks it",
                         [&] { test_tensor_description(ec, tis_io); });
  all_passed &= run_test(
    ec, "A subgroup writes or reads a tensor allocated on all ranks (plain and spin tensors)",
    [&] { test_subgroup_io(ec, tis_io); });
  all_passed &= run_test(ec,
                         "A failed write is not fatal and never replaces an existing tensor file",
                         [&] { test_failed_write_keeps_previous(ec, tis_io); });
  all_passed &= run_test(ec,
                         "A group write where one tensor fails commits none of the tensor files",
                         [&] { test_group_write_one_failure(ec, tis_io); });
  all_passed &=
    run_test(ec, "A dense tensor is written and read back", [&] { test_dense_tensor(ec, tis_io); });
  // last: terminates the program
  // run_test(ec, "A group read with unreadable tensor files is fatal and lists all of them",
  //          [&] { test_group_read_fatal(ec, tis_io); });
  // run_test(ec, "Reading into a tensor with a different tiling is fatal and says what differs",
  //          [&] { test_read_mismatch_fatal(ec, tis_io); });

  tamm::finalize();

  return all_passed ? 0 : 1;
}
