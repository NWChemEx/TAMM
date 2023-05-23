#include "ga/macdecls.h"
#include "mpi.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

using T             = double;
using ComplexTensor = Tensor<std::complex<T>>;
bool   tammio       = true;
bool   profileio    = true;
double init_value   = 21.0;

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
    if(GA_Nodeid() == 0)
      std::cout << std::endl << "Resetting tilesize to: " << tce_tile << std::endl;
  }

  if(GA_Nodeid() == 0) {
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

  write_to_disk(tensor, hdf5_str, tammio, profileio);
  read_from_disk(tensor, hdf5_str, tammio, {}, profileio);
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

int main(int argc, char* argv[]) {
  if(argc < 2) {
    std::cout << "Please provide a dimension size!\n";
    return 0;
  }

  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};

  Scheduler sch{ec_dense};
  Tile      nbf = atoi(argv[1]);
  Tile      ts_ = std::max(30, (int) (nbf * 0.05));

  // auto [TIS, TIS_I, total_orbitals] = setupTIS(nbf, ts_);

  // test_io_2d<T>(sch, TIS, TIS_I);
  // test_io_3d<T>(sch, TIS, TIS_I);
  // test_io_4d<T>(sch, TIS, TIS_I);

  std::vector<Tile> gc_tiles;
  Tile              est_nt    = nbf / ts_;
  Tile              last_tile = nbf % ts_;
  for(tamm::Tile x = 0; x < est_nt; x++) gc_tiles.push_back(ts_);
  if(last_tile > 0) gc_tiles.push_back(last_tile);

  TiledIndexSpace tc_ij{IndexSpace{range(nbf)}, gc_tiles};
  TiledIndexSpace tci{IndexSpace{range(12 * nbf)}, 12 * nbf};
  Tensor<double>  gc{tc_ij, tc_ij, tci};
  gc.set_dense();
  sch.allocate(gc).execute();
  if(ec.print()) std::cout << "Writing a 3D tensor of size (NxNx12N) to disk ... " << std::endl;
  write_to_disk(gc, "tensor3d", true, true);

  sch.deallocate(gc).execute();

  tamm::finalize();

  return 0;
}
