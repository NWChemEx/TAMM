#include "tammx/tammx.h"

namespace tammx {

std::vector<Spin> TCE::spins_;
std::vector<Irrep> TCE::spatials_;
std::vector<int64_t> TCE::sizes_;
std::vector<int64_t> TCE::offsets_;
bool TCE::spin_restricted_;
Irrep TCE::irrep_f_, TCE::irrep_v_, TCE::irrep_t_;
Irrep TCE::irrep_x_, TCE::irrep_y_;
BlockIndex TCE::noa_, TCE::noab_;
BlockIndex TCE::nva_, TCE::nvab_;

// Integer *int_mb_tammx;
// double *dbl_mb_tammx;

Integer* MA::int_mb_;
double* MA::dbl_mb_;

std::map<DistributionFactory::Key, std::shared_ptr<Distribution>,DistributionFactory::KeyLessThan> DistributionFactory::distributions_;

}  // namespace tammx
