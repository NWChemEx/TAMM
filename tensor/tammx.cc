#include "tensor/tammx.h"

namespace tammx {

std::vector<Spin> TCE::spins_;
std::vector<Spatial> TCE::spatials_;
std::vector<size_t> TCE::sizes_;
bool TCE::spin_restricted_;
Irrep TCE::irrep_f_, TCE::irrep_v_, TCE::irrep_t_;
Irrep TCE::irrep_x_, TCE::irrep_y_;
BlockDim TCE::noa_, TCE::noab_;
BlockDim TCE::nva_, TCE::nvab_;

}  // namespace tammx
