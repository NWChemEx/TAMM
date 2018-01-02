#include "BuildEigen3/BuildEigen3.hpp"
#include <Eigen/Core>

void BuildEigen3::run_test()
{
    //Even if this gets optimized out the compiler will have proved it can find
    //Eigen
    Eigen::MatrixXd m(2,2);
}
