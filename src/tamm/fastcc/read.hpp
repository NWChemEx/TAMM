#ifndef READ_HPP
#define READ_HPP
#include "contract.hpp"
#include <fstream>
#include <climits>
#include <iostream>
#include <string>




void CoOrdinate::write(std::string filename) const {
  std::ofstream file(filename, std::ios_base::app);
  file << this->to_string() << std::endl;
  file << std::endl;
  file.close();
}
#endif
