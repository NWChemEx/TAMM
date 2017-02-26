#include <string>
#include <iostream>
#include "Error.h"

namespace tamm{

    namespace frontend {

 void Error(const std::string error_msg) {
     std::cerr << "Error: " << error_msg << std::endl;
     std::exit(EXIT_FAILURE);
    }

// When used, cannot be resolved.
//  void Error(const int line, const std::string error_msg) {
//      std::cerr << "Error at Line " << line << ": " << error_msg << std::endl;
//      std::exit(EXIT_FAILURE);
//     }

 void Error(const int line, const int position, const std::string error_msg) {
     if (position==0) std::cerr << "Error at Line " << line << ": " << error_msg << std::endl;
     else std::cerr << "Error at Line " << line << ", Column " << position << ": " << error_msg << std::endl;
     std::exit(EXIT_FAILURE);
    }

}

}