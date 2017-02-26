//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------

/// Error reporting methods
#ifndef __TAMM_ERROR_H__
#define __TAMM_ERROR_H__

namespace tamm{

    namespace frontend {

 void Error(const std::string error_msg);
 void Error(const int line, const int position, const std::string error_msg);
 
}

}

#endif
