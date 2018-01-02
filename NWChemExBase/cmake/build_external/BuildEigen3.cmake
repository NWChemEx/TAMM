ExternalProject_Add(Eigen3_External
    URL http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    )

