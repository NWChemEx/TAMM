def buildModuleMatrix = [
    		   "GCC 7.3.0":("gcc/7.3.0-xyzezhj openmpi/3.1.2-qve4xat cmake python")
		  ]
node{
    def nwxJenkins
    stage('Build CMakeBuild'){
        sh """
	   module load gcc/7.3.0-xyzezhj
	   git clone https://github.com/NWChemEx-Project/CMakeBuild.git
	   cd CMakeBuild
	   cmake -H. -Bbuild -DBUILD_TESTS=OFF \
	   	     	     -DCMAKE_INSTALL_PREFIX=${WORKSPACE}/install
	   cmake --build build --target install
	   """
    }
    stage('Import Jenkins Commands'){
        sh """
           rm -rf ~/.cpp_cache
           da_url=https://raw.githubusercontent.com/NWChemEx-Project/
           da_url+=DeveloperTools/master/ci/Jenkins/nwxJenkins.groovy
           wget \${da_url}
           """
    	nwxJenkins=load("nwxJenkins.groovy")
    }
    nwxJenkins.commonSteps(buildModuleMatrix, "TAMM")
}