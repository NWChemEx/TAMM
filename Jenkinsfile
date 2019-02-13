def buildModuleMatrix = [
    		   "GCC 7.3.0":("gcc/7.3.0-xyzezhj openmpi/3.1.2-qve4xat cmake python")
		  ]
node{
    def nwxJenkins
    stage('Build CMakeBuild'){
      def installRoot="${WORKSPACE}/install"    
        sh """
	   set +x
	   source /etc/profile
	   rm -rf CMakeBuild
	   module load gcc/7.3.0-xyzezhj cmake
	   gh_token=4dfc676f4c5a2b1b9c3
	   gh_token+=f17bc2c3ebda1efa5f4e9
	   git clone https://NWXJenkins:\${gh_token}@github.com/NWChemEx-Project/CMakeBuild.git
	   cd CMakeBuild
	   cmake -H. -Bbuild -DBUILD_TESTS=OFF \
	   	     	     -DCMAKE_INSTALL_PREFIX=${installRoot}
	   cd build
	   make && make install
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