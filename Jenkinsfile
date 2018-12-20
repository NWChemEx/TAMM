def buildModuleMatrix = [
    		   "GCC 7.3.0":("gcc/7.3.0-xyzezhj openmpi/3.1.2-qve4xat globalarrays/5.7-rwhqwr3 cmake python netlib-lapack")
		  ]
node{
    def nwxJenkins
    stage('Import Jenkins Commands'){
        sh """
           rm -rf ~/.cpp_cache
           da_url=https://raw.githubusercontent.com/NWChemEx-Project/
           da_url+=DeveloperTools/master/ci/Jenkins/nwxJenkins.groovy
           wget \${da_url}
           """
    	nwxJenkins=load("nwxJenkins.groovy")
    }
    nwxJenkins.commonSteps(buildModuleMatrix, "SDE")
}
