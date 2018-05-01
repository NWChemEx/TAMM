def repoName= "TAMM"
def depends = ["CMakeBuild"] as String[]
def commonModules = "cmake llvm "
def MKL_ROOT="/blues/gpfs/home/software/spack-0.10.1/opt/spack/linux-centos7-x86_64/gcc-4.8.5/intel-parallel-studio-cluster.2018.0-tpfbvgalxkjs4i6yhyyuxhyporbxpivw/compilers_and_libraries_2018.1.163/linux/mkl"
def INTEL_ROOT="/blues/gpfs/home/software/spack-0.10.1/opt/spack/linux-centos7-x86_64/gcc-4.8.5/intel-parallel-studio-cluster.2018.0-tpfbvgalxkjs4i6yhyyuxhyporbxpivw/compilers_and_libraries_2018.1.163/linux"

def buildModuleMatrix = [
    		   "GCC":(commonModules + "gcc/7.1.0"),
		   "Intel":(commonModules + "gcc/7.1.0 intel-parallel-studio/cluster.2018.0-tpfbvga")
		  ]
def cmakeCommandMatrix = [
    		   "GCC":"-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran -DBUILD_SHARED_LIBS=OFF",
		   "Intel":"-DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DBUILD_SHARED_LIBS=OFF -DCBLAS_LIBRARIES=\"${MKL_ROOT}/lib/intel64_lin/libmkl_lapack95_ilp64.a;${MKL_ROOT}/lib/intel64_lin/libmkl_blas95_ilp64.a;${MKL_ROOT}/lib/intel64/libmkl_core.a;${MKL_ROOT}/lib/intel64/libmkl_intel_thread.a;${INTEL_ROOT}/compiler/lib/intel64_lin/libiomp5.a\" -DLAPACKE_LIBRARIES=\"${MKL_ROOT}/lib/intel64/libmkl_lapack95_ilp64.a;${MKL_ROOT}/lib/intel64/libmkl_blas95_ilp64.a;${MKL_ROOT}/lib/intel64_lin/libmkl_core.a;${MKL_ROOT}/lib/intel64_lin/libmkl_intel_thread.a;${INTEL_ROOT}/compiler/lib/intel64_lin/libiomp5.a\" -DCBLAS_INCLUDE_DIRS=${MKL_ROOT}/include -DLAPACKE_INCLUDE_DIRS=${MKL_ROOT}/include"
		   ]

def credentialsID = "422b0eed-700d-444d-961c-1e58cc75cda2"

/************************************************
 ************************************************
    Shouldn't need to modify anything below...
 ************************************************    
 ************************************************/

def buildTypeList=buildModuleMatrix.keySet() as String[]		  
def nwxJenkins

node{
for (int i=0; i<buildTypeList.size(); i++){

    def buildType = "${buildTypeList[i]}"
    def cmakeCommand = "${cmakeCommandMatrix[buildType]}"

    stage("${buildType}: Set-Up Workspace"){
        deleteDir()
        checkout scm
    }

    stage('${buildType}: Import Jenkins Commands'){
        sh "wget https://raw.githubusercontent.com/NWChemEx-Project/DeveloperTools/master/ci/Jenkins/nwxJenkins.groovy"
    	nwxJenkins=load("nwxJenkins.groovy")
    }

    stage('${buildType}: Export Module List'){
        def buildModules = "${buildModuleMatrix[buildType]}"
    nwxJenkins.exportModules(buildModules)
    }

    stage('Check Code Formatting'){
        nwxJenkins.formatCode()
    }

    stage('Build Dependencies'){
        nwxJenkins.buildDependencies(depends, cmakeCommand, credentialsID)
    }

    stage('Build Repo'){
        nwxJenkins.compileRepo(repoName, "False", cmakeCommand)
    }

    stage('Test Repo'){
        nwxJenkins.testRepo()
    }

}
}
