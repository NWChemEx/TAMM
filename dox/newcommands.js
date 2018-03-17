/*
 * This file contains approved Latex aliases for various commonly occuring
 * mathematical things.  Please use them throughout your documentation.
 *
 */

MathJax.Hub.Config({
    TeX: {
        Macros: {
            //Number of atomic orbitals
            nao: "{N_\\text{AO}}",
            //Number of alpha electrons
            na : "{N_\\alpha}",
            //Number of beta electrons
            nb : "{N_\\beta}",
            //Number of electrons (either alpha or beta)
            ns : "{N_\\text{x}}",
            //Number of auxillary basis functions
            naux : "{N_\\text{Aux}}",
            //Set to accompany ns detailing the options for x
            spins : "{x\\in\\left\\lbrace\\alpha,\\beta\\right\\rbrace}",
            //A sum over AOs, takes one argument, the index letter
            sumao : ["{\\sum_{#1 =1}^\\nao}",1],
            //Same as sumao except for alpha orbitals
            suma  : ["{\\sum_{#1 =1}^\\na }",1],
            //Same as sumao except for beta orbitals
            sumb  : ["{\\sum_{#1 =1}^\\nb }",1],
            //Same as sumao except for arbitrary spin orbitals
            sums  : ["{\\sum_{#1 =1}^\\ns }",1],
            //Same as sumao except for auxillary orbitals
            sumaux  : ["{\\sum_{#1 =1}^\\naux }",1],
            //A spin density matrix takes spin as argument
            sdens : ["{\\mathbf{{^{#1}}P}}",1],
            //Same as sdens except for the Fock matrix
            sfock : ["{\\mathbf{{^{#1}}F}}",1],
            //Same as sdens except for MO coeffecients
            sC    : ["{\\mathbf{{^{#1}}C}}",1],
            //Same as sdens except for adjoint of MO coeffecients
            sCd   : ["{\\mathbf{{^{#1}}C^\\dagger}}",1],
            //Same as sdens except for Coloumb matrix
            sJ    : ["{\\mathbf{{^{#1}}J}}",1],
            //Same as sdens except for exchange matrix
            sK    : ["{\\mathbf{{^{#1}}K}}",1],
            //Adjoint of density fitting coeffecients
            Dd    : "{\\mathbf{D^\\dagger}}",
            //Adjoint of lower-triangluar matrix from Cholesky decomposition
            Ld    : "{\\mathbf{L^\\dagger}}",
            //Inverse of Ld
            Ldinv : "{\\mathbf{\\left[L^\\dagger\\right]^{-1}}}",
            //Inverse of lower-triangular matrix from Cholesky decomposition
            Linv  : "{\\mathbf{L^{-1}}}",
            //Inverse of the metric matrix for density fitting
            minv  : "{\\mathbf{M^{-1}}}",
            //Command for writing the matrix element of a decorated matrix
            //Takes matrix's symbol and its indices
            mate  : ["{\\left[{#1} \\right]_{#2}}",2],
            //Same as mate except for tensor, third argument is upper indices
            tene  : ["{\\left[{#1} \\right]_{#2}^{#3}}",3],
            //Short-cut for displaying AO Coulomb integral
            cint  : "{\\left(\\mu\\nu\\mid\\lambda\\sigma\\right)}",
            //Short-cut for displaying AO Exchange integral 
            eint  : "{\\left(\\mu\\sigma\\mid\\lambda\\nu\\right)}",
            //Command for Big-O notation
            bigo  : ["{\\mathcal{O}\\left(#1\\right)}",1]
        }
    }
});
