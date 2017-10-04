# README #

### Introduction ###
The first step in the TCE’s code synthesis process is the transformation of input equations into an equivalent form with minimal operation count.

Equations typically range from around ten to over a hundred terms, each involving the contraction of two or more tensors, and most quantum chemical methods involve two or more coupled equations of this type. One of our operation minimization algorithms focuses on the use of single-term optimization (strength reduction or parenthesization), which decomposes multi-tensor contraction operations into a sequence of binary contractions, coupled with a global search of the composite single-term solution space for factorization opportunities. Exhaustive search (for small cases) and a number of heuristics were shown to be effective in minimizing the operation count.

Common subexpression elimination (CSE) is a classical optimization technique used in traditional optimizing compilers to reduce the number of operations, where intermediates are identified that can be computed once and stored for use multiple times later. CSE is routinely used in the manual formulation of quantum chemical methods, but because of the complexity of the equations, it is extremely difficult to explore all possible formulations manually. CSE is a powerful technique that allows the exploration of the much larger algorithmic space than our previous approaches to operation minimization. However, the cost of the search itself grows explosively. We have developed an approach to CSE identification in the context of operation minimization for tensor contraction expressions. The developed approach is shown to be very effective, in that it automatically finds efficient computational forms for challenging tensor equations.

Quantum chemists have proposed domain-specific heuristics for strength reduction and factorization for specific forms of tensor contraction expressions. However, their work does not consider the general form of arbitrary tensor contraction expressions. Approaches to single-term optimizations and factorization of tensor contraction expressions were presented in [1](). Common subexpression identification to enhance single-term optimization is discussed in [2]().

### Requirement ###
Python version 2.3.4 or higher.

### Command Line Arguments ###

    % ./bin/opmin -h
    usage: opmin [options] INPUT_FILE
    description: Compile shell for operation minimization
    options:
        -h, --help          show this help message and exit
        -c, --commonsubexp  turn on common subexpression elimination
        -f, --factorize     turn on factorization

Note that you must use `–c` and `–f` if you want to use `common subexpression` and `factorization` algorithm.

### Input Files ###
Some input equations are available at `dense` directory.

### Output Files ###
After an input file has been run on OpMin, its output will be generated and written in a file of which name starts with underscore (‘_’).