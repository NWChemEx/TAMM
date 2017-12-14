DEVELOPER NOTES
---------------

Code Formatting / Refactoring
-----------------------------

Using clang-tidy and clang-format to refactor/format the code

Use -DCMAKE_EXPORT_COMPILE_COMMANDS=ON when running cmake,
then use clang-tidy as follows:

clang-tidy -p $tamm_root/build/ tensor.cc
clang-tidy -checks="*" -list-checks -header-filter=".*" -p ../build/  tensor.cc

clang-tidy  -checks='-*,modernize-*,mpi-*' -p ../build/  tensor.cc
clang-tidy  -checks='-*,modernize-*,cppcoreguidelines-*,mpi-*' -p ../build/  tensor.cc
clang-tidy  -checks='*,-clang-analyzer-*,-cppcoreguidelines-*' -p ../build/  tensor.cc



This project tries to keep code style conistent:
For vim a .lvimrc is contained here that will keep tabing compliant

Please Install: https://raw.githubusercontent.com/embear/vim-localvimrc/master/plugin/localvimrc.vim to your .vim/plugins directory

Which will auto source this file


Input format:

* Add label to equation block i.e { ... } => label1 { ... }

* Tensor labels: i[1-9][0-9]+ are intermediates. All else are inputs ([txvf][0-9]+), including i0

* Sum() and P() can be removed

* For inputs, irrep is in the suffix

* Contraction labeling: parts
  - File name (t1_, t2_, t3_, ..)
  - At the top-level (indent 0), all contractions are labeled 1_, 2_, ..
  - Within a level (e.g., t2_3), all contractions at the immediate enclosed level are labeled 1_, 2_,..
  - Intermediate tensor name: name of contraction that first defines it (always at the top (_1) of the list of contractions that define it)

* Replace use of intermediate with their unique name/labels

* Array creation: as late as possible
* Array destruction: as early as possible




Clarifications:

* Additions of tensors with different irreps


Assumptions:

- range of h1..hn: O and p1..pn: V

- indices in array reference are always h[1-9]+ or p[1-9]+

- statement label and function name prefixes come from file name.

Parser
------
- If O,V are specified in the equation file, how are they used in tamm ? - Currently they are ignored.
- Intermediate.cc constructs rangeEntry with only range name, not values provided.
- c += alpha*a*b, c += alpha*a


Unfactorized equations Test
----------------------------

- ccsd_t2_hand.eq cannot be Unfactorized

- icsd unfactorizes okay, but does not work - something to do with tamm and NW fortran code

- ipccsd_x1 and x2 do not work - check if they unfactorize okay? original eqs work  
- eaccsd_x1 and x2 do not work - mostly due to uneven indices ?

- and add exp i0[] = a1[] + a2[] does not work --> tamm iterator cannot handle AddOp involving scalars (assert ndim>0 in tensors_and_ops.cc)
  - for this reason, we do not include unfactorized ccsd_e,cisd_e,and small gamma eqns - pp,hh,hp - since they are 1-2 ops only, they do not matter anyway


ANTLR
-----

 java -jar /opt/libraries/ANTLR4/antlr-4.7-complete.jar TAMM.g4  -Dlanguage=Cpp -visitor -no-listener

uuid not found on clusters ??

Need gcc > 5 . We used 6 for c++14 support

TESTING
--------
https://scan.coverity.com/

Catch, GoogleTest (Bandit, Lest)

 g++ test1.cpp -I/opt/tools/googletest/include -L/opt/tools/googletest/lib -lgtest -lpthread

g++ ParserTest.cc -I../../frontend/ -L/home/panyala/EclipseWS/workspacePTP/tamm/build/ -ltamm  -I/opt/libraries/ANTLR4/antlr4-cpp-runtime/include/antlr4-runtime/ -L/opt/libraries/ANTLR4/antlr4-cpp-runtime/lib -I/opt/tools/googletest/include -L/opt/tools/googletest/lib -lgtest -lpthread -lantlr4-runtime

https://www.google.com/search?q=MyErrorStrategy&oq=MyErrorStrategy&aqs=chrome..69i57&sourceid=chrome&ie=UTF-8


ECLIPSE
========
Download Eclipse for Parallel Application Developers
http://www.eclipse.org/downloads/packages/release/Neon/3

Plugins to install from marketplace
cmake editor - use cmakeed (only editor) - use update site for installation
CMAKE Builder - https://marketplace.eclipse.org/content/cmake4eclipse
Python - PyDev
CPPStyle - http://www.cppstyle.com/
Instructions are here - https://github.com/wangzw/CppStyle

Project->properties->C/C++ General->Paths and Symbol

brew install clang-format

###Remote System Explorer (RSE) - remote project with Eclipse via SSH

Window > Open Perspective >  Remote System Explorer

To create an SSH remote project from the RSE perspective in Eclipse:

Define a new connection and choose SSH Only from the Select Remote System Type screen in the New Connection dialog.
Enter the connection information then choose Finish.
Connect to the new host. (Assumes SSH keys are already setup.)
Once connected, drill down into the host's Sftp Files, choose a folder and select Create Remote Project from the item's context menu.

You should now see a new remote project accessible from the Project Explorer.


#Other eclipse stuff

#Get TM Terminal from eclipse marketplace.

This should be enough for eclipse to find all standard libs,includes,etc

C/C++ Build -> Settings -> Tool Settings -> GCC C++ Compiler -> Miscellaneous -> Other Flags.
Put  -std=c++14 at the end . ... instead of GCC C++ Compiler

#To enable c++14 only for a selected project:
Right click on your project and click Properties
Navigate to C/C++ General and Preprocessor Include Paths, Macros etc.
Select the Providers tab, click on compiler settings row for the compiler you use.
Add -std=c++14 to Command to get compiler specs. Should look something like:

${COMMAND} -E -P -v -dD "${INPUTS}" -std=c++14
