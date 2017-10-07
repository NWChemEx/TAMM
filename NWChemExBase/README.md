NWChemEx Base Repository
==============================

The purpose of this repository is to store a crisp, clean version of the CMake
build infrastructure common to the NWChemEx project in such a manner that it:

1. avoids copy/pasting,
2. facilitates uniform maintable builds for all dependencies,
3. is cutomizable on a per dependency basis

If you are building a library that is meant to provide core NWChemEx
functionality it is strongly recommended you use this repository to save
yourself a lot of headaches in putting together a build.

Conventions
-----------

For the purposes of this documentation assume your project is called `MyRepo`.
The infrastructure contained in the `NWChemExBase` repository will be called
`NWChemExBase`.

:memo: CMake is case-sensitive so the capitalization of your project matters. To
that end in the following `MyRepo` is the case used by your project, `MYREPO`
is the name you provided, but in all upper-case letters, and `myrepo` is the
name you provided, but in all lower-case letters.  When needed, the all
upper-/lower- case version of your project's name will be computed automatically
by `NWChemExBase`.


What This Repository Provides
-----------------------------

By using this repository (and adhering to its standards) your library will have
a robust build that ensures its compatability with other parts of the NWChemEx
ecosystem.  Additionally, this repository will ensure:

1. A library named `MyRepo.so` is built (other extensions and the ability for
   static libraries will be added later).
2. `MyRepo.so` will be testable from the build directory via CMake's `ctest`
      command.
3. A file `MyRepoConfig.cmake` will automatically be created for you using the
   build settings you provided.
4. `MyRepo.so` will be installable from the build directory along with the
   header files representing its public API and the aforementined CMake file.
5. Your library is locatable and includable by other CMake projects via CMake's
   `find_package` mechanism (assuming the resting place of `XXXConfig.cmake` is
   included in `CMAKE_PREFIX_PATH`).
6. Because `NWChemExBase` takes care of 99.9% of the build for you, we
   went ahead and wrote your build documentation.  You're welcome.  Just link to
   `NWChemExBase/dox/Building.md` in your documentation and take all the credit
   for some of the best build documentation around.

How To Use
-----------

0. "Install" [git subrepo](https://github.com/ingydotnet/git-subrepo) if you
   haven't already (it's just a bash script)
1. In the top-level directory of your project run:
   ~~~git
   git subrepo clone https://github.com/NWChemEx-Project/NWChemExBase
   ~~~
2. In the same directory run:
   ~~~bash
   ./NWChemExBase/bin/BasicSetup.sh <MyRepo>
   ~~~
3. Add your dependencies to `CMakeLists.txt`
4. Fill in the source files and public headers of your library in
   `MyRepo/CMakeLists.txt`
5. Add your tests to `MyRepo-Test/CMakeLists.txt`
