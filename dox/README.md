Base Documentation Repository for NWChemEx
==========================================

Doxygen supports numerous customization features (take a look at the Doxyfile in
this repo for a complete list).  The point of this repo is to provide a source 
of the NWChemEx project's preferred settings so as to generate uniform
documentation.   

How To Use
----------

:memo: If you ar trying to build documentation for this repository locally, all
paths assume you call Doxygen from the root of your repo, *i.e.* run:
~~~.git
doxygen dox/Doxyfile
~~~

:warning: This repo has only been tested for manually building the
documentation.  CodeDocs offers a means of recursively linking documentations 
together, but can only operate on public repositories.  Until NWChemEx is public
the last two steps will not work.

0. "Install" [git subrepo](https://github.com/ingydotnet/git-subrepo) if you
haven't already (it's just a bash script)
1. In the top-level directory of your project run:
~~~git
git subrepo clone https://github.com/NWChemEx-Project/dox
~~~
2. Modify `dox/settings.doxcfg` to be descriptive of your project.
3. (NYI) Create a file `.codedocs` in the top directory of your repository which
contains the lines:
~~~.sh
DOXYFILE = dox/Doxyfile
TAGLINKS = owner/repo
~~~
The first line will be the same for all repos.  The second line may be blank or
have one or more (separated by spaces) GitHub repos that you would like to
include in your documentation where `owner` refers to who owns the GitHub repo 
and `repo` is the name of the repo (the repo must also be on CodeDocs).
4. (NYI) Enable auto-building of your documentation on 
   [CodeDocs](https://codedocs.xyz)
   
### Technical Aside
Owing to the layout of the GitHub repositories the documentation will have to be
built recursively; however, in Doxygen, having multiple mainpages is undefined 
behavior.  We thus have to be a bit creative to avoid this limitation.  What
this amounts to is we expect each repo to override the stub page `mainpage.md`