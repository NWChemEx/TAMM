Validation for Tensor Declaration:

0- if there is dependent label, the number of labels in parenthesis should be equal to the number of dependent index spaces 
    // T(a(i,j),i,j) a should be created out of 2 independent indexspaces.
1- each label name in a tensor construction should be unique // T(i,i) - not allowed where i is a TiledIndexLabel
2- every label used as a dependent label should be in the tensor construction //handled in dep_map construction
3- should has a topological order for labels where all dependent labels (labels within parenthesis) comes before
