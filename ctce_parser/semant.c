#include "semant.h"

void check_ast(TranslationUnit root, SymbolTable symtab){
  CompoundElemList celist = root->celist;
  while(celist != NULL){
    check_CompoundElem(celist->head,symtab);
    celist = celist->tail;
  }
  celist = NULL;
}

void check_CompoundElem(CompoundElem celem, SymbolTable symtab){
  ElemList elist = celem->elist;
  while(elist != NULL){
    check_Elem( elist->head,symtab);
    elist = elist->tail;
  }
  elist = NULL;
}

void check_Elem(Elem elem, SymbolTable symtab){
  Elem e = elem;
  if(e==NULL) return;

  switch(e->kind) {
  case is_DeclList:
    check_DeclList(elem->u.d,symtab);
    break;
  case is_Statement:
    check_Stmt(e->u.s,symtab);
    break;
  default:
    fprintf(stderr,"Not a Declaration or Statement!\n");
    exit(0);
  }
}

void check_DeclList(DeclList decllist, SymbolTable symtab){
  DeclList dl = decllist;
  while(dl!=NULL){
    check_Decl(dl->head,symtab);
    dl = dl->tail;
  }
}

void verifyVarDecl(SymbolTable symtab, string name, int line_no){
    if (ST_contains(symtab,name)){
        fprintf(stderr,"Error at line %d: %s is already defined\n", line_no, name);
        //exit(2);
    }
}

void verifyRangeRef(SymbolTable symtab, string name, int line_no){
    if (!ST_contains(symtab,name)){
        fprintf(stderr,"Error at line %d: range variable %s is not defined\n", line_no, name);
        exit(2);
    }
}

void check_Decl(Decl d, SymbolTable symtab){
  switch(d->kind) {
  case is_RangeDecl:
  	verifyVarDecl(symtab, d->u.RangeDecl.name, 0);
    ST_insert(symtab,d->u.RangeDecl.name,int_str(d->u.RangeDecl.value));
    break;
  case is_IndexDecl:
    verifyVarDecl(symtab, d->u.IndexDecl.name, 0);
    verifyRangeRef(symtab, d->u.IndexDecl.rangeID,0);
    ST_insert(symtab,d->u.IndexDecl.name,d->u.IndexDecl.rangeID);
    break;
  case is_ArrayDecl:
  	verifyVarDecl(symtab, d->u.ArrayDecl.name, 0);
  	string comb_index_list = combine_indexLists(d->u.ArrayDecl.upperIndices,d->u.ArrayDecl.ulen,d->u.ArrayDecl.lowerIndices,d->u.ArrayDecl.llen);
  	//printf("%s -> %s\n", d->u.ArrayDecl.name, comb_index_list);
  	tce_string_array ind_list = stringToList(comb_index_list);
  	int i=0;
  	for (i=0;i<ind_list->length;i++) verifyRangeRef(symtab,ind_list->list[i],0);
  	ST_insert(symtab,d->u.ArrayDecl.name,comb_index_list);
    break;
  default:
    fprintf(stderr,"Not a valid Declaration!\n");
    exit(0);
  }
}

void check_Stmt(Stmt s, SymbolTable symtab){
  switch(s->kind) {
  case is_AssignStmt:
    check_Exp( s->u.AssignStmt.lhs, symtab);
    //printf(" %s ", s->u.AssignStmt.astype); //astype not needed since we flatten. keep it for now.
    check_Exp( s->u.AssignStmt.rhs, symtab);
    if (s->u.AssignStmt.lhs->kind != is_ArrayRef){
    	fprintf(stderr,"Error at line %d: LHS of assignment must be an array reference\n",0);
			//exit(2);
    }
    else if(s->u.AssignStmt.lhs->coef < 0)
    	fprintf(stderr,"Error at line %d: LHS array reference cannot be negative\n",0);

//    UNCOMMENT FOR DEBUG ONLY
//    print_index_list(getIndices(s->u.AssignStmt.lhs));
//    printf(" = ");
//    print_index_list(getIndices(s->u.AssignStmt.rhs));
//    printf("\n");
    if(!compare_index_lists(getIndices(s->u.AssignStmt.lhs),getIndices(s->u.AssignStmt.rhs))){
    	fprintf(stderr,"Error at line %d: LHS and RHS of assignment must have equal (non-summation) index sets\n",0);
    }
    break;
  default:
    fprintf(stderr,"Not an Assignment Statement!\n");
    exit(0);
  }
}


void check_ExpList(ExpList expList, SymbolTable symtab){
  ExpList elist = expList;
  while(elist != NULL){
    check_Exp( elist->head, symtab);
    elist = elist->tail;
  }
  elist = NULL;
}


void verifyArrayRefName(SymbolTable symtab, string name, int line_no){
  if (!ST_contains(symtab,name)){
      fprintf(stderr,"Error at line %d: array %s is not defined\n", line_no, name);
     // exit(2);
  }
}

void verifyIndexRef(SymbolTable symtab, string name, int line_no ){
  if (!ST_contains(symtab,name)){
      fprintf(stderr,"Error at line %d: index %s is not defined\n", line_no, name);
      //exit(2);
  }
}

void verifyArrayRef(SymbolTable symtab, string name, string *inds, int len, int line_no ){
    verifyArrayRefName(symtab, name, line_no);
    int i = 0;
    for (i=0;i<len;i++) verifyIndexRef(symtab, inds[i], 0);
}


void check_Exp(Exp exp, SymbolTable symtab){
	tce_string_array inames = NULL;
	ExpList el = NULL;
  switch(exp->kind) {
  case is_Parenth:
    check_Exp(exp->u.Parenth.exp,symtab);
    break;
  case is_NumConst:
    //printf("%f ",exp->u.NumConst.value);
    break;
  case is_ArrayRef:
     verifyArrayRef(symtab,exp->u.Array.name,exp->u.Array.indices,exp->u.Array.length,0);
     inames = getIndices(exp);
   	 int tot_len1 = inames->length;
     string* all_ind1 = inames->list;
     int i1=0, ui1=0;
     string* rnames = tce_malloc(sizeof(string) * tot_len1);

     for(i1=0;i1<tot_len1;i1++){
    	 rnames[i1] = ST_get(symtab,all_ind1[i1]);
     }

     tce_string_array rnamesarr = tce_malloc(sizeof(*rnamesarr));
     rnamesarr->list = rnames;
     rnamesarr->length = tot_len1;
     string ulranges = ST_get(symtab,exp->u.Array.name);
     tce_string_array ulr = stringToList(ulranges);

     if(!compare_index_lists(ulr,rnamesarr)){
        fprintf(stderr,"Error at line %d: array reference %s[%s] must have index structure of %s[%s]\n",0,
        		exp->u.Array.name,combine_indices(all_ind1,tot_len1),exp->u.Array.name,combine_indices(ulr->list,ulr->length));
     }
     //Check for repeatetive indices in an array reference
   	string* uind1 = tce_malloc(sizeof(string) * tot_len1);

     i1=0, ui1=0;
     for (i1=0;i1<tot_len1;i1++){
     	if(!exists_index(uind1,ui1,all_ind1[i1])) {
     		uind1[ui1] = all_ind1[i1];
     		ui1++;
     	}
     }

   	for (i1=0;i1<ui1;i1++){
   		if(count_index(all_ind1,tot_len1,uind1[i1]) > 1) {
   			fprintf(stderr,"Error at line %d: indistinct index %s must not exist in array reference %s[%s]\n",
   					   0,uind1[i1],exp->u.Array.name,combine_indices(exp->u.Array.indices,exp->u.Array.length));
   		}
   	}

    break;
  case is_Addition:
    check_ExpList(exp->u.Addition.subexps,symtab);
    inames = getIndices(exp);
    el = exp->u.Addition.subexps;
  	while(el!=NULL){
  		tce_string_array op_inames = getIndices(el->head);
  		if(!compare_index_lists(inames,op_inames)){
  			fprintf(stderr,"Error at line %d: subexpressions of an addition must have equal index sets\n", 0);
  		}
  		op_inames = NULL;
  		el = el->tail;
  	}
    break;
  case is_Multiplication:
    check_ExpList(exp->u.Multiplication.subexps,symtab);

  	el = exp->u.Multiplication.subexps;
  	int tot_len = 0;
  	while(el!=NULL){
  		//print_Exp(el->head);
  		tce_string_array se = getIndices(el->head);
  		if(se!=NULL) tot_len += se->length;
  		se = NULL;
  		el = el->tail;
  	}

  	el =  exp->u.Multiplication.subexps;
  	string* all_ind = tce_malloc(sizeof(string) * tot_len);

  	int i=0, ui = 0;
  	while(el!=NULL){
  		tce_string_array se = getIndices(el->head);
  		i=0;
  		if(se!=NULL){
  			for(i=0;i<se->length;i++){
  				all_ind[ui] = se->list[i];
  				ui++;
  			}
  		}
  		se = NULL;
  		el = el->tail;
  	}
  	assert(ui==tot_len);
  	string* uind = tce_malloc(sizeof(string) * tot_len);

    i=0, ui=0;
    for (i=0;i<tot_len;i++){
    	if(!exists_index(uind,ui,all_ind[i])) {
    		uind[ui] = all_ind[i];
    		ui++;
    	}
    }

  	for (i=0;i<ui;i++){
  		if(count_index(all_ind,tot_len,uind[i]) > 2) {
  			fprintf(stderr,"Error at line %d: summation index %s must occur exactly twice in a multiplication\n", 0,uind[i]);
  		}
  	}

    break;
  default:
    fprintf(stderr,"Not a valid Expression!\n");
    exit(0);
  }
}




tce_string_array getIndices(Exp exp){
	ExpList el = NULL;
	tce_string_array p = NULL;
  switch(exp->kind) {
  case is_Parenth:
    return getIndices(exp->u.Parenth.exp);
    break;
  case is_NumConst:
  	return NULL;
    break;
  case is_ArrayRef:
		p = tce_malloc(sizeof(*p));
		p->list = replicate_indices(exp->u.Array.indices,exp->u.Array.length);
		p->length = exp->u.Array.length;
		return p;
    break;
  case is_Addition:
    return getIndices(exp->u.Addition.subexps->head);
    break;
  case is_Multiplication:
  	el = exp->u.Multiplication.subexps;
  	int tot_len = 0;
  	while(el!=NULL){
  		//print_Exp(el->head);
  		tce_string_array se = getIndices(el->head);
  		if(se!=NULL) tot_len += se->length;
  		se = NULL;
  		el = el->tail;
  	}

  	el =  exp->u.Multiplication.subexps;
  	string* all_ind = tce_malloc(sizeof(string) * tot_len);

  	int i=0, ui = 0;
  	while(el!=NULL){
  		tce_string_array se = getIndices(el->head);
  		i=0;
  		if(se!=NULL){
  			for(i=0;i<se->length;i++){
  				all_ind[ui] = se->list[i];
  				ui++;
  			}
  		}
  		se = NULL;
  		el = el->tail;
  	}
  	assert(ui==tot_len);
  	string* uind = tce_malloc(sizeof(string) * tot_len);

    i=0, ui=0;
//  	for (i=0;i<tot_len;i++){
//  		if(!exists_index(uind,ui,all_ind[i])) {
//  			uind[ui] = all_ind[i];
//  			ui++;
//  		}
//  	}

  	for (i=0;i<tot_len;i++){
  		if(count_index(all_ind,tot_len,all_ind[i]) == 1) {
        uind[ui]=all_ind[i];
  			ui++;
  		}
  	}

  	string* uniq_ind = tce_malloc(sizeof(string) * ui);
  	for (i=0;i<ui;i++) uniq_ind[i] = strdup(uind[i]);

//  	free(all_ind);
//  	all_ind = NULL;
    //uind = NULL;

		p = tce_malloc(sizeof(*p));
		p->list = uniq_ind;
		p->length = ui;

    return p;
    break;
  default:
    fprintf(stderr,"Not a valid Expression!\n");
    exit(0);
  }
}


void print_ExpList(ExpList expList, string am){
  ExpList elist = expList;
  while(elist != NULL){
    print_Exp(elist->head);
    elist = elist->tail;
    if(elist!=NULL) printf("%s ",am);
  }
  elist = NULL;
}

void print_Exp(Exp exp){
  switch(exp->kind) {
  case is_Parenth:
    print_Exp(exp->u.Parenth.exp);
    break;
  case is_NumConst:
    printf("%f ",exp->u.NumConst.value);
    break;
  case is_ArrayRef:
    printf("%s[%s] ",exp->u.Array.name,combine_indices(exp->u.Array.indices,exp->u.Array.length));
    break;
  case is_Addition:
    print_ExpList(exp->u.Addition.subexps,"+");
    break;
  case is_Multiplication:
    print_ExpList(exp->u.Multiplication.subexps,"*");
    break;
  default:
    fprintf(stderr,"Not a valid Expression!\n");
    exit(0);
  }
}



