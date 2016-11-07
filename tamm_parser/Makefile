CC = g++
LCC = gcc
CFLAGS = -g -Wall

all: libtammparser.a tamm_parser_test

tamm_parser_test: test.o libtammparser.a
	$(CC) -o $@ $(CFLAGS) test.o -L. -ltammparser -lfl

test.o: test.cc libtammparser.a
	$(CC) $(CFLAGS) -c test.cc


libtammparser.a: tamm_parser.o parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o intermediate.o
	ar -cvq libtammparser.a parser.o scanner.o visitor.o semant.o absyn.o symtab.o util.o intermediate.o tamm_parser.o

tamm_parser.o: parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o intermediate.o
	$(CC) $(CFLAGS) -c tamm_parser.cc

visitor.o: visitor.cc parser.o util.o
	$(CC) $(CFLAGS) -c visitor.cc 

intermediate.o: intermediate.cc semant.o
	$(CC) $(CFLAGS) -c intermediate.cc

semant.o: semant.cc parser.o
	$(CC) $(CFLAGS) -c semant.cc 

util.o: util.cc 
	$(CC) $(CFLAGS) -c util.cc
			
absyn.o: absyn.cc util.o
	$(CC) $(CFLAGS) -c absyn.cc

symtab.o: symtab.cc util.o 
	$(CC) $(CFLAGS) -c symtab.cc 
		
parser.o: parser.cc absyn.o absyn.h 
	$(CC) $(CFLAGS) -c parser.cc 
	
parser.cc: parser.y lemon
	./lemon -s parser.y && mv parser.c parser.cc

scanner.o: scanner.cc scanner.h
	$(CC) $(CFLAGS) -c scanner.cc
	
lemon: lemon.c
	${LCC} lemon.c -o lemon

scanner.cc: scanner.l
	flex scanner.l && mv scanner.c scanner.cc

.PHONY: clean

clean: 
	rm -f tamm_parser_test parser.out parser.cc parser.h scanner.cc scanner.h lemon *.o libtammparser.a output.txt lex.yy.cc
