CC = gcc
CFLAGS = -g -Wall

all: tamm_test libtammparser.a

tamm_test: test.o parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o intermediate.o tamm_vector.o tamm_parser.o
	$(CC) -o $@ $(CFLAGS) test.o parser.o scanner.o visitor.o semant.o absyn.o symtab.o util.o intermediate.o tamm_vector.o tamm_parser.o -lfl

test.o: test.c parser.o scanner.o *.h
	$(CC) $(CFLAGS) -c test.c


libtammparser.a: tamm_parser.o parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o intermediate.o tamm_vector.o
	ar -cvq libtammparser.a tamm_parser.o parser.o scanner.o visitor.o semant.o absyn.o symtab.o util.o intermediate.o tamm_vector.o

tamm_parser.o: parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o intermediate.o tamm_vector.o
	$(CC) $(CFLAGS) -c tamm_parser.c

visitor.o: visitor.c parser.o util.o
	$(CC) $(CFLAGS) -c visitor.c 

intermediate.o: intermediate.c semant.o tamm_vector.o
	$(CC) $(CFLAGS) -c intermediate.c

tamm_vector.o: tamm_vector.c  util.o
	$(CC) $(CFLAGS) -c tamm_vector.c

semant.o: semant.c parser.o  
	$(CC) $(CFLAGS) -c semant.c 

util.o: util.c 
	$(CC) $(CFLAGS) -c util.c
			
absyn.o: absyn.c util.o
	$(CC) $(CFLAGS) -c absyn.c

symtab.o: symtab.c util.o 
	$(CC) $(CFLAGS) -c symtab.c 
		
parser.o: parser.c absyn.o absyn.h 
	$(CC) $(CFLAGS) -c parser.c 
	
parser.c: parser.y lemon 
	./lemon -s parser.y

scanner.o: scanner.c scanner.h
	$(CC) $(CFLAGS) -c scanner.c
	
lemon: lemon.c
	${CC} lemon.c -o lemon

scanner.c: scanner.l
	flex scanner.l 

.PHONY: clean

clean: 
	rm -f tamm_test parser.out parser.c parser.h scanner.c scanner.h lemon *.o libtammparser.a output.txt lex.yy.c
