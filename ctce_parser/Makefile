CC = gcc
CFLAGS = -g -Wall

all: tcemain

tcemain: main.o parser.o scanner.o absyn.o visitor.o semant.o symtab.o util.o 
	$(CC) -o $@ $(CFLAGS) main.o parser.o scanner.o visitor.o semant.o absyn.o symtab.o util.o -lfl 

main.o: main.c parser.o scanner.o *.h
	$(CC) $(CFLAGS) -c main.c

visitor.o: visitor.c parser.o 
	$(CC) $(CFLAGS) -c visitor.c 
	
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
	rm -f parser.out parser.c parser.h scanner.c scanner.h tcemain lemon *.o output.txt lex.yy.c
