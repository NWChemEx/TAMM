CC = gcc
CFLAGS = -g 

all: tcemain

tcemain: main.o parser.o scanner.o absyn.o visitor.o semant.o
	$(CC) -o $@ $(CFLAGS) main.o parser.o scanner.o visitor.o semant.o absyn.o -lfl 

main.o: main.c parser.o scanner.o absyn.o *.h
	$(CC) $(CFLAGS) -c main.c

visitor.o: visitor.c parser.o scanner.o absyn.o absyn.h
	$(CC) $(CFLAGS) -c visitor.c 
	
semant.o: semant.c visitor.h parser.o scanner.o absyn.o absyn.h
	$(CC) $(CFLAGS) -c semant.c 
		
absyn.o: absyn.c absyn.h
	$(CC) $(CFLAGS) -c absyn.c

parser.o: parser.c absyn.o absyn.h 
	$(CC) $(CFLAGS) -c parser.c
	
parser.c: parser.y lemon
	./lemon -s parser.y

parser.h: parser.c scanner.h
	echo "parser.h was created along with parser.c"

scanner.o: scanner.c scanner.h
	$(CC) $(CFLAGS) -c scanner.c
	
lemon: lemon.c
	${CC} lemon.c -o lemon

scanner.c: scanner.l
	flex scanner.l

.PHONY: clean

clean: 
	rm -f parser.out parser.c parser.h scanner.c scanner.h tcemain lemon *.o output.txt lex.yy.c
