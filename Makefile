## Makefile for project 1b
## Hongfei Yang 
## September 2018

CC = mpicc

CFLAGS = -Wall -std=gnu99 -fopenmp

EXE2 = a2
OBJ2 = a2.o

CORES = 2

## sample input for test
FILE = city1.txt
FILE1 = att48_xy.txt
FILE2 = xqf131.tsp.txt
FILE3 = xql662.tsp.txt


$(EXE2): $(OBJ2)
	$(CC) $(CFLAGS) -o $(EXE2) $(OBJ2)



## MPI by core test
mpi_test:
	mpirun -n $(CORES) ./$(EXE) $(FILE2)

seq_test:
	./$(EXE3) $(FILE2)

acotsp_test:
	mpirun -n $(CORES) ./$(EXE2) $(FILE2)


clean:
	rm -f $(OBJ2) $(EXE2)