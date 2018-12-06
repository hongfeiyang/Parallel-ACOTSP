// COMP90025 Project 2
// Hongfei Yang
// Oct 2018

// Notice:

// NUM_TOURS * NUM_COMMS * #procs is equal to the total number of iterations
// in a sequential algorithm

// To achieve near optimal solution, increase NUM_ANTS and NUM_TOURS

// DO NOT set number of threads per node beyond maximum number of threads per
// node otherwise performance may drop



#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


// constants
#define Q 80                // amount of pheromone deposited
#define RHO 0.5             // rate of pheromone evaporation
#define ALPHA 1.0           // appeal of pheromone on a path
#define BETA 2.0            // appeal of an edge

#define NUM_CITIES 131      // Number of cities
#define NUM_ANTS 100       // Number of ants
#define NUM_TOURS 30      // number of tours run before each communication
#define NUM_COMMS 2         // Number of communications limit
#define NUM_THREADS_PER_NODE 4 // number of threads per node, do not set this
                                // beyond maximum number of threads per CPU
                                // otherwise performance may drop
// ant
typedef struct {
    int city;
    int next_city;
    int visited[NUM_CITIES];
    int path[NUM_CITIES];
    int path_index; // number of cities in path
    double tour_distance;
} ant_t;

// city
typedef struct {
    int x;
    int y;
} city_t;

// a tour by an ant
typedef struct { 
    double distance;
    int path[NUM_CITIES];
} tour_t;


ant_t ant[NUM_ANTS]; // all ants
city_t city[NUM_CITIES]; // all cities
tour_t best; // local best
tour_t *all_best; //array of all best solutions gather by master

double distance[NUM_CITIES][NUM_CITIES], pheromone[NUM_CITIES][NUM_CITIES];
int rank, procs;
int best_ant_index;


int main(int argc, char * argv[]);
int choose_next_city(int ant_index);
double get_prob_product(int from, int to);
double get_inter_city_dist(int x1, int y1, int x2, int y2);
void build_best_tour_struct();
void reset_ants();
void update_best();
void read_from_input(char *filename);
void initalise_pheromone_matrix();


int main(int argc, char *argv[])
{
    int i, j, k, min_index;
    double start = 0.0, finish = 0.0;
    //MPI_Status status;
    MPI_Datatype MPI_CITY, MPI_BEST;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Capture the starting time
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    if (!rank) {
        read_from_input(argv[1]);
        printf("Cities: %d\nProcesses: %d\nAnts: %d\nAlpha: %3.2f\nBeta: %3.2f\nRho: %3.2f\nQ: %d\n\n", NUM_CITIES, procs, NUM_ANTS, ALPHA, BETA, RHO, Q);
        all_best = (tour_t *)malloc(sizeof(tour_t)*procs);
    }
    
    // broadcast all cities info to other processes
    MPI_Type_contiguous(2, MPI_INT, &MPI_CITY);
    MPI_Type_commit(&MPI_CITY);
    MPI_Bcast(city, NUM_CITIES, MPI_CITY, 0, MPI_COMM_WORLD);
    
    // initalise pheromone matrix from city list and ants
    initalise_pheromone_matrix();
    reset_ants();
    
    // build derived datatypes to communicate best tour 
    build_best_tour_struct(&best, &MPI_BEST);

    // limit the number of communications
    for(i=0; i<NUM_COMMS; i++) {

        // individual colony runs this many tours to optimise its local best
        // solution
        for (k=0; k<NUM_TOURS; k++) {
        
            // to complete a cycle of solution construction
            
            #pragma omp parallel num_threads(NUM_THREADS_PER_NODE) 
            {

                // move all ants to their next cities
                #pragma omp for private(j) schedule(dynamic,1)
                for(int ant_index=0; ant_index<NUM_ANTS; ant_index++) {

                    for(j=0; j<NUM_CITIES; j++) {

                        if(ant[ant_index].path_index < NUM_CITIES) {
                            // choose next city
                            ant[ant_index].next_city = choose_next_city(ant_index);

                            // move to next city
                            ant[ant_index].tour_distance += distance[ant[ant_index].city][ant[ant_index].next_city];
                            
                            ant[ant_index].path[ant[ant_index].path_index++] = ant[ant_index].next_city;
                            
                            ant[ant_index].visited[ant[ant_index].next_city] = 1;
                            ant[ant_index].city = ant[ant_index].next_city;
                            
                            if(ant[ant_index].path_index == NUM_CITIES) {
                                // this means an ant has completed a cycle, so travel
                                // to origin
                                
                                ant[ant_index].tour_distance += distance[ant[ant_index].path[NUM_CITIES-1]][ant[ant_index].path[0]];
                                
                            }
                        }
                    }
                }

                // pheromone updates
                int x, y, from, to;

                #pragma omp for private(x, y)
                // Evaporate pheromone
                for(x=0; x<NUM_CITIES; x++) {
                    for(y=0; y<NUM_CITIES; y++) {
                        if(x != y) {
                            
                            // no race conditions here, because two threads
                            // cannot have the same x and y
                            pheromone[x][y] *= 1.0-RHO;
                            if(pheromone[x][y] < 0.0) {
                                pheromone[x][y] = 1.0/NUM_CITIES;
                            }
                            
                        }
                    }
                }
                
                #pragma omp for private(x, y, from, to) schedule(dynamic,1)
                // Deposit pheromone
                for(x=0; x<NUM_ANTS; x++) {
                    for(y=0; y<NUM_CITIES; y++) {
                        from = ant[x].path[y];
                        
                        if(y < NUM_CITIES-1) to = ant[x].path[y+1];
                        else to = ant[x].path[0];
                        
                        // prevent race condition
                        #pragma omp critical
                        {
                            pheromone[from][to] += Q/ant[x].tour_distance;
                            pheromone[to][from] = pheromone[from][to];
                        }
                    }
                }
            }

            // find best
            update_best();

            // reset all ants
            reset_ants();
        }
        
        // Collect best tours from all processes
        MPI_Gather(&best, 1, MPI_BEST, all_best, 1, MPI_BEST, 0, MPI_COMM_WORLD);


        if(!rank) {
            // master computes the best global solution

            min_index = 0;
            for(j=1; j<procs; j++) {
                if(all_best[j].distance < all_best[min_index].distance) { 
                    min_index = j;
                }
            }

            // master updates best global solution
            best.distance = all_best[min_index].distance;
            for(j=0; j<NUM_CITIES; j++) {
                best.path[j] = all_best[min_index].path[j];
            }
        }
        

        if (i < NUM_COMMS - 1) {
        // avoid the last unnecessary broadcast

            // master broadcast global best to other processes
            // all processes receive global best
            MPI_Bcast(&best, 1, MPI_BEST, 0, MPI_COMM_WORLD);

            // all processes reset pheromone matrix
            for(j=0; j<NUM_CITIES; j++) {
                for(k=0; k<NUM_CITIES; k++) {
                    pheromone[j][k] = 1.0/NUM_CITIES;
                }
            }

            // each process reinforces best path in its local pheromone matrix
            for(j=0; j<NUM_CITIES; j++) {
                if(j < NUM_CITIES-1) {
                    pheromone[best.path[j]][best.path[j+1]] += Q/best.distance;
                    pheromone[best.path[j+1]][best.path[j]] = pheromone[best.path[j]][best.path[j+1]];
                }
            }
        }
        
    }
    
    // Capture the ending time
    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    if(!rank) {
        printf("Best Tour (%.15f): %.15f\n", finish-start, best.distance);

    }
    
    free(all_best);
    MPI_Type_free(&MPI_CITY);
    MPI_Type_free(&MPI_BEST);
    MPI_Finalize();


    return 0;
}


// build a MPI data type mpi_type which is used for communicating a tour between
// nodes
void build_best_tour_struct(tour_t *tour, MPI_Datatype *mpi_type)
{
    int block_lengths[2];
    MPI_Aint displacements[3];
    MPI_Datatype types[3];
    
    block_lengths[0] = 1;
    block_lengths[1] = NUM_CITIES;

    displacements[0] = (size_t)&(best.distance) - (size_t)&best;
    displacements[1] = (size_t)&(best.path) - (size_t)&best;

    types[0] = MPI_DOUBLE;
    types[1] = MPI_INT;
    
    
    MPI_Type_create_struct(2, block_lengths, displacements, types, mpi_type);
    MPI_Type_commit(mpi_type);
}



// reset each ant's location to randomly distribute them
void reset_ants()
{
    int i, j, uniform = 0;
    
    for(i=0; i<NUM_ANTS; i++) {
        if(uniform == NUM_CITIES) uniform = 0;
        ant[i].city = uniform;
        ant[i].path_index = 1;
        ant[i].tour_distance = 0.0;
        
        for(j=0; j<NUM_CITIES; j++) {
            ant[i].visited[j] = 0;
            ant[i].path[j] = -1;
        }
        ant[i].visited[ant[i].city] = 1;
        ant[i].path[0] = ant[i].city;
        
        uniform++;
    }
}

// return the next city index given an ant index
// referenced from Brette B. (brett@intraspirit.net) for this calculation part
int choose_next_city(int ant_index)
{
    double denominator = 0.0, c = 0.0, r;
    int i;
    struct timeval time;
    
    gettimeofday(&time, 0);


    srandom((int)(time.tv_usec * 1000000 + time.tv_sec)+rank);
    // srandom((int)rank);
    r = (double)random()/(double)RAND_MAX;
    
    for(i=0; i<NUM_CITIES; i++) {
        if(!ant[ant_index].visited[i]) {

            #pragma omp critical
            {
                denominator += get_prob_product(ant[ant_index].city, i);
            }
        }
    }
    
    if(denominator != 0.0) {    
        for(i=0; i<NUM_CITIES; i++) {
            if(!ant[ant_index].visited[i]) {
                #pragma omp critical
                {
                    c += get_prob_product(ant[ant_index].city, i)/denominator;
                }
                if(r <= c) break;
            }
        }

        return i;
    } else {
        return -1;
    }
}

// find best solution after all ants completed their tour once.
// store best solution in a global variable
void update_best()
{
    int i, j;
    
    for(i=0; i<NUM_ANTS; i++) {
        if(ant[i].tour_distance < best.distance || best.distance == 0.0) {
            best.distance = ant[i].tour_distance;
            for(j=0; j<NUM_CITIES; j++) best.path[j] = ant[i].path[j];
        }
    }
}



// read cities info from file
void read_from_input(char *filename)
{
    FILE *fp;
    int i;
    int city_index;
    fp = fopen(filename, "r");

    for(i=0; i<NUM_CITIES; i++) {
        fscanf(fp, "%d %d %d", &city_index, &city[i].x, &city[i].y);
    }
    
    fclose(fp);
}

// initialise pheromone matrix to be 1/number of cities for all edges
void initalise_pheromone_matrix()
{
    int i, j;
    
    for(i=0; i<NUM_CITIES; i++) {
        for(j=0; j<NUM_CITIES; j++) {
            distance[i][j] = 0.0;
            if(i != j) {
                distance[i][j] = distance[j][i] = get_inter_city_dist(city[i].x, city[i].y, city[j].x, city[j].y);
            }
            pheromone[i][j] = 1.0/NUM_CITIES;
        }
    }
}


// calculates the pheromone/distance product for use in the ACO probability function
// referenced from Brett C. Buddin (brett@intraspirit.net)
double get_prob_product(int from, int to)
{
    return pow(pheromone[from][to], ALPHA) * pow((1.0/distance[from][to]), BETA);
}

//Calculates the distance between two cities
double get_inter_city_dist(int x1, int y1, int x2, int y2)
{
    return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

