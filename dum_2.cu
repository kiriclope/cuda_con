#include <stdio.h> 
#include <math.h> 

#define N_NEURONS (unsigned long) 20000 
#define K (double) 2000 
#define sqrt_K (double) sqrt(K)

#define N_NEURONS_PER_CHUNCK (unsigned long) 10000 
#define N_CHUNCKS N_NEURONS / N_NEURONS_PER_CHUNCK // otherwise memory problem if N_NEURONS > 20000 because of array[CHUNCK_SIZE] too large 
#define CHUNCK_SIZE N_NEURONS * N_NEURONS_PER_CHUNCK 

int main(int argc, char *argv[]) {
  printf("N_NEURONS %lu K %.0f sqrt(K) %.2f \n", N_NEURONS, K, sqrt_K) ; 
  printf("N_CHUNCKS %lu N_NEURONS_PER_CHUNCK %lu CHUNCK_SIZE %lu \n", N_CHUNCKS, N_NEURONS_PER_CHUNCK, CHUNCK_SIZE) ; 
}