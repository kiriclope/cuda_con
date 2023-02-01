#ifndef _NEURON_COUNTS 
#define _NEURON_COUNTS 

#define N_THREADS 512

#define n_pop 1 // number of populations 
#define N_NEURONS 10000ULL // total number of neurons 
#define nbPref 10000.0 
#define K 500. // average number of connections

#define popSize 1.0 // proportion of excitatory neurons
#define IF_Nk 0 // different number of neurons in each pop then fix popSize 

#define IF_PRES 1 

#define IF_CHUNKS 0
#define NCHUNKS 4 
#define MAXNEURONS 19200ULL 

// constants for structured matrix
const double Sigma[4] = {0.125,.075,.125,.075} ; 
#define IF_Dij 0 
const double Dij[16] ={1.0,1.0,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.,1.,1.,1.,1.,1.};

#define L 3.0 // M_PI // size of the ring 
#define IF_RING 0 // standard ring with cosine interactions 
#define IF_SPACE 0 // Spatially structured connections 
#define IF_GAUSS 0 // Gaussian profile 
#define IF_EXP 0 // Exponential profile 

#define DIMENSION 1 // Dimension of the ring 
#define IF_PERIODIC 1 // periodic gaussian 

#define IF_SPEC 0 // sqrt(K) specific connections 

#define IF_MATRIX 0 // save Cij matrix 
#define IF_SPARSEVEC 1 // save sparse vectors 

#endif
