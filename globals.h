#ifndef __GLOBALS__
#define __GLOBALS__

using namespace:: std ; 

#define n_pop 2 
#define N_NEURONS (unsigned long) 40000 
#define K (float) 2000 
#define sqrt_K (float) sqrt(K) 

#define E_frac (float) 0.8

const float n_frac[2] = { (float) E_frac, (float) round( (1.0 - E_frac)*100.0) / (float) 100.0 } ; 

// string path = "/homecentral/alexandre.mahrach/IDIBAPS/connectivity/" ;
string path = "../../cpp/model/connectivity/" ;

#define IF_CON_DIR 0 
#define SEED_CON (float) 3.0 

#define IF_REMAP_SQRT_K 0 

unsigned long i, j, i_neuron ;
int pre_pop, post_pop ; 
int n_pref ; 

unsigned long n_per_pop[n_pop] ; 
unsigned long cum_n_per_pop[n_pop+1] ; 

int which_pop[N_NEURONS] ; 
float K_over_Na[n_pop] ; 

#define IF_SAVE_CON_VEC 0 // save Cij matrix 
#define IF_SAVE_SPARSE_REP 1 

////////////////////////////////// 
// structure 
////////////////////////////////// 

int IF_STRUCTURE ; 
const float IS_STRUCT_SYN[4] = {1.0, 1.0, 1.0, 1.0} ; // WARNING check that it is the same in cuda globals
__device__ const float DEV_IS_STRUCT_SYN[4] = {1.0, 1.0, 0.85, 1.0} ;
/* __device__ const float DEV_IS_STRUCT_SYN[4] = {1.0, 1.0, 0.65, 1.0} ;  */

#define IF_RING 0 
#define IF_SPEC 0 
#define IF_GAUSS 0 

#define KAPPA (float) 4.7 
#define KAPPA_E (float) KAPPA 
#define KAPPA_I (float) .125 

#define SIGMA (float) 1.0

#define IF_LOW_RANK 1 
#define IF_GEN_KSI 0 

#define DUM (int) (0.8 * N_NEURONS) 
float ksi[DUM], ksi_1[DUM] ;

#define RANK 2 
#define FIX_KSI_SEED 1
#define SEED_KSI 2 
#define KAPPA_FRAC 1.0
string ksi_path ;

#endif
