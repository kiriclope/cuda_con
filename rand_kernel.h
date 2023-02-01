#include <curand.h> 
#include <curand_kernel.h> 

curandState *dev_states ; 
__device__ double rand_number[N_NEURONS] ;

__global__ void setup_kernel(curandState *state) { 

  unsigned long id = threadIdx.x + blockIdx.x * blockDim.x ; 
  
  if(id < N_NEURONS)
    curand_init(clock64(), id, 0, &state[id]) ; 
  
  if(id<2) { 
    cuPrintf("currand init: ") ; 
    cuPrintf("%d", state[id]) ; 
  }
  
}

__device__ double unif_dist(curandState *state, unsigned long i_neuron) { 
  
  double randNumber= 0.0 ; 
  if(i_neuron < N_NEURONS) { 
    /* save state in local memory for efficiency */ 
    curandState localState = state[i_neuron] ;
    randNumber = curand_uniform(&localState) ; 
    state[i_neuron] = localState ; 
  } 
  return randNumber ; 
}
